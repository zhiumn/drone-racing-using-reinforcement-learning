import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from ppo import PPO
import glob
import json

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


#################################### Finding Best Checkpoint ###################################
def evaluate_checkpoint(checkpoint_path, env, ppo_agent, n_eval_episodes=5, verbose=False):
    """
    評估單個checkpoint的性能
    """
    # 載入checkpoint
    ppo_agent.load(checkpoint_path)
    
    total_rewards = []
    gates_passed_list = []
    success_count = 0
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        done = False
        steps = 0
        max_steps = env.EPISODE_LEN_SEC * env.CTRL_FREQ
        
        while not done and steps < max_steps:
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        # 記錄通過的門框數
        gates_passed = sum(env.passing_flag) if hasattr(env, 'passing_flag') else 0
        gates_passed_list.append(gates_passed)
        total_rewards.append(episode_reward)
        
        # 判斷是否成功（通過3個門框）
        if gates_passed >= 3:
            success_count += 1
        
        # 清除buffer
        ppo_agent.buffer.clear()
        
        if verbose:
            print(f"    Episode {episode+1}: Reward={episode_reward:.2f}, Gates={gates_passed}/3")
    
    # 計算統計指標
    metrics = {
        'checkpoint': checkpoint_path,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'max_reward': np.max(total_rewards),
        'avg_gates': np.mean(gates_passed_list),
        'max_gates': np.max(gates_passed_list),
        'success_rate': success_count / n_eval_episodes,
        'success_count': success_count
    }
    
    return metrics


def find_best_checkpoint(log_dirs=["log_dir/racing2"], n_eval_episodes=5):
    """
    搜尋所有checkpoint並找出最佳的
    """
    print("============================================================================================")
    print("Searching for the best checkpoint...")
    print("============================================================================================")
    
    # 收集所有checkpoint文件
    all_checkpoints = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            checkpoints = glob.glob(os.path.join(log_dir, "*_ppo_drone.pth"))
            all_checkpoints.extend(checkpoints)
            print(f"Found {len(checkpoints)} checkpoints in {log_dir}")
    
    if not all_checkpoints:
        print("No checkpoints found!")
        return None
    
    print(f"\nTotal checkpoints to evaluate: {len(all_checkpoints)}")
    print("--------------------------------------------------------------------------------------------")
    
    # 初始化環境和PPO agent
    DEFAULT_GUI = False  # 評估時不顯示GUI以加快速度
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    
    env = FlyThruGateAvitary(
        gui=DEFAULT_GUI,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        record=False
    )
    
    # PPO參數
    state_dim = 36
    action_dim = 4
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    action_std = 0.1
    
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    # 評估所有checkpoints
    results = []
    best_score = -float('inf')
    best_checkpoint = None
    
    for i, checkpoint_path in enumerate(all_checkpoints, 1):
        print(f"\n[{i}/{len(all_checkpoints)}] Evaluating: {os.path.basename(checkpoint_path)}")
        
        try:
            metrics = evaluate_checkpoint(
                checkpoint_path, 
                env, 
                ppo_agent, 
                n_eval_episodes=n_eval_episodes,
                verbose=True
            )
            
            # 計算綜合分數（可調整權重）
            score = (
                metrics['success_rate'] * 100.0 +     # 成功率最重要
                metrics['avg_gates'] * 10.0 +         # 平均通過門框數
                metrics['avg_reward'] * 0.01          # 平均獎勵
            )
            metrics['score'] = score
            
            results.append(metrics)
            
            print(f"    Summary: Success Rate={metrics['success_rate']*100:.1f}%, "
                  f"Avg Gates={metrics['avg_gates']:.2f}, "
                  f"Avg Reward={metrics['avg_reward']:.2f}, "
                  f"Score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint_path
                print(f"    🏆 New best checkpoint!")
        
        except Exception as e:
            print(f"    ❌ Error evaluating {checkpoint_path}: {e}")
            continue
    
    env.close()
    
    # 排序並顯示結果
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 保存評估結果
    output_file = "checkpoint_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {output_file}")
    
    # 顯示前5名
    print("\n============================================================================================")
    print("TOP 5 CHECKPOINTS:")
    print("============================================================================================")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {os.path.basename(result['checkpoint'])}")
        print(f"   Success Rate: {result['success_rate']*100:.1f}%")
        print(f"   Avg Gates: {result['avg_gates']:.2f}")
        print(f"   Max Gates: {result['max_gates']}")
        print(f"   Avg Reward: {result['avg_reward']:.2f}")
        print(f"   Score: {result['score']:.2f}")
        print()
    
    if best_checkpoint:
        print("============================================================================================")
        print(f"🏆 BEST CHECKPOINT: {best_checkpoint}")
        print("============================================================================================")
    
    return best_checkpoint


#################################### Testing ###################################
def test(checkpoint_path=None, auto_find_best=True):
    print("============================================================================================")
    
    ################## hyperparameters ##################
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving
    
    render = True               # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames
    
    total_test_episodes = 10    # total num of testing episodes
    
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic
    
    #####################################################
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False
    
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    
    # 如果需要自動尋找最佳checkpoint
    if auto_find_best and checkpoint_path is None:
        # 可以指定要搜尋的目錄
        search_dirs = ["log_dir/racing2", "log_dir", "log_dir/6", "log_dir/5"]
        checkpoint_path = find_best_checkpoint(
            log_dirs=search_dirs,
            n_eval_episodes=3  # 評估時使用較少的episodes以加快速度
        )
        
        if checkpoint_path is None:
            print("No suitable checkpoint found!")
            return
    
    # 如果仍然沒有指定checkpoint，使用預設值
    if checkpoint_path is None:
        checkpoint_path = "log_dir/racing2/72045_ppo_drone.pth"
        print(f"Using default checkpoint: {checkpoint_path}")
    
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        print(filename)
        os.makedirs(filename+'/')
    
    env = FlyThruGateAvitary(
        gui=DEFAULT_GUI,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        record=DEFAULT_RECORD_VIDEO
    )
    
    # state space dimension
    state_dim = 36
    # action space dimension
    action_dim = 4
    
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    
    print(f"\nLoading network from: {checkpoint_path}")
    ppo_agent.load(checkpoint_path)
    
    print("--------------------------------------------------------------------------------------------")
    print("Starting test with best checkpoint...")
    print("--------------------------------------------------------------------------------------------")
    
    # 測試統計
    all_rewards = []
    all_gates = []
    success_count = 0
    
    for episode in range(total_test_episodes):
        obs, info = env.reset(seed=42 + episode, options={})
        ep_reward = 0
        start_time = datetime.now().replace(microsecond=0)
        start = time.time()
        
        print(f"\n--- Episode {episode + 1}/{total_test_episodes} ---")
        
        for i in range((env.EPISODE_LEN_SEC+20)*env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            sync(i, start, env.CTRL_TIMESTEP)
            if terminated or truncated:
                break
        
        # 記錄結果
        gates_passed = sum(env.passing_flag) if hasattr(env, 'passing_flag') else 0
        all_rewards.append(ep_reward)
        all_gates.append(gates_passed)
        
        if gates_passed >= 3:
            success_count += 1
            print(f'✅ Episode {episode + 1}: SUCCESS! Reward: {round(ep_reward, 2)}, Gates: {gates_passed}/3')
        else:
            print(f'❌ Episode {episode + 1}: Reward: {round(ep_reward, 2)}, Gates: {gates_passed}/3')
        
        # clear buffer
        ppo_agent.buffer.clear()
    
    env.close()
    
    # 顯示測試總結
    print("\n============================================================================================")
    print("TEST SUMMARY")
    print("============================================================================================")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Total Episodes: {total_test_episodes}")
    print(f"Success Rate: {success_count/total_test_episodes*100:.1f}% ({success_count}/{total_test_episodes})")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Gates Passed: {np.mean(all_gates):.2f}")
    print(f"Max Gates Passed: {np.max(all_gates)}")
    print("============================================================================================")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Specific checkpoint path to test')
    parser.add_argument('--auto-find', type=bool, default=True,
                       help='Automatically find the best checkpoint')
    parser.add_argument('--search-dirs', nargs='+', 
                       default=["log_dir/racing2", "log_dir"],
                       help='Directories to search for checkpoints')
    
    args = parser.parse_args()
    
    # 如果指定了checkpoint，則直接使用
    if args.checkpoint:
        test(checkpoint_path=args.checkpoint, auto_find_best=False)
    else:
        # 自動尋找最佳checkpoint
        test(checkpoint_path=None, auto_find_best=True)
