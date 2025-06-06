from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

import argparse
import pygame
import numpy as np
import random
from datetime import datetime
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Output shape: (batch,)

def parse_args():

    parser = argparse.ArgumentParser(description="Train REINFORCE agent on Overcooked")

    parser.add_argument("--train_model", type=str, default="reinforce_baseline")
    parser.add_argument("--num_training_episodes", type=int, default=8000)
    parser.add_argument("--episode_length", type=int, default=400)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-3)
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--shape_factor", type=float, default=1)
    parser.add_argument("--layout", type=str, default="cramped_room", help=" cramped_room - large_room - asymmteric_advantages - centre_pots")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],help="Device to run on (cpu, cuda, or mps)")

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--visual", type=bool, default=False)
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)
    return parser.parse_args()

def load_env(episode_length, layout):
    base_mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 5,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
})
    env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=episode_length)

    return env

def run_episode(env, a0_policy, a0_valuenet, a1_policy, a1_valuenet, gamma, shape_factor, visual, device):
    global rendering_flag
    state = env.reset()
    state = env.mdp.get_standard_start_state()
    a0_states = []
    a1_states = []
    log_probs = []
    rewards = []
    sparse_rewards = []
    returns = []
    entropies = []
    done = False
    step = 0

    while not done:

        if visual:
            pygame_event = pygame.event.get()
            if rendering_flag == False:
                for event in pygame_event:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        rendering_flag = True
                        print("Rendering enabled")
            
            if rendering_flag == True:
                for event in pygame_event:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        rendering_flag = False
                        pygame.event.clear()
                        print("Rendering disabled")
                screen.blit(visualizer.render_state(state, grid=env.mdp.terrain_mtx), (0, 0))
                pygame.display.flip()
                clock.tick(120)

        a0_obs = torch.tensor(env.featurize_state_mdp(state)[0], dtype=torch.float32).to(device) #get the first player's observation as a tensor
        a1_obs = torch.tensor(env.featurize_state_mdp(state)[1], dtype=torch.float32).to(device) #get the second player's observation as a tensor
        a0_states.append(a0_obs)   
        a1_states.append(a1_obs)   

        a0_logits = a0_policy(a0_obs)
        a0_dist = Categorical(logits=a0_logits)
        a0_action = a0_dist.sample()
        a0_log_prob = a0_dist.log_prob(a0_action)  # Get the log probability of the action

        a1_logits = a1_policy(a1_obs)
        a1_dist = Categorical(logits=a1_logits)
        a1_action = a1_dist.sample()
        a1_log_prob = a1_dist.log_prob(a1_action)

        log_probs.append((a0_log_prob, a1_log_prob))

        a0_entropy = a0_dist.entropy()
        a1_entropy = a1_dist.entropy()
        entropies.append((a0_entropy,a1_entropy))

        joint_action = (Action.ALL_ACTIONS[a0_action.item()], Action.ALL_ACTIONS[a1_action.item()])
        state, sparse_rew, done, info = env.step(joint_action)

        # shaped_reward_1 = info["shaped_r_by_agent"][0]
        # shaped_reward_2 = info["shaped_r_by_agent"][1]
        
        #include shaped rewards in the reward signal
        shaped_reward = sum(info.get("shaped_r_by_agent", [0, 0]))
        dense_reward = shaped_reward/shape_factor + sparse_rew
        # if sparse_rew > 0:
        #     print(f"Soup made at step {step} with reward: {sparse_rew}")

        rewards.append(dense_reward)
        sparse_rewards.append(sparse_rew)
        step += 1

    total_raw_reward = sum(rewards)
    soup_reward = sum(sparse_rewards)

    #Compute the return backtracking the episode
    G = 0
    for r in reversed(rewards):
        G = G * gamma + r
        returns.insert(0,G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)  # Convert returns to a tensor

    #noramlization - this is maybe not ideal for the sparse rewards
    #returns = (returns - returns.mean()) / (returns.std() + 1e-8) #normalize returns
    return log_probs, returns, entropies, total_raw_reward, soup_reward, a0_states, a1_states


def update(policy_optimizer, value_optimizer, valuenet, log_probs, returns, entropies, global_step, states, device): #writer
    states_tensor = torch.stack(states).to(device)  # (T, obs_dim)
    values = valuenet(states_tensor)  # (T,)

    advantages = returns - values.detach()  # Stop gradients to critic
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #normalize advantages
    
    policy_loss = (-torch.stack(log_probs) * advantages).mean()
    value_loss = nn.functional.mse_loss(values, returns)

    entropy = torch.stack(entropies).mean()
    entropy_bonus = 0.01 * entropy


    # === Update networks ===
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()

    total_loss = value_loss + (policy_loss - entropy_bonus)
    total_loss.backward() #combine them to avoid backtracking twice in same update step

    policy_optimizer.step()
    value_optimizer.step()

    return
    
if __name__ == "__main__":
    args = parse_args()
    env = load_env(args.episode_length, args.layout)
    env.reset()

    print(f"Running with {args}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #visualizing
    if args.visual:
        rendering_flag = False  # Global flag to control whether to render
        visualizer = StateVisualizer(window_fps=60)
        pygame.init()
        screen = pygame.display.set_mode((env.mdp.width * visualizer.tile_size, env.mdp.height * visualizer.tile_size))
        clock = pygame.time.Clock()

    start_state = env.mdp.get_standard_start_state()
    obs = env.featurize_state_mdp(start_state)
    obs_dim = len(obs[0])
    act_dim = len(Action.ALL_ACTIONS)
    
    episode_rewards = []
    episode_soup_rewards = []
    best_reward = -float("inf")

    if args.track:
        run_name = f"{args.train_model}_{args.layout}{datetime.now().strftime('%m-%d_%H-%M-%S-%f')}"
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        writer.add_text(
        "hyperparameters",
        f"|parameter|value|\n|-|-|\n|train_model|{args.train_model}|\n|num_training_episodes|{args.num_training_episodes}|\n|episode_length|{args.episode_length}|\n|policy_learning_rate|{args.policy_learning_rate}|\n|value_learning_rate|{args.value_learning_rate}|\n|gamma|{args.gamma}|\n|shape_factor|{args.shape_factor}|\n|layout|{args.layout}|\n|seed|{args.seed}|\n")
    global_step = 0

    a0_policy = PolicyNetwork(obs_dim, act_dim).to(args.device)
    a0_valuenet = ValueNetwork(obs_dim).to(args.device)
    a1_policy = PolicyNetwork(obs_dim, act_dim).to(args.device)
    a1_valuenet = ValueNetwork(obs_dim).to(args.device)

    a0_policy_optimizer = optim.Adam(a0_policy.parameters(), lr=args.policy_learning_rate)
    a0_value_optimizer = optim.Adam(a0_valuenet.parameters(), lr=args.value_learning_rate)
    a1_policy_optimizer = optim.Adam(a1_policy.parameters(), lr=args.policy_learning_rate)
    a1_value_optimizer = optim.Adam(a1_valuenet.parameters(), lr=args.value_learning_rate)

    print("Training starts...")
    for episode in range(args.num_training_episodes):
        log_probs, returns, entropies, total_raw_reward, soup_reward, a0_states, a1_states = run_episode(env, a0_policy, a0_valuenet, a1_policy, a1_valuenet, args.gamma, args.shape_factor, args.visual, args.device)
        a0_log_probs = [lp[0] for lp in log_probs]
        a1_log_probs = [lp[1] for lp in log_probs]
        a0_entropies = [e[0] for e in entropies]
        a1_entropies = [e[1] for e in entropies]

        update(a0_policy_optimizer, a0_value_optimizer, a0_valuenet, a0_log_probs, returns, a0_entropies, global_step, a0_states, args.device)
        update(a1_policy_optimizer, a1_value_optimizer, a1_valuenet, a1_log_probs, returns, a1_entropies, global_step, a1_states, args.device)

        total_reward = returns[0].cpu().item()
        episode_rewards.append(total_raw_reward)
        episode_soup_rewards.append(soup_reward)
        #Uncomment below to print each episode
        #print(f"Episode {episode} | Return: {total_reward:.2f} | Sparse reward: {soup_reward} | Total raw reward: {total_raw_reward:.2f}")
        if episode % 100  == 0 and episode > 0:
            average = np.mean(episode_rewards[-100:])
            averages_soup = np.mean(episode_soup_rewards[-100:])
            print(f"[policy_lr:{args.policy_learning_rate}, value_lr:{args.value_learning_rate}, gamma:{args.gamma}] Episode {episode-100} to {episode} finished with the average raw reward: {average:.2f} and average soup score of {averages_soup:.2f}")

        if args.track:
            writer.add_scalar("Return/Total", total_reward, global_step)
            writer.add_scalar("Raw Reward/Total", total_raw_reward, global_step)
            writer.add_scalar("Soup reward/Total", soup_reward, global_step)    
        global_step += 1
        if args.save:
            if total_raw_reward > best_reward:
                best_reward = total_raw_reward
                #Uncomment to print each time a new policy is saved
                #print(f"New best reward: {best_reward:.5f} at episode {episode}")
                torch.save(a0_policy.state_dict(), f"trained/reinforce_baseline_multi_{args.layout}_a0.pth")                
                torch.save(a1_policy.state_dict(), f"trained/reinforce_baseline_multi_{args.layout}_a1.pth")


    if args.visual:
        pygame.quit()
    if args.track:
        writer.close()