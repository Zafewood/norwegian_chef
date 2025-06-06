from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

import argparse
import pygame
import numpy as np
import random
from datetime import datetime
from distutils.util import strtobool

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
        return self.net(x).squeeze(-1)

def parse_args():

    parser = argparse.ArgumentParser(description="Train REINFORCE agent on Overcooked")

    parser.add_argument("--train_model", type=str, default="reinforce_baseline")
    parser.add_argument("--num_training_episodes", type=int, default=5000)
    parser.add_argument("--episode_length", type=int, default=400)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-3)
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--shape_factor", type=float, default=1)
    parser.add_argument("--layout", type=str, default="cramped_room", help=" cramped_room - large_room - asymmteric_advantages - centre_pots")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],help="Device to run on (cpu, cuda, or mps)")

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--visual", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
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

def run_episode(env, policy, valuenet, gamma, shape_factor, visual, device):
    global rendering_flag
    state = env.reset()
    state = env.mdp.get_standard_start_state()
    states = []
    log_probs = []
    rewards = []
    sparse_rewards = []
    returns = []
    values = []
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
                screen.blit(visual.render_state(state, grid=env.mdp.terrain_mtx), (0, 0))
                pygame.display.flip()
                clock.tick(120)

        state_tensor = torch.tensor(env.featurize_state_mdp(state)[0], dtype=torch.float32).to(device) #get the first player's observation as a tensor
        states.append(state_tensor)  # Store the state tensor for later use
        logits = policy(state_tensor) # forward pass raw scores through the policy network
        dist = Categorical(logits=logits)  # Create a categorical distribution from the logits
        action = dist.sample()  # Sample an action from the distribution 
        log_prob = dist.log_prob(action)  # Get the log probability of the action
        log_probs.append(log_prob)  # Store the log probability of the action
        
        entropy = dist.entropy()
        entropies.append(entropy)

        values.append(valuenet(state_tensor))

        joint_action = (Action.ALL_ACTIONS[action.item()], random.choice(Action.ALL_ACTIONS))
        next_obs, timestep_sparse_reward, done, info = env.step(joint_action)
        
        #includes shaped rewards in the reward signal
        shaped_reward = sum(info.get("shaped_r_by_agent", [0, 0]))
        dense_reward = shaped_reward/shape_factor + timestep_sparse_reward
        # Decomment to print for each soup made
        # if timestep_sparse_reward > 0:
        #     print(f"Soup made at step {step} with reward: {timestep_sparse_reward}")

        rewards.append(dense_reward)
        sparse_rewards.append(timestep_sparse_reward)
        state = next_obs
        step += 1

    total_raw_reward = sum(rewards)
    soup_reward = sum(sparse_rewards)

    #Compute the return backtracking the episode
    G = 0
    for r in reversed(rewards):
        G = G * gamma + r
        returns.insert(0,G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    #Normalization - had this for the longest time, not good for the sparse rewards
    #returns = (returns - returns.mean()) / (returns.std() + 1e-8) #normalize returns
    return log_probs, returns, entropies, total_raw_reward, soup_reward, states


def update(policy_optimizer, value_optimizer, policy, valuenet, log_probs, returns, entropies, global_step, states, device): #writer
    states_tensor = torch.stack(states).to(device)  # (T, obs_dim)
    values = valuenet(states_tensor)  # (T,)

    advantages = returns - values.detach()  # Stop gradients to the critic as baseline
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #normalize the advantages
    
    policy_loss = (-torch.stack(log_probs) * advantages).mean()

    entropy = torch.stack(entropies).mean()
    entropy_bonus = 0.01 * entropy

    value_loss = nn.functional.mse_loss(values, returns)

    # -Update policy network-
    policy_optimizer.zero_grad()
    (policy_loss - entropy_bonus).backward()
    policy_optimizer.step()

    # -Update value network-
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return policy_loss.item(), value_loss.item()
    
if __name__ == "__main__":
    args = parse_args()
    env = load_env(args.episode_length, args.layout)
    env.reset()

    print(f"Running with {args}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Visualizing
    if args.visual:
        rendering_flag = False  # Global flag to control rendering
        visual = StateVisualizer(window_fps=60)
        pygame.init()
        screen = pygame.display.set_mode((env.mdp.width * visual.tile_size, env.mdp.height * visual.tile_size))
        clock = pygame.time.Clock()

    start_state = env.mdp.get_standard_start_state()
    obs = env.featurize_state_mdp(start_state)
    obs_dim = len(obs[0]) #for first player
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

    policy = PolicyNetwork(obs_dim, act_dim).to(args.device)
    valuenet = ValueNetwork(obs_dim).to(args.device)

    policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_learning_rate)
    value_optimizer = optim.Adam(valuenet.parameters(), lr=args.value_learning_rate)

    print("Training starts...")
    for episode in range(args.num_training_episodes):
        log_probs, returns, entropies, total_raw_reward, soup_reward, states = run_episode(env, policy, valuenet, args.gamma, args.shape_factor, args.visual, args.device)
        policy_loss, value_loss = update(policy_optimizer, value_optimizer, policy, valuenet, log_probs, returns, entropies, global_step, states, args.device) #writer
        total_reward = returns[0].cpu().item()
        episode_rewards.append(total_raw_reward)
        episode_soup_rewards.append(soup_reward)
        #Decomment below to print each episode
        #print(f"Episode {episode} | Return: {total_reward:.2f} | Sparse reward: {soup_reward} | Total raw reward: {total_raw_reward:.2f}")
        if episode % 100  == 0 and episode > 0:
            average = np.mean(episode_rewards[-100:])
            averages_soup = np.mean(episode_soup_rewards[-100:])
            print(f"Episode {episode-100} to {episode} finished with the average raw reward: {average:.2f} and average soup score of {averages_soup:.2f}")

        if args.track:
            writer.add_scalar("Return/Total", total_reward, global_step)
            writer.add_scalar("Raw Reward/Total", total_raw_reward, global_step)
            writer.add_scalar("Soup reward/Total", soup_reward, global_step)
            writer.add_scalar("Loss/Policy", policy_loss, global_step)
            writer.add_scalar("Loss/Value", value_loss, global_step)   
            writer.add_scalar("Loss/Total", policy_loss + value_loss, global_step)     
        global_step += 1
        if args.save:
            if total_raw_reward > best_reward:
                best_reward = total_raw_reward
                #Decomment below to print for each time a new policy is being saved
                #print(f"New best reward: {best_reward:.5f} at episode {episode}")
                torch.save(policy.state_dict(), f"trained/reinforce_baseline_{args.layout}{datetime.now().strftime('%m-%d_%H-%M')}.pth")

    if args.visual:
        pygame.quit()
    if args.track:
        writer.close()