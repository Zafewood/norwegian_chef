from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

#Visualization
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pygame
import cv2
import numpy as np

#Seeding
import random

#Network
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

def load_env():
    base_mdp = OvercookedGridworld.from_layout_name("cramped_room", rew_shaping_params = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 5,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
})
    env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=600)

    return env

def run_episode(env, policy, gamma, episode):
    state = env.reset()
    state = env.mdp.get_standard_start_state()
    log_probs = []
    rewards = []
    returns = []
    entropies = []
    done = False
    step = 0

    while True:
        # if episode > 200:
            # rstate = visual.render_state(state, grid=env.mdp.terrain_mtx)
            # rstate = pygame.surfarray.array3d(rstate)
            # rstate = np.transpose(rstate, (1, 0, 2))
            # rstate = cv2.cvtColor(rstate, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Rendering: ",rstate)
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            #     break
            # clock.tick(10000) 

        state_tensor = torch.tensor(env.featurize_state_mdp(state)[0], dtype=torch.float32) #get the first player's observation as a tensor
        logits = policy(state_tensor) # forward pass raw scores through the policy network
        dist = Categorical(logits=logits)  # Create a categorical distribution from the logits
        action = dist.sample()  # Sample an action from the distribution 
        log_prob = dist.log_prob(action)  # Get the log probability of the action
        log_probs.append(log_prob)  # Store the log probability of the action
        
        entropy = dist.entropy()
        entropies.append(entropy)


        joint_action = (Action.ALL_ACTIONS[action.item()], random.choice(Action.ALL_ACTIONS))
        next_obs, timestep_sparse_reward, done, info = env.step(joint_action)
        
        #if next state is a finish, end the episode
        if done:
            break

        #include shaped rewards in the reward signal
        shaped_reward = sum(info.get("shaped_r_by_agent", [0, 0])) # or /100
        dense_reward = shaped_reward + timestep_sparse_reward
        if timestep_sparse_reward > 0:
            print(f"Soup made at step {step} with reward: {timestep_sparse_reward}")

        rewards.append(dense_reward)
        state = next_obs
        step += 1

    total_raw_reward = sum(rewards)

    #Compute the return backtracking the episode
    G = 0
    for r in reversed(rewards):
        G = G * gamma + r
        returns.insert(0,G)
    returns = torch.tensor(returns, dtype=torch.float32)  # Convert returns to a tensor

    #noramlization - this is maybe not ideal for the sparse rewards
    returns = (returns - returns.mean()) / (returns.std() + 1e-8) #normalize returns
    return log_probs, returns, entropies, total_raw_reward


def update_policy(optimizer, policy, log_probs, returns, entropies, global_step): #writer
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G
    loss = loss / len(log_probs)
    entropy = torch.stack(entropies).mean()
    loss -= 0.01 * entropy
    
    optimizer.zero_grad()
    loss.backward() 
    #for name, param in policy.named_parameters():
        # writer.add_histogram(f"{name}.grad", param.grad, global_step)
        # writer.add_histogram(f"{name}.weight", param.data, global_step)
    optimizer.step()

    return loss
    

if __name__ == "__main__":
    env = load_env()
    env.reset()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    #visualizing
    visual = StateVisualizer(window_fps=60)
    pygame.init()
    clock = pygame.time.Clock() 

    start_state = env.mdp.get_standard_start_state()
    obs = env.featurize_state_mdp(start_state)
    obs_dim = len(obs[0]) #for first player
    act_dim = len(Action.ALL_ACTIONS)
    
    #logging
    episode_rewards = []
    #writer = SummaryWriter()
    global_step = 0

    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(5000):
        log_probs, returns, entropies, total_raw_reward = run_episode(env, policy, 0.999, episode)
        loss = update_policy(optimizer, policy, log_probs, returns, entropies, global_step) #writer
        total_reward = returns[0].item()
        print(f"Episode {episode} finished with the return: {total_reward:.5f} and the total raw reward: {total_raw_reward:.5f}")
        # writer.add_scalar("Reward/Total", total_reward, global_step)
        # writer.add_scalar("Loss/Policy", loss.item(), global_step)
        # writer.add_scalar("Raw Reward/Total", total_raw_reward, global_step)
        #print(f"Episode {episode} Reward: {total_reward:.2f}")
        # episode_rewards.append(total_reward)
        global_step += 1
    torch.save(policy.state_dict(), "policy_checkpoint.pth")

    # plt.plot(episode_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Training Progress")
    # plt.show()

    cv2.destroyAllWindows()
    pygame.quit()
    #writer.close()