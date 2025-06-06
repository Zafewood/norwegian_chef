
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

# -Visualization and logging-
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pygame
import cv2
import numpy as np

# -Seeding-
import random

# -Network-
import torch
import torch.nn as nn
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

def load_env(episode_length=400, layout="cramped_room"):
    base_mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=episode_length)
    return env

def run_episode(env, policy, visual=False):
    global rendering_flag
    state = env.reset()
    state = env.mdp.get_standard_start_state()
    rewards = []
    sparse_rewards = []
    done = False
    step = 0

    while True:
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
        state_tensor = torch.tensor(env.featurize_state_mdp(state)[0], dtype=torch.float32) #get the first player's observation as a tensor
        with torch.no_grad():
            action_probs = torch.softmax(policy(state_tensor), dim=-1)
            action = torch.multinomial(action_probs, 1)

        joint_action = (Action.ALL_ACTIONS[action.item()], random.choice(Action.ALL_ACTIONS))
        next_obs, timestep_sparse_reward, done, info = env.step(joint_action)
        
        if done:
            break

        shaped_reward = sum(info.get("shaped_r_by_agent", [0, 0]))
        dense_reward = shaped_reward + timestep_sparse_reward
        rewards.append(dense_reward)
        sparse_rewards.append(timestep_sparse_reward)
        state = next_obs
        step += 1

    total_raw_reward = sum(rewards)
    soup_reward = sum(sparse_rewards)

    return total_raw_reward, soup_reward


if __name__ == "__main__":
    # -Hyperparameters-
    test_model = "INSERT MODEL NAME (from the .pth file)"
    num_testing_episodes = 500
    episode_length = 400
    layout = "cramped_room"
    seed = 65
    visual = True
    track = True

    env = load_env(episode_length, layout)
    env.reset()
    run_name = f"test_{test_model}"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if visual:
        rendering_flag = False
        visual = StateVisualizer(window_fps=60)
        pygame.init()
        screen = pygame.display.set_mode((env.mdp.width * visual.tile_size, env.mdp.height * visual.tile_size))
        clock = pygame.time.Clock()

    start_state = env.mdp.get_standard_start_state()
    obs = env.featurize_state_mdp(start_state)
    obs_dim = len(obs[0])
    act_dim = len(Action.ALL_ACTIONS)
    
    episode_rewards = []
    global_step = 0
    if track:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")

    #Load policy
    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(torch.load(f"trained/{test_model}.pth"))
    policy.eval()

    for episode in range(1,num_testing_episodes+1):
        total_raw_reward, soup_reward = run_episode(env, policy, visual)
        print(f"Episode {episode} | Total raw reward: {total_raw_reward:.2f} | Soup reward: {soup_reward:.2f}")
        if track:
            writer.add_scalar("Raw Reward/Total", total_raw_reward, global_step)
            writer.add_scalar("Soup reward/Total", soup_reward, global_step)
        global_step += 1

    if visual:
        cv2.destroyAllWindows() 
        pygame.quit()
    if track:
        writer.close()