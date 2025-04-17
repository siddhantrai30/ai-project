# ai-project
import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import imageio
import glob
import io
import base64
from IPython.display import HTML, display

# Hyperparameters
number_episodes = 2000
timesteps = 1000
epsilon_initial = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
discount_factor = 0.99
replay_buffer_size = 10000
minibatch_size = 64
interpolation_perameter = 0.005

# Create environment
env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Replay Memory Class
class ReplayMemory(object):
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones

# Neural Network (Q-Network)
class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent Class
class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_Qnetwork = Network(state_size, action_size).to(self.device)
        self.target_Qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_Qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_Qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_Qnetwork(state)
        self.local_Qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_Qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_Qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_Qnetwork, self.target_Qnetwork, interpolation_perameter)
    
    def soft_update(self, local_model, target_model, interpolation_perameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_perameter * local_param.data + (1.0 - interpolation_perameter) * target_param.data)

# Train the agent
score_100_episodes = deque(maxlen=100)
agent = Agent(state_size, action_size)

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(timesteps):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    score_100_episodes.append(score)
    epsilon = max(epsilon_final, epsilon_decay * epsilon)
    print(f'\rEpisode {episode}\tAverage Score: {np.mean(score_100_episodes):.2f}', end="")
    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(score_100_episodes):.2f}')
    if np.mean(score_100_episodes) >= 200.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(score_100_episodes):.2f}')
        torch.save(agent.local_Qnetwork.state_dict(), 'checkpoint.pth')
        break

# Generate and show video of the trained agent
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state, epsilon)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=f'''<video alt="test" autoplay loop controls style="height: 400px;">
                              <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
                             </video>'''))
    else:
        print("Could not find video")

# Show the video of agent's performance
show_video_of_model(agent, 'LunarLander-v3')
show_video()
