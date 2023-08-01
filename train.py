import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import malmo.MalmoPython as MalmoPython
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Constants
xml_file = "zombie_fight.xml"
episodes = 5000
baseline = True
video_shape = (480, 640, 3)
input_shape = (3, 480, 640)
save = True
load_last_trained = True
load_model = "models/1zombie_fighter_dqn_ep300.pth"
start_episode = 1001
max_steps_per_episode = 1000
running_average_length = episodes // 20
num_zombies = 1
agent_cfg = {
    "alpha": 0.0005,
    "gamma": 0.85,
    "batch_size": 64,
    "epsilon": 1,
    "epsilon_decay": 0.992,
    "epsilon_min": 0.05,
    "copy_period": 300,
    "mem_size": 5000,
    "n_actions": 5
}


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),  # Change kernel_size to 4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class ZombieFighterAgent(object):
    def __init__(self, agent_cfg, input_shape, num_actions, model_path=None):
        self.cfg = agent_cfg
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.action_space = np.arange(self.num_actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_eval = DQN(self.input_shape, self.num_actions).to(self.device)
        if model_path is not None:
            self.q_eval.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        self.q_target = DQN(self.input_shape, self.num_actions).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.cfg["alpha"])
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(self.cfg["mem_size"], self.input_shape)

        self.epsilon = self.cfg["epsilon"]
        self.step_count = 0

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.cfg["n_actions"])
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.cfg["batch_size"]:
            return

        self.optimizer.zero_grad()

        max_mem = self.memory.mem_cntr if self.memory.mem_cntr < self.cfg["mem_size"] else self.cfg["mem_size"]
        batch = np.random.choice(max_mem, self.cfg["batch_size"], replace=False)
        batch_index = np.arange(self.cfg["batch_size"], dtype=np.int32)

        states, actions, rewards, states_, dones = self.memory.sample(self.cfg["batch_size"])

        # Convert the arrays to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.q_eval.device)
        actions = torch.tensor(actions, dtype=torch.long).clamp(0, self.cfg["n_actions"] - 1).to(self.q_eval.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.q_eval.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.q_eval.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.q_eval.device)

        # print(states_)
        q_eval = self.q_eval(states)
        q_target = self.q_target(states_)

        q_target[dones] = 0.0
        indices = np.stack([batch_index, actions.cpu().clamp(0, self.num_actions - 1)], axis=1)
        # print("q_eval shape:", q_eval.shape)
        # print("indices:", indices)
        # print("indices shape:", indices.shape)
        q_eval = q_eval.gather(1, torch.tensor(indices[:, 1], dtype=torch.long).unsqueeze(1).to(self.q_eval.device))

        q_next = torch.max(q_target, dim=1)[0]
        q_target = rewards + self.cfg["gamma"] * q_next
        # Add unsqueeze to match the dimensions
        q_target = q_target.unsqueeze(1)
        loss = self.loss_fn(q_target, q_eval)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.cfg["epsilon_min"], self.cfg["epsilon_decay"] * self.epsilon)

        if self.step_count % self.cfg["copy_period"] == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        self.step_count += 1


# Add the ReplayBuffer class for handling the agent's experience
class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        # print(f"state shape in store method: {state.shape}")
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


def take_action(env, action):
    global previous_zombie_killed, previous_zombie_damage, previous_time, previous_damage_taken, count_zombie_killed
    # Send the command to the Malmo environment
    if action == 0:
        env.sendCommand("move 1")
    elif action == 1:
        env.sendCommand("turn -1")
    elif action == 2:
        env.sendCommand("turn 1")
    elif action == 3:
        env.sendCommand("attack 1")
    elif action == 4:
        env.sendCommand("move -1")

    # Get the new state, reward, and done
    world_state = env.getWorldState()
    if world_state.number_of_rewards_since_last_state > 0:
        reward = world_state.rewards[0].getValue()
    else:
        reward = 0

    # Loop until a valid frame is received
    state_ = None
    start = time.time()
    while state_ is None and time.time() - start < 3:
        world_state = env.getWorldState()
        if len(world_state.video_frames) > 0:  # Check if video_frames is not empty
            state_ = process_video(world_state.video_frames[0])
        time.sleep(0.1)
        if time.time() - start > 3:
            break
    #print(world_state.is_mission_running)
    # Check if the agent is dead
    if world_state.is_mission_running is True:
        try:
            obs = json.loads(world_state.observations[-1].text)
        except IndexError:
            return reward, state_, True, False
        print(f"obs: {obs}")  # Add this line to print the observations
        if previous_zombie_damage == -1:
            previous_zombie_damage = obs["DamageDealt"]
        if previous_zombie_killed == -1:
            previous_zombie_killed = obs["MobsKilled"]
        if previous_time == -1:
            previous_time = obs["TimeAlive"]
        if previous_damage_taken == -1:
            previous_damage_taken = obs["DamageTaken"]
        if "Life" in obs and obs["Life"] <= 0:
            done = True
        else:
            done = False

        zombie_damage = obs["DamageDealt"]
        zombie_killed = obs["MobsKilled"]
        current_time = obs["TimeAlive"]
        damage_taken = obs["DamageTaken"]

        if zombie_damage > previous_zombie_damage:
            reward += 100
            previous_zombie_damage = zombie_damage

        if zombie_killed > previous_zombie_killed:
            reward += 500
            count_zombie_killed += 1
            previous_zombie_killed = zombie_killed

        if current_time > previous_time:
            reward -= (current_time - previous_time) * 0.1
            previous_time = current_time

        if damage_taken > previous_damage_taken:
            reward -= (damage_taken - previous_damage_taken) * 0.3
            previous_damage_taken = damage_taken
    else:
        done = True  # Set done to True if the mission is not running

    if count_zombie_killed == num_zombies:
        all_zombies_killed = True
    else:
        all_zombies_killed = False
    return reward, state_, done, all_zombies_killed


def process_video(frame):
    # Convert the frame to the desired input_shape
    img = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(*video_shape)
    img = img[:, :, :3]  # Remove the depth channel
    img = cv2.resize(img, (input_shape[2], input_shape[1]))  # Swap input_shape[1] and input_shape[2]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Move channels to the first dimension
    return img


fout = open('results.csv', 'w')
# Load and process the XML
with open(xml_file, "r") as f:
    xml = f.read()
    xml = xml.replace("{{width}}", str(input_shape[2]))
    xml = xml.replace("{{height}}", str(input_shape[1]))
# Initialize the agent and environment
agent = ZombieFighterAgent(agent_cfg, input_shape, num_actions=agent_cfg["n_actions"], model_path=load_model)

env = MalmoPython.AgentHost()
# Run the training loop
for episode in range(start_episode, episodes + 1):
    while True:
        try:
            # Attempt to start the mission:
            env.startMission(MalmoPython.MissionSpec(xml, True), MalmoPython.MissionRecordSpec())
            break
        except RuntimeError as e:
            print("waiting for previous mission to close")
            time.sleep(1)
    print(f"Episode {episode}: Mission started.")
    world_state = env.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = env.getWorldState()

    # Initialize previous damage values for the new mission
    previous_zombie_damage = -1
    previous_zombie_killed = -1
    previous_time = -1
    previous_damage_taken = -1
    count_zombie_killed = 0

    # Reset environment
    state = np.zeros(input_shape, dtype=np.float32)
    total_reward = 0
    done = False
    while not done:
        world_state = env.getWorldState()
        if world_state.number_of_video_frames_since_last_state > 0:
            frame = process_video(world_state.video_frames[-1])
            frame = torch.tensor(frame).unsqueeze(0)  # Add a batch dimension
            action = agent.choose_action(frame)
            reward, state_, done, all_zombies_killed = take_action(env, action)
            agent.store_transition(state, action, reward, state_, done)

            agent.learn()
            state = state_
            total_reward += reward
            if done or all_zombies_killed:
                break


    print(f"Episode {episode} - Total reward: {total_reward}")
    if save and episode % 100 == 0:
        torch.save(agent.q_eval.state_dict(), f"models/zombie_fighter_dqn_ep{episode}.pth")
    fout.write(str(episode) + ',' + str(total_reward) + '\n')
    fout.flush()

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = env.getWorldState()
        if count_zombie_killed == num_zombies:
            break
    time.sleep(3)

fout.close()
