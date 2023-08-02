# Training a Reinforcement Learning Agent to Fight Zombies in Minecraft

The primary objective of this study is to develop a RL agent proficient in combating zombies within the Minecraft gaming environment. The agent is expected to navigate the game, learn from its experiences, and make informed decisions that optimize rewards while reducing penalties. In this paper, our goal is to create an agent capable of effectively eliminating zombies and surviving the game, thereby showcasing the potential of RL algorithms in intricate decision-making tasks set in dynamic environments. 

## System Design

### Environment

The environment is a confined, flat Minecraft world with dimensions predefined in the `zombie_fight.xml` file. It is illuminated with glowstones, providing visibility for the agent. A single zombie is placed within this environment, serving as the adversary for the agent.

### Agent

The agent is equipped with diamond armor and a diamond sword, and starts at a specific location within the environment. It operates in the survival mode, which means it is subject to damage and has a limited health and hunger.

The agent is implemented using a reinforcement learning (RL) approach. The RL model parameters are saved under the `models` folder. The training process is controlled by the `train.py` script.

## Training Process

The training process involves multiple episodes, where the agent interacts with the environment, learning from its actions and their consequences. The agent's objective is to maximize its cumulative reward. The RL algorithm used to achieve this is defined in the `train.py` script.

At every time step, the agent is rewarded or penalized depending on its actions. For instance, sending commands decreases the agent's reward by a certain amount. Over time, the agent learns to take actions that maximize its cumulative reward, which corresponds to effectively fighting the zombie.

## Results

Results from the training process are saved under the `results` folder in the form of CSV files and figures. These results indicate how well the agent learns to fight the zombie over the course of training. They can be visualized using the `result.ipynb` Jupyter notebook.

## Illustrations

The `slide` folder contains illustrations that help to explain the project and its objectives, the environment and the agent, and the training process. They can serve as a useful visual aid when explaining the project to others. For the presentation of this project please refer to this [demonstration video](https://www.youtube.com/watch?v=0m9NwT1oHZg).

## Dependencies

A crucial dependency of the project is the Microsoft Project Malmo platform. The `zlib.dll` file is required for running the Malmo client. Additionally, successful execution of the project requires that the Malmo platform is correctly installed and running.

## Conclusion

This project serves as an interesting and challenging application of AI, demonstrating how an agent can learn to navigate and interact within a complex environment. While the current task is relatively simple - fighting a zombie - the principles applied can be extended to more complex tasks and environments.
