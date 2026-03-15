import numpy as np
import torch
from memory import Memory
from agent import PPOAgent, ComplexPolicyNetwork


def extract_state_features(detection_results, max_objects=10):
    state_features = []
    for detection in detection_results[:max_objects]:
        box = detection["box"]
        center_x = (box["x1"] + box["x2"]) / 2
        center_y = (box["y1"] + box["y2"]) / 2
        confidence = detection["confidence"]
        fiber_class = detection["class"]
        state_features.extend([center_x, center_y, confidence, fiber_class])

    while len(state_features) < max_objects * 4:
        state_features.append(0.0)

    return state_features


detection_results = [
    {
        "name": "cotton",
        "class": 1,
        "confidence": 0.9121813178062439,
        "box": {
            "x1": 744.4118041992188,
            "y1": 404.649169921875,
            "x2": 827.045654296875,
            "y2": 477.1366882324219,
        },
    },
    {
        "name": "cotton",
        "class": 1,
        "confidence": 0.9019400477409363,
        "box": {
            "x1": 1176.33154296875,
            "y1": 269.1721496582031,
            "x2": 1239.0404052734375,
            "y2": 337.95953369140625,
        },
    },
    {
        "name": "flax",
        "class": 2,
        "confidence": 0.34675437211990356,
        "box": {
            "x1": 1294.613525390625,
            "y1": 288.8346862792969,
            "x2": 1349.121826171875,
            "y2": 346.90240478515625,
        },
    },
]


def train():
    input_dim = 40  # 10 objects * 4 features (x, y, confidence, class)
    action_space = 8  # click + q + w + e + r + d + f
    policy_net = ComplexPolicyNetwork(input_dim, action_space)
    ppo_agent = PPOAgent(policy_net, input_dim, action_space)
    memory = Memory()

    num_episodes = 1000
    max_timesteps = 300

    for episode in range(num_episodes):
        state_features = extract_state_features(detection_results, max_objects=10)
        state = state_features

        for t in range(max_timesteps):
            action, log_prob, value = ppo_agent.select_action(state)
            if action is None or log_prob is None or value is None:
                print("Skipping step due to NaNs in action selection")
                continue

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.values.append(value)

            reward = np.random.rand()  # Replace with actual reward from the environment
            is_terminal = (
                np.random.rand() > 0.95
            )  # Replace with actual terminal condition

            memory.rewards.append(reward)
            memory.is_terminals.append(is_terminal)

            state_features = extract_state_features(detection_results, max_objects=10)
            state = state_features

            if is_terminal:
                break

        ppo_agent.update(memory)
        memory.clear_memory()

        if episode % 10 == 0:
            print(f"Episode {episode} completed")


train()
