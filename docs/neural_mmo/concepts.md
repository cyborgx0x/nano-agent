# Multi-Agent Reinforcement Learning Concepts

This document explains key concepts in multi-agent RL using Neural MMO as context.

---

## Table of Contents

1. [Reinforcement Learning Basics](#reinforcement-learning-basics)
2. [Multi-Agent RL](#multi-agent-rl)
3. [Emergent Behavior](#emergent-behavior)
4. [Neural MMO Specifics](#neural-mmo-specifics)
5. [Advanced Topics](#advanced-topics)

---

## Reinforcement Learning Basics

### The MDP Framework

**Markov Decision Process (MDP)** is the foundation of RL:

```
Agent observes State (S) →
  chooses Action (A) →
    receives Reward (R) →
      transitions to new State (S')
```

**Components**:
- **State (S)**: What the agent observes (e.g., HP, position, nearby entities)
- **Action (A)**: What the agent can do (e.g., move north, attack)
- **Reward (R)**: Feedback signal (e.g., +1 for surviving, +10 for kill)
- **Policy (π)**: Strategy for choosing actions (neural network)
- **Value Function (V)**: Expected future rewards from a state

### Key RL Algorithms

#### 1. **Q-Learning** (Value-Based)

Learn Q(s, a) = expected future reward for taking action `a` in state `s`

```python
# Update rule
Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]
```

**Pros**: Simple, works well for discrete actions
**Cons**: Struggles with large state/action spaces

#### 2. **Policy Gradient** (Policy-Based)

Directly optimize the policy π(a|s)

```python
# REINFORCE algorithm
θ ← θ + α ∇ log π(a|s) * G  # G = cumulative reward
```

**Pros**: Works with continuous actions
**Cons**: High variance, slow convergence

#### 3. **PPO** (Actor-Critic)

Combines value and policy learning with clipped objective

```python
# Simplified PPO objective
L = min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
```

**Pros**: Stable, widely used (ChatGPT was trained with PPO!)
**Cons**: Requires careful hyperparameter tuning

---

## Multi-Agent RL

### Single-Agent vs. Multi-Agent

| Single-Agent RL | Multi-Agent RL |
|----------------|----------------|
| Static environment | Environment changes with other agents |
| Learn fixed strategy | Must adapt to opponents' strategies |
| Example: Chess AI vs. fixed rules | Example: Neural MMO (100+ adapting agents) |

### Key Challenges in MARL

#### 1. **Non-Stationarity**

Other agents are learning too, so the environment keeps changing!

```
Agent A learns to attack Agent B →
  Agent B learns to dodge →
    Agent A must adapt again →
      Endless cycle!
```

**Solution**: Independent learning or centralized training

#### 2. **Credit Assignment**

With many agents, hard to know who caused what outcome.

```
Team wins battle →
  Which agent's actions were most important?
  How to distribute reward?
```

**Solution**: Reward shaping, counterfactual reasoning

#### 3. **Scalability**

100 agents = 100^N possible joint actions!

**Solution**: Mean field approximation, graph neural networks

### MARL Paradigms

#### **Independent Learning**

Each agent learns its own policy, ignoring others.

```python
for agent in agents:
    agent.update(own_experience)  # Treat others as environment
```

**Pros**: Simple, scalable
**Cons**: Non-stationary, may not converge

#### **Centralized Training, Decentralized Execution (CTDE)**

Training: Use global info (all agents' states)
Execution: Each agent acts independently

```python
# Training
critic.update(global_state, all_actions)  # Uses full info

# Execution
action = policy(local_observation)  # Uses only own observation
```

**Pros**: Stable training, practical deployment
**Cons**: Requires centralized training infrastructure

#### **Communication**

Agents share information explicitly.

```python
message = agent_A.send_message(observation)
agent_B.receive_message(message)
action = agent_B.act(observation, message)
```

**Pros**: Can coordinate complex strategies
**Cons**: Needs to learn communication protocol

---

## Emergent Behavior

**Emergence**: Complex behaviors arising from simple rules.

### Examples in Neural MMO

#### 1. **Specialization**

Without explicit programming, agents may specialize:

- **Farmers**: Gather food, avoid combat
- **Warriors**: Attack others, steal resources
- **Traders**: Mediate exchanges

Why? Different strategies can be locally optimal!

#### 2. **Alliance Formation**

Agents near each other may cooperate:

- Share food during scarcity
- Defend common territory
- Trade complementary resources

#### 3. **Economic Systems**

Supply and demand emerge naturally:

```
Many agents gather food → Food abundant → Low trade value
Few agents gather stone → Stone rare → High trade value
```

### How to Study Emergence

#### **Metrics**:
- **Diversity**: How different are agent strategies?
- **Clustering**: Do similar agents group together?
- **Trade volume**: How much exchange occurs?

#### **Visualization**:
- Plot agent positions over time
- Color-code by strategy type
- Analyze social networks

---

## Neural MMO Specifics

### Survival Mechanics

Agents must balance 3 resources:

```python
# Simplified survival logic
if food <= 0:
    health -= 1  # Starvation damage
if water <= 0:
    health -= 1  # Dehydration damage
if health <= 0:
    die()
```

**Learning Challenge**: Long-term planning (gather food now or explore?)

### Combat System

Combat uses virtual dice rolls with stats:

```python
damage = attack_level - defense_level + random_noise
```

**Learning Challenge**: When to fight vs. flee?

### Exploration vs. Exploitation

Classic RL dilemma:

- **Exploration**: Discover new areas (potential resources)
- **Exploitation**: Stay in known safe area (guaranteed survival)

Neural MMO map is large (128x128) with fog of war!

### Multi-Objective Optimization

Agents optimize for:
- **Survival** (live long)
- **Combat** (defeat enemies)
- **Foraging** (gather resources)
- **Exploration** (discover map)

How to balance? Reward shaping!

```python
reward = w1 * survival_time + w2 * kills + w3 * resources + w4 * tiles_explored
```

---

## Advanced Topics

### 1. **Curriculum Learning**

Start with easy tasks, gradually increase difficulty:

```
Stage 1: Learn to survive (food/water only)
Stage 2: Add combat (weak enemies)
Stage 3: Add trading
Stage 4: Full game (100+ agents)
```

### 2. **Population-Based Training**

Train a population of diverse agents:

```python
population = [Agent(seed=i) for i in range(100)]

while training:
    for agent in population:
        agent.play_games()
        agent.update_policy()

    # Evolution: Replace worst agents with mutated best agents
    population = evolve(population)
```

**Benefit**: Diversity prevents convergence to local optimum

### 3. **Meta-Learning**

Learn to learn!

```python
# Inner loop: Adapt to specific environment
for env in environments:
    fast_adapt(agent, env)

# Outer loop: Improve adaptation ability
meta_update(agent)
```

**Goal**: Agent quickly adapts to new maps/opponents

### 4. **Graph Neural Networks**

Represent agents and resources as graph:

```
Nodes: Agents, resources, terrain
Edges: Proximity, line-of-sight
```

**Benefit**: Handles variable number of agents

---

## Comparison: Albion Bot vs. Neural MMO Agent

| Aspect | Albion Bot | Neural MMO |
|--------|-----------|------------|
| **State Representation** | Raw pixels (1920x1080x3) | Symbolic (position, HP, etc.) |
| **Learning** | None (scripted) | RL (learns from experience) |
| **Perception** | YOLO (object detection) | Direct access to entities |
| **Decision Making** | If-else rules | Neural network policy |
| **Adaptation** | Manual updates | Automatic through training |
| **Complexity** | Fixed strategy | Emergent strategies |

**Key Insight**:
- **Albion bot** = Engineering solution (works now, hard to generalize)
- **Neural MMO** = Learning solution (slow to train, generalizes better)

---

## Learning Resources

### Papers
1. **Neural MMO**: [Original paper](https://arxiv.org/abs/1903.00784)
2. **Multi-Agent RL survey**: [Zhang et al., 2019](https://arxiv.org/abs/1911.10635)
3. **Emergent Complexity**: [Leibo et al., 2017](https://arxiv.org/abs/1702.03037)

### Books
- **RL: An Introduction** (Sutton & Barto) - Chapter 1-6 for basics
- **Multi-Agent Systems** (Wooldridge) - For game theory foundations

### Code Examples
- Stable Baselines3 tutorials
- PettingZoo multi-agent environments
- Neural MMO official examples

---

## Exercises

1. **Implement a baseline agent** that survives as long as possible
2. **Add a combat strategy** - when to attack vs. flee?
3. **Experiment with reward shaping** - what weights work best?
4. **Visualize emergent behavior** - do alliances form?
5. **Compare algorithms** - PPO vs. DQN vs. random

---

## Next Steps

1. Run the starter agent (`starter_agent.py`)
2. Modify reward function and observe behavior changes
3. Implement a simple strategy (e.g., "always run from combat")
4. Read research papers on emergent behavior
5. Join the Neural MMO Discord to discuss ideas!

Happy learning! 🧠🎮
