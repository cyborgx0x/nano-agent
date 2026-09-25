# Neural MMO - Multi-Agent Reinforcement Learning Environment

## What is Neural MMO?

Neural MMO is a **massively multiagent research environment** inspired by Massively Multiplayer Online (MMO) games. It's designed for studying:

- **Multi-agent reinforcement learning** at scale (100+ agents)
- **Emergent behaviors** in complex social systems
- **Long-term planning** and survival strategies
- **Competition and cooperation** dynamics

Think of it as a simplified MMO game world built specifically for AI research.

---

## Why Neural MMO?

### Traditional RL vs. Multi-Agent RL

| Single-Agent RL | Multi-Agent RL (Neural MMO) |
|----------------|----------------------------|
| Static environment | Dynamic, changing agents |
| Fixed strategy works | Must adapt to other agents |
| Limited complexity | Emergent complexity |
| Examples: Atari, CartPole | Examples: Neural MMO, Starcraft II |

### Comparison to Albion Bot (in this repo)

| Albion Bot (this repo) | Neural MMO |
|----------------------|------------|
| **Real game** (Albion Online) | **Simulated environment** |
| Computer vision (YOLO, OCR) | Symbolic state (coordinates, HP, items) |
| Scripted behavior | Reinforcement learning |
| Single agent | 100+ agents interacting |
| Resource gathering only | Survival + combat + trading + exploration |

---

## Core Features

### 1. Survival Mechanics
- **Food**: Agents need to forage food to avoid starvation
- **Water**: Agents need water to stay hydrated
- **Health**: Combat and environmental damage

### 2. Combat System
- **Melee** and **ranged** attacks
- **Range** and **mage** skills
- PvP (player vs. player) combat

### 3. Resource Gathering
- **Foraging**: Collect food and water from tiles
- **Alchemy**: Craft items (future feature)

### 4. Trading & Economics
- **Item exchange** between agents
- **Emergent economy** (supply/demand)
- **Specialization** (some agents gather, others fight)

### 5. Exploration
- **Procedurally generated maps**
- **Fog of war** (limited vision)
- **Tile-based movement**

---

## Environment Details

### Observation Space
Each agent receives:
- **Position**: Current (x, y) coordinates
- **Health/Food/Water**: Survival stats
- **Inventory**: Items held
- **Nearby entities**: Other agents, resources, terrain
- **Vision range**: Limited field of view

### Action Space
Agents can:
- **Move**: North, South, East, West
- **Attack**: Melee, ranged, mage
- **Use**: Consume food/water
- **Give**: Trade items with other agents

### Reward Structure
- **Survival time**: Longer survival = higher reward
- **Combat**: Defeating enemies
- **Foraging**: Collecting resources
- **Exploration**: Discovering new areas

---

## Key Research Questions

Neural MMO enables research into:

1. **Emergent Behavior**
   - Do agents form alliances?
   - Does specialization emerge (farmers vs. warriors)?
   - Do trading economies develop naturally?

2. **Long-Horizon Planning**
   - Managing food/water over 1000+ timesteps
   - Balancing exploration vs. exploitation
   - Strategic resource accumulation

3. **Multi-Agent Coordination**
   - Can agents learn to cooperate?
   - How do agents handle competition for resources?
   - Does communication emerge?

4. **Generalization**
   - Can agents trained in one map adapt to new maps?
   - Transfer learning across different agent populations?

---

## How It Differs from Other Environments

| Environment | Focus | Complexity |
|------------|-------|-----------|
| **OpenAI Gym** | Single-agent control | Low-Medium |
| **StarCraft II** | Real-time strategy | Very High |
| **PettingZoo** | Multi-agent games | Medium |
| **Neural MMO** | **Massively multi-agent survival** | **High** |

Neural MMO sits between simple multi-agent games and full game simulations like Dota 2.

---

## Use Cases for Learning

### For Beginners
- Learn multi-agent RL fundamentals
- Understand emergent behavior
- Practice with symbolic (non-vision) state

### For Intermediate
- Implement PPO/A3C for multi-agent settings
- Experiment with reward shaping
- Study agent communication protocols

### For Advanced
- Research curriculum learning
- Explore meta-learning across populations
- Study economic systems and game theory

---

## Neural MMO vs. Albion Bot (Side-by-Side)

| Aspect | Albion Bot | Neural MMO |
|--------|-----------|------------|
| **Input** | Screenshots (images) | Symbolic state (numbers) |
| **Perception** | YOLO + OCR | Direct access to game state |
| **Learning** | None (scripted) | Reinforcement Learning |
| **Agents** | 1 bot | 100+ agents |
| **Complexity** | Real game (very complex) | Simplified simulation |
| **Goal** | Automate gathering | Research emergent AI |

**Key Insight**: Albion bot shows **practical application** of CV + automation.
Neural MMO shows **fundamental AI research** in multi-agent systems.

---

## Getting Started

See:
- `setup_guide.md` - Installation instructions
- `concepts.md` - Deep dive into RL concepts
- `starter_agent.py` - Example baseline agent
- `notes.md` - Research ideas and experiments

---

## Resources

- **Official Repo**: https://github.com/NeuralMMO/environment
- **Paper**: "Neural MMO: A Massively Multiagent Game Environment" (Suarez et al., 2019)
- **Documentation**: https://neuralmmo.github.io/
- **Discord**: Community for discussions and support

---

## Next Steps

1. Install Neural MMO (see `setup_guide.md`)
2. Run a baseline agent to see how it works
3. Study the concepts in `concepts.md`
4. Implement your own agent strategy
5. Experiment with emergent behaviors!
