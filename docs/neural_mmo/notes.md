# Neural MMO - Research Notes & Experiment Ideas

Personal notes and ideas for exploring Neural MMO.

---

## Experiment Ideas

### Beginner Experiments

#### 1. Survival Time Comparison
**Goal**: Compare different strategies' survival times

**Agents to test**:
- Random agent (baseline)
- Aggressive forager (always prioritize food/water)
- Conservative explorer (balance survival and exploration)
- Pacifist (never attack)
- Warrior (aggressive combat)

**Metrics**:
- Average survival time
- Total resources gathered
- Number of kills

**Hypothesis**: Aggressive foragers will survive longest in sparse environments

---

#### 2. Resource Scarcity Impact
**Goal**: How does resource availability affect behavior?

**Experiment**:
```python
configs = [
    {'RESOURCE_DENSITY': 'high'},
    {'RESOURCE_DENSITY': 'medium'},
    {'RESOURCE_DENSITY': 'low'},
]

for config in configs:
    run_simulation(config)
    analyze_survival_times()
```

**Expected result**: Low resource density → more competition → shorter survival

---

#### 3. Reward Shaping Exploration
**Goal**: Find optimal reward weights

**Test different reward functions**:
```python
# Survival-focused
reward = 10 * survival_time

# Combat-focused
reward = 100 * kills - 10 * deaths

# Balanced
reward = 5 * survival_time + 20 * kills + 2 * resources_gathered

# Exploration-focused
reward = 1 * tiles_explored + 5 * survival_time
```

**Analysis**: Which reward leads to most interesting emergent behavior?

---

### Intermediate Experiments

#### 4. Emergent Specialization
**Goal**: Do agents naturally specialize into roles?

**Setup**:
- Train 100 agents with identical initialization
- Track their behavior over time
- Cluster agents by strategy

**Clustering features**:
- Combat frequency
- Resource gathering rate
- Movement patterns
- Trading volume

**Question**: Do farmers, warriors, and traders emerge naturally?

---

#### 5. Alliance Formation
**Goal**: Can agents learn to cooperate?

**Approach 1 - Shared Reward**:
```python
# Reward nearby agents for helping each other
if agent_A near agent_B:
    if agent_A.shares_food(agent_B):
        reward_A += 5
        reward_B += 5
```

**Approach 2 - Team Competition**:
```python
teams = [Team1, Team2]
reward = team_survival_time + team_total_kills
```

**Metrics**:
- Proximity clustering (do allies stay together?)
- Resource sharing frequency
- Coordinated attacks

---

#### 6. Communication Protocols
**Goal**: Can agents learn to communicate?

**Setup**:
```python
class CommunicatingAgent:
    def send_message(self):
        return [0.0, 1.0, 0.5]  # 3D message vector

    def receive_message(self, msg):
        # Use message to inform decision
        ...
```

**Analysis**:
- What do message dimensions encode?
- Does communication improve coordination?
- Visualize message patterns

---

### Advanced Experiments

#### 7. Meta-Learning Across Maps
**Goal**: Train agents that quickly adapt to new environments

**Approach**: MAML (Model-Agnostic Meta-Learning)

```python
# Inner loop: Adapt to specific map
for map in training_maps:
    agent.fast_adapt(map, steps=100)

# Outer loop: Improve adaptation
meta_optimizer.step()

# Test: Deploy on unseen map
test_map = generate_new_map()
agent.fast_adapt(test_map, steps=10)  # Should adapt quickly!
```

**Hypothesis**: Meta-learned agents generalize better than single-map specialists

---

#### 8. Economic System Emergence
**Goal**: Does supply/demand emerge naturally?

**Setup**:
- Enable trading system
- Track item prices over time
- Analyze market dynamics

**Metrics**:
- Price fluctuations
- Trade volume
- Specialization (some agents produce, others trade)

**Question**: Do market equilibria emerge? Can we predict prices?

---

#### 9. Population-Based Training
**Goal**: Evolve diverse strategies

**Algorithm**:
```
1. Initialize population of 100 agents with different seeds
2. Each agent plays games and collects experience
3. Train all agents with RL
4. Evaluate fitness (survival time)
5. Replace worst 20% with mutated copies of best 20%
6. Repeat 2-5 for N generations
```

**Analysis**:
- Diversity metrics (behavioral distance between agents)
- Evolutionary dynamics (do certain strategies dominate?)
- Emergent arms races (attackers vs. defenders)

---

## Comparison Questions

### Albion Bot vs. Neural MMO Agent

**Question 1**: Can we apply Neural MMO techniques to Albion?
- Use RL instead of scripted YOLO pipeline?
- Train agent end-to-end from pixels?
- Challenge: Real game too complex, sample inefficient

**Question 2**: Can we simulate Albion in Neural MMO?
- Add gathering mechanics like Albion
- Test strategies in simulation before deploying to real game
- Benefit: Fast iteration, no risk of game ban

**Question 3**: Hybrid approach?
- Use Neural MMO to learn high-level strategy (where to go, when to fight)
- Use Albion bot's vision system for low-level control (clicking)
- Best of both worlds?

---

## Implementation TODOs

### Short-term (1-2 weeks)
- [ ] Run all 3 starter agents and compare survival times
- [ ] Implement custom reward function (survival + exploration)
- [ ] Visualize agent trajectories on map
- [ ] Read 3 research papers on emergent behavior

### Medium-term (1 month)
- [ ] Train PPO agent for 1M timesteps
- [ ] Implement simple communication protocol
- [ ] Analyze emergent specialization (clustering analysis)
- [ ] Create visualization dashboard (agent stats over time)

### Long-term (3 months)
- [ ] Implement meta-learning (MAML or similar)
- [ ] Run population-based training experiment
- [ ] Study economic systems (supply/demand)
- [ ] Write blog post or paper about findings

---

## Open Research Questions

1. **Optimal reward structure**: What reward leads to most human-like behavior?
2. **Communication emergence**: Can meaningful communication arise without explicit rewards?
3. **Generalization**: How to train agents that work on any map?
4. **Social dynamics**: Do reputation systems emerge (trust/distrust)?
5. **Long-term planning**: Can agents plan 1000+ steps ahead?

---

## Resources & References

### Papers to Read
- [x] Neural MMO (Suarez et al., 2019)
- [ ] Emergent Tool Use (OpenAI, 2019)
- [ ] Multi-Agent DDPG (Lowe et al., 2017)
- [ ] QMIX (Rashid et al., 2018)
- [ ] Hide and Seek (Baker et al., 2020)

### Code Repositories
- Neural MMO official repo
- RLlib examples (multi-agent)
- PettingZoo tutorials
- Stable Baselines3 docs

### Community
- Neural MMO Discord
- /r/reinforcementlearning subreddit
- Papers with Code (Neural MMO page)

---

## Observations & Insights

### 2025-10-09: Initial Exploration
- Random agents surprisingly survive 100+ steps sometimes
- Food scarcity is the main early-game challenge
- Combat is rare in early steps (agents spread out)

### [Your Date]: [Your Observation]
- ...

---

## Debugging Tips

### Common Issues

**Agent dies immediately**:
- Check food/water consumption rate
- Verify starting positions (not in water tiles)
- Debug reward function (negative rewards?)

**Training not converging**:
- Reduce learning rate
- Increase batch size
- Add reward shaping (incremental progress)
- Curriculum learning (start easy)

**Out of memory**:
- Reduce number of agents
- Use smaller map size
- Enable gradient checkpointing

---

## Next Actions

1. **Today**: Run random vs. scripted agent comparison
2. **This week**: Implement custom reward and visualize results
3. **This month**: Train first RL agent successfully
4. **Long-term**: Publish findings or contribute to Neural MMO

---

## Personal Goals

What do you want to learn from Neural MMO?

- [ ] Understand multi-agent RL fundamentals
- [ ] Implement state-of-the-art MARL algorithms
- [ ] Study emergent social behavior
- [ ] Apply techniques to real-world problems (game bots, robotics, etc.)
- [ ] Contribute to research community

---

**Keep these notes updated as you experiment! Document everything - failures teach as much as successes. 📝**
