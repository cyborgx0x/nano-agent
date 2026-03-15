"""Quick validation: teacher + phase 5 + new rewards."""

from state_sim.environment import AlbionStateSim
from state_sim.ppo.trainer import _teacher_action_gather

for seed in range(10):
    env = AlbionStateSim(seed=seed, max_steps=1500)
    env.inventory_full_gather_events = 99
    env.set_task_type("gather")
    env.set_single_zone_training("forest_cross", enabled=False)
    env.set_gather_spawn_zones(["forest_cross"])
    env.set_phase(5)
    obs = env.reset()
    done = False
    s = 0
    info = {}
    total_reward = 0.0
    while not done:
        action = _teacher_action_gather(env)
        obs, rew, done, info = env.step(action)
        total_reward += rew
        s += 1
    g = info.get("episode_gather_events", 0)
    full = info.get("inventory_full", False)
    dead = info.get("dead", False)
    print(
        f"seed={seed:2d}  steps={s:4d}  gathers={g:.0f}  inv_full={full}"
        f"  dead={dead}  total_reward={total_reward:.2f}"
    )
