"""Quick debug script to verify gather mechanics work."""

from state_sim.environment import AlbionStateSim
from state_sim.ppo.trainer import _teacher_action_gather

env = AlbionStateSim(seed=42, max_steps=1500)
env.inventory_full_gather_events = 99
env.set_task_type("gather")
# Use multi-zone (not single zone) so it picks a non-city zone with city access
env.set_single_zone_training("forest_cross", enabled=False)
env.set_gather_spawn_zones(["forest_cross"])

obs = env.reset()
done = False
s = 0
info = {}

while not done:
    action = _teacher_action_gather(env)
    obs, rew, done, info = env.step(action)
    s += 1
    if s <= 10 or s % 100 == 0 or env.state.hp < 0.55:
        print(
            f"s={s} zone={env.state.zone_id} g={env._episode_gather_events} "
            f"stay={env._gather_stay_ticks} hp={env.state.hp:.2f} inv={env.state.inventory_load:.3f} "
            f"threat={env.state.threat_score:.2f}"
        )

print(
    f"DONE s={s} gathers={info.get('episode_gather_events', 0)} "
    f"inv_full={info.get('inventory_full', False)} dead={info.get('dead', False)} "
    f"gather_success={info.get('gather_success', 0):.3f}"
)
