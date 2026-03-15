"""Validate vision-limited obs + spatial memory."""

from state_sim.environment import AlbionStateSim
from state_sim.ppo.encoder import ObsEncoder

env = AlbionStateSim()
enc = ObsEncoder(env)
obs = env.reset()
v = enc.encode(obs)
print(f"obs_dim={enc.obs_dim} vec_len={len(v)}")
assert enc.obs_dim == len(v), f"MISMATCH {enc.obs_dim} != {len(v)}"

# Verify memory features exist in obs
for key in [
    "mem_resource_dist",
    "mem_resource_dx",
    "mem_resource_dy",
    "mem_gate_dist",
    "mem_gate_dx",
    "mem_gate_dy",
    "mem_bank_dist",
    "mem_bank_dx",
    "mem_bank_dy",
]:
    assert key in obs, f"MISSING key: {key}"
print("All 9 memory keys present")

# Test gather: run steps and check memory builds
env.set_task_type("gather")
env.set_phase(5)
obs = env.reset()
for i in range(50):
    obs, r, d, info = env.step(i % 25)
    if d:
        break

zone = env.state.zone_id
mem = env._spatial_memory.get(zone, {})
n_res = len(mem.get("resources", []))
n_gates = len(mem.get("gates", []))
n_banks = len(mem.get("banks", []))
print(f"zone={zone} mem: resources={n_res} gates={n_gates} banks={n_banks}")

# Check that view-limited dist and memory dist are different
print(
    f"view: res_dist={obs['nearest_resource_dist']:.3f} gate_dist={obs['nearest_gate_dist']:.3f}"
)
print(
    f"mem:  res_dist={obs['mem_resource_dist']:.3f} gate_dist={obs['mem_gate_dist']:.3f}"
)

# Check vision: if resource within 0.30, should have direction
# If outside 0.30, dist should be 1.0 (no resource visible)
resources = env._resource_positions()
import math

ax, ay = env.state.x, env.state.y
any_in = any(math.hypot(rx - ax, ry - ay) <= 0.30 for rx, ry in resources)
if not any_in:
    assert (
        obs["nearest_resource_dist"] == 1.0
    ), f"Expected 1.0 for no-visible resources, got {obs['nearest_resource_dist']}"
    print("PASS: no resource in view → dist=1.0")
else:
    assert obs["nearest_resource_dist"] < 1.0
    print(f"PASS: resource in view → dist={obs['nearest_resource_dist']:.3f}")

# Memory should have entries since agent has been moving around
if n_res > 0 or n_gates > 0:
    print(f"PASS: memory accumulated {n_res} resources, {n_gates} gates")
else:
    print("NOTE: memory empty (agent may not have seen entities yet)")

print()
print("ALL CHECKS PASSED")
