from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from state_sim.curriculum import should_promote
from state_sim.environment import Action, AlbionStateSim


def _target_cell(x: float, y: float, grid_size: int) -> int:
    row = min(grid_size - 1, max(0, int(y * grid_size)))
    col = min(grid_size - 1, max(0, int(x * grid_size)))
    return row * grid_size + col


def heuristic_policy(obs: dict[str, float | int | str | bool], grid_size: int) -> int:
    hp = float(obs["hp"])
    zone_type = str(obs["zone_type"])
    x = float(obs["x"])
    y = float(obs["y"])
    inventory_load = float(obs["inventory_load"])
    threat = float(obs["threat_score"])
    nearest_resource_dist = float(obs.get("nearest_resource_dist", 1.0))
    nearest_mob_dist = float(obs.get("nearest_mob_dist", 1.0))
    nearest_gate_dist = float(obs.get("nearest_gate_dist", 1.0))
    nearest_obstacle_dist = float(obs.get("nearest_obstacle_dist", 1.0))

    center_cell = _target_cell(0.5, 0.5, grid_size)

    if hp < 0.35 or threat > 0.75:
        return center_cell

    if inventory_load > 0.78:
        if zone_type == "city":
            if abs(x - 0.5) + abs(y - 0.56) < 0.14:
                return grid_size * grid_size + int(Action.INTERACT)
            return _target_cell(0.5, 0.56, grid_size)
        if nearest_gate_dist < 0.16:
            return grid_size * grid_size + int(Action.INTERACT)
        edge_x = 0.5 if x < 0.5 else 0.9
        edge_y = 0.1 if y > 0.5 else 0.9
        return _target_cell(edge_x, edge_y, grid_size)

    if zone_type != "city" and nearest_resource_dist < 0.14:
        return grid_size * grid_size + int(Action.INTERACT)

    if zone_type != "city" and nearest_mob_dist < 0.15 and random.random() < 0.25:
        return grid_size * grid_size + int(Action.ATTACK)

    if nearest_obstacle_dist < 0.08:
        safe_x = min(0.95, max(0.05, x + random.uniform(-0.30, 0.30)))
        safe_y = min(0.95, max(0.05, y + random.uniform(-0.30, 0.30)))
        return _target_cell(safe_x, safe_y, grid_size)

    if zone_type == "city":
        if random.random() < 0.25:
            return grid_size * grid_size + int(Action.MOUNT_TOGGLE)
        return _target_cell(0.5, 0.1, grid_size)

    if random.random() < 0.15:
        return grid_size * grid_size + int(Action.IDLE)

    jitter_x = min(0.95, max(0.05, x + random.uniform(-0.18, 0.18)))
    jitter_y = min(0.95, max(0.05, y + random.uniform(-0.18, 0.18)))
    return _target_cell(jitter_x, jitter_y, grid_size)


def run_phase(
    sim: AlbionStateSim, episodes: int = 30, render: bool = False
) -> dict[str, float]:
    success_rates: list[float] = []
    death_rates: list[float] = []
    avg_return_steps: list[float] = []

    can_render_human = True

    for _ in range(episodes):
        obs = sim.reset()
        done = False
        while not done:
            action = heuristic_policy(obs, sim.grid_size)
            obs, _, done, info = sim.step(action)
            if render and can_render_human:
                try:
                    sim.render(mode="human")
                    time.sleep(0.02)
                except RuntimeError:
                    can_render_human = False
                    sim.render(mode="rgb_array")

        success_rates.append(float(info.get("success_rate", 0.0)))
        death_rates.append(float(info.get("death_rate", 1.0)))
        avg_return_steps.append(float(info.get("avg_return_steps", sim.max_steps)))

    return {
        "success_rate": mean(success_rates),
        "death_rate": mean(death_rates),
        "avg_return_steps": mean(avg_return_steps),
    }


def main() -> None:
    sim = AlbionStateSim(seed=42)

    for phase_index, phase in enumerate(sim.curriculum):
        sim.set_phase(phase_index)
        render = phase_index == len(sim.curriculum) - 1
        metrics = run_phase(sim, episodes=25, render=render)
        promoted = should_promote(phase, metrics)
        print(
            f"[{phase_index}] {phase.name} | "
            f"success={metrics['success_rate']:.2f}, "
            f"death={metrics['death_rate']:.2f}, "
            f"return_steps={metrics['avg_return_steps']:.1f}, "
            f"promote={promoted}"
        )

    sim.close()


if __name__ == "__main__":
    main()
