"""Inter-map transition benchmark (Epic #1, issue #4).

Runs one or more agent architectures on solvable navigation tasks and
measures how each behaves in the steps immediately after it crosses a
gate into a new map.

Example::

    python -m state_sim.transition_benchmark --agents random greedy tabular_q
    python -m state_sim.transition_benchmark --agents greedy ppo \\
        --observation-mode both --episodes 200 --output-csv results/transition.csv

Metrics (measured over a fixed window of ``--window`` steps after each
gate transition, then pooled across all transitions of an agent):

- transitions_per_ep   : gate crossings per episode
- dwell_steps          : steps spent in a zone before the next transition
- displacement         : path length travelled inside the post-transition window
- stagnation           : mean env stagnation counter inside the window
- goal_progress        : drop in BFS zone-distance to the goal over the window
- backtrack_rate       : fraction of transitions whose next move returns to
                         the zone just left (disorientation / oscillation)
- first_action_entropy : policy entropy on the first action in the new zone
                         (only defined for stochastic-policy agents, e.g. PPO)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from state_sim.agent_interface import (
    DEFAULT_PPO_CHECKPOINT,
    AgentProtocol,
    build_agent,
    connected_components,
)
from state_sim.environment import AlbionStateSim


def solvable_tasks(env: AlbionStateSim, seed: int, count: int) -> list[tuple[str, str]]:
    """Sample (start, goal) pairs that lie in the same connected component.

    Cross-component pairs are unsolvable (the zone graph is split into
    per-region clusters) and would swamp the metrics with noise.
    """
    components = connected_components(env.zone_manager.zone_graph)
    zone_component = {z: i for i, comp in enumerate(components) for z in comp}
    zones = [z for z, t in env.zone_manager.zone_types.items() if t != "city"]
    rng = random.Random(seed)
    tasks: list[tuple[str, str]] = []
    while len(tasks) < count:
        start = rng.choice(zones)
        reachable = [
            z
            for z in zones
            if z != start and zone_component[z] == zone_component[start]
        ]
        if reachable:
            tasks.append((start, rng.choice(reachable)))
    return tasks


def run_episode(
    env: AlbionStateSim, agent: AgentProtocol, start: str, goal: str
) -> tuple[list[dict], float]:
    """Play one episode and return its per-step trace plus the success flag."""
    env.set_navigation_task(start, goal, enabled=True)
    obs = env.reset()
    agent.reset()
    trace: list[dict] = []
    prev = (float(obs["x"]), float(obs["y"]))
    success = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        entropy = float(getattr(agent, "last_action_entropy", float("nan")))
        obs, _reward, done, info = env.step(action)
        cur = (float(obs["x"]), float(obs["y"]))
        event = info.get("transition_event")
        trace.append(
            {
                "move": math.hypot(cur[0] - prev[0], cur[1] - prev[1]),
                "goal_distance": float(obs["goal_distance"]),
                "stagnation": float(obs["stagnation_ticks"]),
                "transition": (
                    (str(event["from_zone"]), str(event["to_zone"]))
                    if isinstance(event, dict)
                    else None
                ),
                "entropy": entropy,
            }
        )
        prev = cur
        if done:
            success = float(info.get("navigation_success", 0.0))
    return trace, success


def episode_metrics(trace: list[dict], success: float, window: int) -> dict:
    """Extract episode-level and per-transition metrics from a trace."""
    transition_steps = [i for i, r in enumerate(trace) if r["transition"]]
    per_transition: list[dict] = []
    for k, idx in enumerate(transition_steps):
        win = trace[idx : idx + window]
        from_zone, _to_zone = trace[idx]["transition"]
        has_next = k + 1 < len(transition_steps)
        next_idx = transition_steps[k + 1] if has_next else len(trace)
        backtrack = 0.0
        if has_next:
            _nf, next_to = trace[transition_steps[k + 1]]["transition"]
            backtrack = 1.0 if next_to == from_zone else 0.0
        first_entropy = (
            trace[idx + 1]["entropy"] if idx + 1 < len(trace) else float("nan")
        )
        per_transition.append(
            {
                "dwell": float(next_idx - idx),
                "backtrack": backtrack,
                "has_next": has_next,
                "displacement": sum(r["move"] for r in win),
                "stagnation": mean(r["stagnation"] for r in win) if win else 0.0,
                "goal_progress": trace[idx]["goal_distance"]
                - win[-1]["goal_distance"],
                "first_entropy": first_entropy,
            }
        )
    return {
        "transitions": len(transition_steps),
        "episode_steps": len(trace),
        "success": success,
        "per_transition": per_transition,
    }


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return mean(finite) if finite else float("nan")


def benchmark_agent(
    agent_name: str,
    env: AlbionStateSim,
    tasks: list[tuple[str, str]],
    window: int,
    *,
    seed: int,
    ppo_checkpoint: str,
    tabular_train_episodes: int,
    observation_mode: str,
) -> dict | None:
    """Build one agent and run it over every task; return aggregate metrics."""
    try:
        agent = build_agent(
            agent_name,
            env,
            seed=seed,
            ppo_checkpoint=ppo_checkpoint,
            tabular_train_episodes=tabular_train_episodes,
            observation_mode=observation_mode,
        )
    except FileNotFoundError as exc:
        print(f"  skip {agent_name} ({observation_mode}): {exc}")
        return None

    episodes = [
        episode_metrics(*run_episode(env, agent, start, goal), window)
        for start, goal in tasks
    ]
    transitions = [t for ep in episodes for t in ep["per_transition"]]
    with_next = [t for t in transitions if t["has_next"]]

    return {
        "agent": agent.name,
        "observation_mode": observation_mode,
        "episodes": len(episodes),
        "nav_success": mean(ep["success"] for ep in episodes),
        "transitions_per_ep": mean(ep["transitions"] for ep in episodes),
        "episode_steps": mean(ep["episode_steps"] for ep in episodes),
        "n_transitions": len(transitions),
        "dwell_steps": _safe_mean([t["dwell"] for t in transitions]),
        "displacement": _safe_mean([t["displacement"] for t in transitions]),
        "stagnation": _safe_mean([t["stagnation"] for t in transitions]),
        "goal_progress": _safe_mean([t["goal_progress"] for t in transitions]),
        "backtrack_rate": (
            mean(t["backtrack"] for t in with_next) if with_next else float("nan")
        ),
        "first_action_entropy": _safe_mean(
            [t["first_entropy"] for t in transitions]
        ),
    }


_COLUMNS = [
    ("agent", "agent", "{}"),
    ("observation_mode", "obs_mode", "{}"),
    ("episodes", "episodes", "{}"),
    ("nav_success", "nav_success", "{:.2f}"),
    ("transitions_per_ep", "transitions/ep", "{:.2f}"),
    ("dwell_steps", "dwell_steps", "{:.1f}"),
    ("displacement", "displacement", "{:.3f}"),
    ("stagnation", "stagnation", "{:.2f}"),
    ("goal_progress", "goal_progress", "{:.3f}"),
    ("backtrack_rate", "backtrack_rate", "{:.2f}"),
    ("first_action_entropy", "first_entropy", "{:.3f}"),
    ("n_transitions", "n_transitions", "{}"),
]


def _fmt(value: object, fmt: str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "—"
    return fmt.format(value)


def render_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(label for _, label, _ in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [_fmt(row[key], fmt) for key, _, fmt in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _, _ in _COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key, _, _ in _COLUMNS})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark agent behavior at inter-map transitions"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["random", "greedy", "tabular_q"],
        help="agent architectures to compare",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--observation-mode",
        choices=["partial", "full", "both"],
        default="partial",
    )
    parser.add_argument("--ppo-checkpoint", type=str, default=DEFAULT_PPO_CHECKPOINT)
    parser.add_argument("--tabular-train-episodes", type=int, default=3000)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = ["partial", "full"] if args.observation_mode == "both" else [
        args.observation_mode
    ]

    env = AlbionStateSim(seed=args.seed, max_steps=args.max_steps)
    tasks = solvable_tasks(env, args.seed, args.episodes)
    print(
        f"benchmark: {len(tasks)} solvable tasks | agents={args.agents} "
        f"| modes={modes} | window={args.window}"
    )

    rows: list[dict] = []
    for mode in modes:
        env.set_observation_mode(mode)
        for agent_name in args.agents:
            print(f"running {agent_name} (obs_mode={mode}) ...")
            result = benchmark_agent(
                agent_name,
                env,
                tasks,
                args.window,
                seed=args.seed,
                ppo_checkpoint=args.ppo_checkpoint,
                tabular_train_episodes=args.tabular_train_episodes,
                observation_mode=mode,
            )
            if result is not None:
                rows.append(result)
    env.close()

    if not rows:
        print("no agents ran successfully")
        return

    table = render_markdown(rows)
    print("\n" + table + "\n")

    if args.output_csv:
        write_csv(rows, args.output_csv)
        print(f"wrote csv: {args.output_csv}")
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"wrote markdown: {args.output_md}")


if __name__ == "__main__":
    main()
