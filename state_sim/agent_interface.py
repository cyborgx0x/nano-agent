"""Standard agent interface for the inter-map transition testbed (Epic #1).

Any agent architecture that satisfies :class:`AgentProtocol` can be plugged
into ``transition_benchmark.py`` and compared on the same footing.

Lifecycle per episode::

    agent.reset()
    obs = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        agent.observe(obs, reward, done, info)
"""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Protocol, runtime_checkable

from state_sim.environment import Action, AlbionStateSim

DEFAULT_PPO_CHECKPOINT = "runs/state_sim/map_to_map_ppo_navigation_best.pt"


@runtime_checkable
class AgentProtocol(Protocol):
    """Interface every plug-in architecture must satisfy."""

    @property
    def name(self) -> str:
        """Short identifier used in benchmark reports."""
        ...

    def reset(self) -> None:
        """Clear per-episode state before a new episode begins."""
        ...

    def act(self, obs: dict) -> int:
        """Return an action index in ``[0, env.action_count)``."""
        ...

    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        """Receive the outcome of the last action (next obs, reward, done, info)."""
        ...


def bfs_first_step(graph: dict[str, list[str]], src: str, dst: str) -> str:
    """Return the first zone to move to along the shortest path src -> dst."""
    if src == dst:
        return src
    parent: dict[str, str | None] = {src: None}
    queue: deque[str] = deque([src])
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == dst:
                cur = nxt
                while parent[cur] is not None and parent[cur] != src:
                    cur = parent[cur]  # type: ignore[assignment]
                return cur
            queue.append(nxt)
    return src


def connected_components(graph: dict[str, list[str]]) -> list[set[str]]:
    """Return the connected components of the (undirected) zone graph.

    The map graph is split into per-region clusters, so a navigation task
    is only solvable when start and goal share a component.
    """
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in graph:
        if node in seen:
            continue
        stack = [node]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            stack.extend(graph.get(cur, []))
        components.append(comp)
    return components


def _cell_action(x: float, y: float, grid_size: int) -> int:
    """Return the click-cell action whose cell contains point (x, y)."""
    col = min(grid_size - 1, max(0, int(x * grid_size)))
    row = min(grid_size - 1, max(0, int(y * grid_size)))
    return row * grid_size + col


def _sign(value: float, eps: float = 0.15) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


class RandomAgent:
    """Lower-bound baseline: uniform random actions."""

    def __init__(self, env: AlbionStateSim, seed: int = 0):
        self._action_count = env.action_count
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        return self._rng.randrange(self._action_count)

    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        pass


class GreedyAgent:
    """Heuristic baseline with full zone-graph knowledge.

    BFS over the zone graph picks the gate that leads toward the goal; the
    agent walks to that gate and interacts. It does not learn — it is the
    upper-bound reference for directed inter-map navigation. A short random
    "escape" burst is used to break free when an obstacle blocks the path.
    """

    def __init__(self, env: AlbionStateSim, seed: int = 0):
        self._env = env
        self._rng = random.Random(seed)
        self._prev_pos: tuple[float, float] | None = None
        self._stuck = 0
        self._escape = 0

    @property
    def name(self) -> str:
        return "greedy"

    def reset(self) -> None:
        self._prev_pos = None
        self._stuck = 0
        self._escape = 0

    def act(self, obs: dict) -> int:
        env = self._env
        zone = str(obs["zone_id"])
        goal = str(obs["goal_zone"])
        ax, ay = float(obs["x"]), float(obs["y"])

        if zone == goal:
            return env.move_action_count + int(Action.IDLE)

        # Obstacle escape: a wedged agent makes no progress, so jitter free.
        if self._prev_pos is not None:
            moved = math.hypot(ax - self._prev_pos[0], ay - self._prev_pos[1])
            self._stuck = self._stuck + 1 if moved < 0.01 else max(0, self._stuck - 1)
        self._prev_pos = (ax, ay)
        if self._escape > 0:
            self._escape -= 1
            return self._rng.randrange(env.move_action_count)
        if self._stuck >= 3:
            self._stuck = 0
            self._escape = 4
            return self._rng.randrange(env.move_action_count)

        graph = env.zone_manager.zone_graph
        next_zone = bfs_first_step(graph, zone, goal)
        neighbors = graph.get(zone, [])
        gates = env.zone_entities.get(zone, {}).get("gates", [])
        if not gates:
            return self._rng.randrange(env.move_action_count)

        if next_zone in neighbors and neighbors.index(next_zone) < len(gates):
            gx, gy = gates[neighbors.index(next_zone)]
        else:
            gx, gy = gates[0]

        if math.hypot(float(gx) - ax, float(gy) - ay) < 0.13:
            return env.move_action_count + int(Action.INTERACT)
        return _cell_action(float(gx), float(gy), env.grid_size)

    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        pass


class TabularQLearningAgent:
    """Tabular Q-learning over discretised egocentric observation features.

    No function approximation and no zone-graph knowledge: the policy is a
    lookup table keyed on coarse, observation-derived features. The agent
    only reacts to gates it can currently see, so it cannot perform goal-
    directed routing the way :class:`GreedyAgent` does — this limitation is
    the point of comparison.
    """

    def __init__(self, env: AlbionStateSim, seed: int = 0):
        self._action_count = env.action_count
        self._grid_size = env.grid_size
        self._rng = random.Random(seed)
        self._q: dict[tuple, list[float]] = {}
        self.trained = False

    @property
    def name(self) -> str:
        return "tabular_q"

    def _state_key(self, obs: dict) -> tuple:
        if str(obs["zone_id"]) == str(obs["goal_zone"]):
            return ("at_goal",)
        gate_dist = float(obs.get("nearest_gate_dist", 1.0))
        if gate_dist < 0.95:  # a gate is inside the observed view
            dx = float(obs.get("nearest_gate_dx", 0.0))
            dy = float(obs.get("nearest_gate_dy", 0.0))
            dist_bin = 0 if gate_dist < 0.13 else (1 if gate_dist < 0.30 else 2)
            return ("gate", _sign(dx), _sign(dy), dist_bin)
        cr = min(self._grid_size - 1, int(float(obs["y"]) * self._grid_size))
        cc = min(self._grid_size - 1, int(float(obs["x"]) * self._grid_size))
        return ("explore", cr, cc)

    def _q_row(self, key: tuple) -> list[float]:
        row = self._q.get(key)
        if row is None:
            row = [0.0] * self._action_count
            self._q[key] = row
        return row

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        row = self._q_row(self._state_key(obs))
        return max(range(self._action_count), key=lambda a: row[a])

    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        pass

    def train(
        self,
        episodes: int = 3000,
        alpha: float = 0.2,
        gamma: float = 0.97,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        max_steps: int = 200,
        seed: int = 0,
        observation_mode: str = "partial",
        log_interval: int = 0,
    ) -> "TabularQLearningAgent":
        """Fill the Q-table by interacting with a private environment.

        Only solvable (same connected component) navigation tasks are sampled.
        """
        env = AlbionStateSim(seed=seed, max_steps=max_steps)
        env.set_observation_mode(observation_mode)
        components = connected_components(env.zone_manager.zone_graph)
        zone_component = {z: i for i, comp in enumerate(components) for z in comp}
        zones = [z for z, t in env.zone_manager.zone_types.items() if t != "city"]
        rng = random.Random(seed + 1)
        for ep in range(episodes):
            start = rng.choice(zones)
            reachable = [
                z
                for z in zones
                if z != start and zone_component[z] == zone_component[start]
            ]
            goal = rng.choice(reachable) if reachable else start
            env.set_navigation_task(start, goal, True)
            obs = env.reset()
            eps = epsilon_end + (epsilon_start - epsilon_end) * max(
                0.0, 1.0 - ep / episodes
            )
            done = False
            while not done:
                key = self._state_key(obs)
                row = self._q_row(key)
                if rng.random() < eps:
                    action = rng.randrange(self._action_count)
                else:
                    action = max(range(self._action_count), key=lambda a: row[a])
                next_obs, reward, done, _info = env.step(action)
                next_row = self._q_row(self._state_key(next_obs))
                td_target = reward + (0.0 if done else gamma * max(next_row))
                row[action] += alpha * (td_target - row[action])
                obs = next_obs
            if log_interval and (ep + 1) % log_interval == 0:
                print(f"  tabular train episode={ep + 1}/{episodes} states={len(self._q)}")
        env.close()
        self.trained = True
        return self


class PPOAgent:
    """Deep RL agent: a PPO actor-critic loaded from a trained checkpoint."""

    def __init__(
        self,
        env: AlbionStateSim,
        checkpoint_path: str = DEFAULT_PPO_CHECKPOINT,
        deterministic: bool = True,
        seed: int = 0,
    ):
        import torch

        from state_sim.ppo import ActorCritic, ObsEncoder

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found: {path}. Train one first with "
                "`python -m state_sim.train_map_to_map_ppo`."
            )
        self._torch = torch
        ckpt = torch.load(path, map_location="cpu")
        self._encoder = ObsEncoder(env)
        obs_dim = int(ckpt.get("obs_dim", self._encoder.obs_dim))
        action_dim = int(ckpt.get("action_dim", env.action_count))
        self._model = ActorCritic(obs_dim, action_dim)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._deterministic = deterministic
        self.last_action_entropy = float("nan")
        torch.manual_seed(seed)

    @property
    def name(self) -> str:
        return "ppo"

    def reset(self) -> None:
        self._encoder.reset()

    def act(self, obs: dict) -> int:
        torch = self._torch
        vec = self._encoder.encode(obs)
        obs_t = torch.from_numpy(vec).float().unsqueeze(0)
        with torch.no_grad():
            logits, _value, _ = self._model(obs_t)
            probs = torch.softmax(logits, dim=-1)
            self.last_action_entropy = float(
                -(probs * torch.log(probs + 1e-9)).sum(dim=-1).item()
            )
            if self._deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            return int(torch.multinomial(probs, 1).item())

    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        pass


def build_agent(
    name: str,
    env: AlbionStateSim,
    *,
    seed: int = 0,
    ppo_checkpoint: str = DEFAULT_PPO_CHECKPOINT,
    tabular_train_episodes: int = 3000,
    observation_mode: str = "partial",
) -> AgentProtocol:
    """Construct an agent by name. Tabular agents are trained on creation."""
    key = name.lower()
    if key == "random":
        return RandomAgent(env, seed=seed)
    if key == "greedy":
        return GreedyAgent(env, seed=seed)
    if key in ("tabular", "tabular_q", "tabular_q_learning"):
        agent = TabularQLearningAgent(env, seed=seed)
        agent.train(
            episodes=tabular_train_episodes,
            seed=seed,
            observation_mode=observation_mode,
        )
        return agent
    if key == "ppo":
        return PPOAgent(env, ppo_checkpoint, seed=seed)
    raise ValueError(
        f"unknown agent: {name}. valid=['random', 'greedy', 'tabular_q', 'ppo']"
    )
