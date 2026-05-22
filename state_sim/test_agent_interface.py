"""Smoke test for the agent interface (Epic #1, issue #3).

Checks that every adapter satisfies ``AgentProtocol`` and can drive the
environment for a handful of steps without error.

Run: ``python -m state_sim.test_agent_interface``
"""

from __future__ import annotations

from state_sim.agent_interface import (
    AgentProtocol,
    GreedyAgent,
    RandomAgent,
    TabularQLearningAgent,
)
from state_sim.environment import AlbionStateSim


def _run_steps(agent: AgentProtocol, env: AlbionStateSim, steps: int = 10) -> None:
    agent.reset()
    obs = env.reset()
    for _ in range(steps):
        action = agent.act(obs)
        assert 0 <= action < env.action_count, f"{agent.name}: bad action {action}"
        obs, reward, done, info = env.step(action)
        agent.observe(obs, reward, done, info)
        if done:
            obs = env.reset()
            agent.reset()


def main() -> None:
    env = AlbionStateSim(seed=7, max_steps=60)
    env.set_navigation_task("forest_cross", "green_forest_1", True)

    agents: list[AgentProtocol] = [
        RandomAgent(env, seed=0),
        GreedyAgent(env, seed=0),
        TabularQLearningAgent(env, seed=0),  # untrained: still a valid adapter
    ]

    for agent in agents:
        assert isinstance(agent, AgentProtocol), f"{agent.name} fails AgentProtocol"
        _run_steps(agent, env, steps=10)
        print(f"OK  {agent.name:12s} isinstance(AgentProtocol)=True, 10 steps ran")

    env.close()
    print("all agent-interface smoke tests passed")


if __name__ == "__main__":
    main()
