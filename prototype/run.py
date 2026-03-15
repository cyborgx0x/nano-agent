import argparse
import time

from env import GameEnv

# Local imports when running from inside the `prototype/` folder
from agent import Agent


def main(args):
    env = GameEnv(capture_region=None, fps=8)
    agent = Agent(device=args.device)

    print("Resetting environment, please focus the game window (5s)...")
    time.sleep(5)
    obs = env.reset()

    try:
        steps = 0
        while steps < args.steps:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            steps += 1
            if done:
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    main(args)
