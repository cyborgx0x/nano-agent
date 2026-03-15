Prototype: minimal play interface and pretrained-encoder agent

What this provides
- `GameEnv` (`prototype/env.py`): minimal Gym-like wrapper that captures screen and can send simple key presses.
- `Agent` (`prototype/agent.py`): loads a pretrained `resnet18` encoder (ImageNet) and a tiny untrained policy head. This satisfies the requirement to use a pretrained model for perception.
- `run.py`: demo runner that resets, then runs the agent loop. The policy is untrained; it's scaffold for integrating BC or RL later.

Mouse support
- `GameEnv` now supports a small set of mouse actions (relative moves, left/right click, move-to-center). The discrete action mapping in the prototype is:

	0: noop
	1: key 'w'
	2: key 's'
	3: key 'a'
	4: key 'd'
	5: key 'space'
	6: mouse move up
	7: mouse move down
	8: mouse move left
	9: mouse move right
 10: mouse left click
 11: mouse right click
 12: mouse move to capture center

Use these indices when mapping policy outputs to actions. The `Agent` policy head was expanded to match the action count.

Quick start
1. Install requirements (recommended in a venv):

```bash
pip install -r requirements.txt
```

2. Focus your game window and run:

```bash
python prototype/run.py --device cpu --steps 1000
```

Notes
- `mss` is required for screen capture and `pyautogui` for sending keys. On some systems additional permissions are required to capture the screen.
- The policy in this scaffold is intentionally untrained. Replace `prototype/agent.py` policy weights after training or provide a saved policy via `Agent.load_policy()`.
- This is a minimal prototype to get playing loops working quickly; next steps are: collect instrumented demos, train inverse dynamics, and bootstrap BC/Dreamer.
