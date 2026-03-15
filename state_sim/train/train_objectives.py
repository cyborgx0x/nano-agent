from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from state_sim.ppo import PPOTrainConfig, train_ppo
from state_sim.train.evaluate_objective_checkpoint import evaluate_objective_checkpoint


def run_objective_benchmark(
    base_config: PPOTrainConfig,
    objectives: list[str],
    target_success: float,
    stop_on_fail: bool,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []

    for objective in objectives:
        config = copy.deepcopy(base_config)
        config.objective = objective
        if not config.use_wandb:
            config.wandb_mode = "disabled"

        print(f"\n=== Train objective: {objective} ===")
        outcome = train_ppo(config)
        best_success = float(outcome.get("best_success_window", 0.0))
        outcome["target_success"] = target_success
        outcome["passed"] = best_success >= target_success
        results.append(outcome)

        if stop_on_fail and best_success < target_success:
            print(
                f"STOP: objective '{objective}' did not reach target_success={target_success:.2f} "
                f"(best_success_window={best_success:.3f})."
            )
            break

    return results


def optimize_gather_first(
    base_config: PPOTrainConfig,
    target_success: float,
    max_trials: int,
) -> tuple[PPOTrainConfig, list[dict[str, float | str]], bool]:
    history: list[dict[str, float | str]] = []
    best_config = copy.deepcopy(base_config)
    best_score = -1.0
    best_row: dict[str, float | str] | None = None

    grid: list[dict[str, float | int]] = []
    for lr in [1e-4, 7e-5, 5e-5]:
        for entropy_coef in [0.005, 0.01, 0.02]:
            for clip_eps in [0.15, 0.12]:
                for update_every_episodes in [10, 5]:
                    for minibatch_size in [256, 128]:
                        for gather_teacher_start in [0.8, 0.9, 1.0]:
                            for gather_teacher_min in [0.2, 0.3, 0.4]:
                                grid.append(
                                    {
                                        "lr": lr,
                                        "entropy_coef": entropy_coef,
                                        "clip_eps": clip_eps,
                                        "update_every_episodes": update_every_episodes,
                                        "minibatch_size": minibatch_size,
                                        "gather_teacher_start": gather_teacher_start,
                                        "gather_teacher_min": gather_teacher_min,
                                    }
                                )

    rng = random.Random(base_config.seed)
    rng.shuffle(grid)
    trials = grid[: max(1, max_trials)]

    print(f"\n=== Gather-first optimization ({len(trials)} trials) ===")
    for idx, params in enumerate(trials, start=1):
        trial_cfg = copy.deepcopy(base_config)
        trial_cfg.objective = "gather"

        for key, value in params.items():
            setattr(trial_cfg, key, value)

        print(
            f"trial={idx}/{len(trials)} lr={trial_cfg.lr} entropy={trial_cfg.entropy_coef} "
            f"clip={trial_cfg.clip_eps} mb={trial_cfg.minibatch_size} up_every={trial_cfg.update_every_episodes} "
            f"teacher=({trial_cfg.gather_teacher_start},{trial_cfg.gather_teacher_min})"
        )
        outcome = train_ppo(trial_cfg)
        score = float(outcome.get("best_success_window", 0.0))

        row: dict[str, float | str] = {
            "objective": "gather",
            "trial": float(idx),
            "best_success_window": score,
            "final_success": float(outcome.get("final_success", 0.0)),
            "final_return": float(outcome.get("final_return", 0.0)),
            "lr": float(trial_cfg.lr),
            "entropy_coef": float(trial_cfg.entropy_coef),
            "clip_eps": float(trial_cfg.clip_eps),
            "minibatch_size": float(trial_cfg.minibatch_size),
            "update_every_episodes": float(trial_cfg.update_every_episodes),
            "gather_teacher_start": float(trial_cfg.gather_teacher_start),
            "gather_teacher_min": float(trial_cfg.gather_teacher_min),
            "passed": score >= target_success,
            "target_success": target_success,
        }
        history.append(row)

        if score > best_score:
            best_score = score
            best_config = copy.deepcopy(trial_cfg)
            best_row = row

        if score >= target_success:
            print(
                f"gather reached target_success={target_success:.2f} at trial={idx} with best_success_window={score:.3f}"
            )
            return best_config, history, True

    if best_row is not None:
        print(
            f"gather target not reached. best_success_window={best_score:.3f} "
            f"(target={target_success:.2f})"
        )
    return best_config, history, False


def _trial_grid_for_objective(
    objective: str,
    seed: int,
    max_trials: int,
) -> list[dict[str, float | int]]:
    objective = objective.lower()
    grid: list[dict[str, float | int]] = []

    if objective == "gather":
        for lr in [1e-4, 7e-5, 5e-5]:
            for entropy_coef in [0.005, 0.01, 0.02]:
                for clip_eps in [0.15, 0.12]:
                    for update_every_episodes in [10, 5]:
                        for minibatch_size in [256, 128]:
                            for gather_teacher_start in [0.8, 0.9, 1.0]:
                                for gather_teacher_min in [0.2, 0.3, 0.4]:
                                    grid.append(
                                        {
                                            "lr": lr,
                                            "entropy_coef": entropy_coef,
                                            "clip_eps": clip_eps,
                                            "update_every_episodes": update_every_episodes,
                                            "minibatch_size": minibatch_size,
                                            "gather_teacher_start": gather_teacher_start,
                                            "gather_teacher_min": gather_teacher_min,
                                        }
                                    )
    elif objective == "navigation":
        for lr in [1e-4, 7e-5, 5e-5]:
            for entropy_coef in [0.003, 0.005, 0.01]:
                for clip_eps in [0.15, 0.12]:
                    for update_every_episodes in [10, 5]:
                        for minibatch_size in [256, 128]:
                            for nav_teacher_start in [0.4, 0.6, 0.8]:
                                for nav_teacher_min in [0.05, 0.1, 0.2]:
                                    grid.append(
                                        {
                                            "lr": lr,
                                            "entropy_coef": entropy_coef,
                                            "clip_eps": clip_eps,
                                            "update_every_episodes": update_every_episodes,
                                            "minibatch_size": minibatch_size,
                                            "nav_teacher_start": nav_teacher_start,
                                            "nav_teacher_min": nav_teacher_min,
                                        }
                                    )
    elif objective == "mixed":
        for lr in [1e-4, 7e-5, 5e-5]:
            for entropy_coef in [0.005, 0.01, 0.015]:
                for clip_eps in [0.15, 0.12]:
                    for update_every_episodes in [10, 5]:
                        for minibatch_size in [256, 128]:
                            for mixed_teacher_start in [0.5, 0.7, 0.9]:
                                for mixed_teacher_min in [0.1, 0.2, 0.3]:
                                    for mixed_navigation_ratio in [0.5, 0.65, 0.8]:
                                        grid.append(
                                            {
                                                "lr": lr,
                                                "entropy_coef": entropy_coef,
                                                "clip_eps": clip_eps,
                                                "update_every_episodes": update_every_episodes,
                                                "minibatch_size": minibatch_size,
                                                "mixed_teacher_start": mixed_teacher_start,
                                                "mixed_teacher_min": mixed_teacher_min,
                                                "mixed_navigation_ratio": mixed_navigation_ratio,
                                            }
                                        )
    else:
        raise ValueError(
            f"unsupported objective={objective}. expected gather|navigation|mixed"
        )

    rng = random.Random(seed)
    rng.shuffle(grid)
    return grid[: max(1, max_trials)]


def optimize_objective(
    base_config: PPOTrainConfig,
    objective: str,
    target_success: float,
    max_trials: int,
    strict_eval_episodes: int,
    strict_eval_threshold: float,
) -> tuple[PPOTrainConfig, dict[str, float | str], bool, list[dict[str, float | str]]]:
    history: list[dict[str, float | str]] = []
    best_config = copy.deepcopy(base_config)
    best_score = -1.0
    best_row: dict[str, float | str] = {
        "objective": objective,
        "final_success": 0.0,
        "best_success_window": 0.0,
        "final_return": 0.0,
        "passed": False,
    }

    trials = _trial_grid_for_objective(objective, base_config.seed, max_trials)

    print(f"\n=== Optimize objective: {objective} ({len(trials)} trials) ===")
    for idx, params in enumerate(trials, start=1):
        trial_cfg = copy.deepcopy(base_config)
        trial_cfg.objective = objective
        for key, value in params.items():
            setattr(trial_cfg, key, value)

        print(
            f"trial={idx}/{len(trials)} lr={trial_cfg.lr} entropy={trial_cfg.entropy_coef} "
            f"clip={trial_cfg.clip_eps} mb={trial_cfg.minibatch_size} up_every={trial_cfg.update_every_episodes}"
        )

        outcome = train_ppo(trial_cfg)
        score = float(outcome.get("best_success_window", 0.0))

        row: dict[str, float | str] = {
            "objective": objective,
            "trial": float(idx),
            "best_success_window": score,
            "final_success": float(outcome.get("final_success", 0.0)),
            "final_return": float(outcome.get("final_return", 0.0)),
            "tail_success_window": float(outcome.get("tail_success_window", 0.0)),
            "lr": float(trial_cfg.lr),
            "entropy_coef": float(trial_cfg.entropy_coef),
            "clip_eps": float(trial_cfg.clip_eps),
            "minibatch_size": float(trial_cfg.minibatch_size),
            "update_every_episodes": float(trial_cfg.update_every_episodes),
            "strict_eval_success": 0.0,
            "passed": False,
            "target_success": target_success,
        }

        if score >= target_success and strict_eval_episodes > 0:
            checkpoint_path = str(outcome.get("checkpoint_path", ""))
            if checkpoint_path:
                eval_metrics = evaluate_objective_checkpoint(
                    checkpoint_path=checkpoint_path,
                    objective=objective,
                    episodes=strict_eval_episodes,
                    max_steps=trial_cfg.max_steps,
                    seed=trial_cfg.seed + 10_000 + idx,
                    mixed_navigation_ratio=trial_cfg.mixed_navigation_ratio,
                )
                row["strict_eval_success"] = float(eval_metrics["eval_success"])
                row["strict_eval_return"] = float(eval_metrics["eval_return"])
                row["strict_eval_length"] = float(eval_metrics["eval_length"])

        row["passed"] = bool(
            score >= target_success
            and float(row.get("strict_eval_success", 0.0)) >= strict_eval_threshold
        )
        history.append(row)

        if score > best_score:
            best_score = score
            best_config = copy.deepcopy(trial_cfg)
            best_row = row

        if bool(row["passed"]):
            print(
                f"{objective} reached gates: train>={target_success:.2f} and strict_eval>={strict_eval_threshold:.2f} "
                f"at trial={idx} (best_success_window={score:.3f}, strict_eval_success={float(row['strict_eval_success']):.3f})"
            )
            return best_config, best_row, True, history

    print(
        f"{objective} target not reached. best_success_window={best_score:.3f} "
        f"(train_target={target_success:.2f}, strict_eval_target={strict_eval_threshold:.2f})"
    )
    return best_config, best_row, False, history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO separately for navigation/gather/mixed objectives"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-every-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--mixed-navigation-ratio", type=float, default=0.5)
    parser.add_argument(
        "--gather-train-mode",
        type=str,
        default="multi",
        choices=["multi", "single"],
    )
    parser.add_argument(
        "--gather-zone-scheduler",
        type=str,
        default="round_robin",
        choices=["round_robin", "random"],
    )
    parser.add_argument("--gather-holdout-ratio", type=float, default=0.2)
    parser.add_argument("--gather-eval-interval", type=int, default=30)
    parser.add_argument("--gather-eval-episodes", type=int, default=30)
    parser.add_argument("--gather-teacher-min", type=float, default=0.35)
    parser.add_argument("--gather-teacher-decay-power", type=float, default=2.0)
    parser.add_argument("--gather-teacher-recovery-boost", type=float, default=0.20)
    parser.add_argument("--gather-teacher-recovery-ratio", type=float, default=0.75)
    parser.add_argument("--gather-teacher-boost-decay", type=float, default=0.96)
    parser.add_argument("--inventory-full-target", type=int, default=99)
    parser.add_argument("--memory-size", type=int, default=0)
    parser.add_argument(
        "--objectives", nargs="+", default=["navigation", "gather", "mixed"]
    )
    parser.add_argument("--target-success", type=float, default=0.95)
    parser.add_argument("--no-stop-on-fail", action="store_true")
    parser.add_argument("--gather-first", action="store_true")
    parser.add_argument("--gather-max-trials", type=int, default=12)
    parser.add_argument("--navigation-max-trials", type=int, default=12)
    parser.add_argument("--mixed-max-trials", type=int, default=12)
    parser.add_argument("--strict-eval-episodes", type=int, default=60)
    parser.add_argument("--strict-eval-threshold", type=float, default=0.85)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nano-agent-state-sim")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--checkpoint-dir", type=str, default="runs/state_sim")
    return parser.parse_args()


def _print_summary(results: list[dict[str, float | str]]) -> None:
    print("\n=== Objective Accuracy Summary ===")
    print(
        "objective      final_success   best_success_window   tail_success   strict_eval   passed"
    )
    print(
        "----------------------------------------------------------------------------"
    )
    for row in results:
        objective = str(row.get("objective", "unknown"))
        final_success = float(row.get("final_success", 0.0))
        best_success_window = float(row.get("best_success_window", 0.0))
        tail_success = float(row.get("tail_success_window", 0.0))
        strict_eval = float(row.get("strict_eval_success", 0.0))
        passed = bool(row.get("passed", False))
        print(
            f"{objective:<13} {final_success:>11.3f} {best_success_window:>20.3f} {tail_success:>14.3f} {strict_eval:>12.3f} {str(passed):>8}"
        )


def _print_gather_optimization_summary(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    print("\n=== Gather Optimization Trials ===")
    print(
        "trial  best_success_window  final_success  lr      entropy  clip   mb  up_every  passed"
    )
    print(
        "-------------------------------------------------------------------------------------------"
    )
    for row in rows:
        print(
            f"{int(float(row['trial'])):>5}"
            f"{float(row['best_success_window']):>21.3f}"
            f"{float(row['final_success']):>15.3f}"
            f"{float(row['lr']):>9.1e}"
            f"{float(row['entropy_coef']):>9.3f}"
            f"{float(row['clip_eps']):>7.3f}"
            f"{int(float(row['minibatch_size'])):>4}"
            f"{int(float(row['update_every_episodes'])):>10}"
            f"{str(bool(row['passed'])):>8}"
        )


def _print_objective_trials_summary(
    objective: str, rows: list[dict[str, float | str]]
) -> None:
    if not rows:
        return
    print(f"\n=== {objective.capitalize()} Optimization Trials ===")
    print(
        "trial  best_success_window  final_success  tail_success  strict_eval  passed"
    )
    print("--------------------------------------------------------------------------")
    for row in rows:
        print(
            f"{int(float(row['trial'])):>5}"
            f"{float(row['best_success_window']):>21.3f}"
            f"{float(row['final_success']):>15.3f}"
            f"{float(row.get('tail_success_window', 0.0)):>14.3f}"
            f"{float(row.get('strict_eval_success', 0.0)):>13.3f}"
            f"{str(bool(row['passed'])):>8}"
        )


def main() -> None:
    args = _parse_args()

    base_config = PPOTrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        update_every_episodes=args.update_every_episodes,
        log_interval=args.log_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        checkpoint_dir=args.checkpoint_dir,
        mixed_navigation_ratio=args.mixed_navigation_ratio,
        gather_train_mode=args.gather_train_mode,
        gather_zone_scheduler=args.gather_zone_scheduler,
        gather_holdout_ratio=args.gather_holdout_ratio,
        gather_eval_interval=args.gather_eval_interval,
        gather_eval_episodes=args.gather_eval_episodes,
        gather_teacher_min=args.gather_teacher_min,
        gather_teacher_decay_power=args.gather_teacher_decay_power,
        gather_teacher_recovery_boost=args.gather_teacher_recovery_boost,
        gather_teacher_recovery_ratio=args.gather_teacher_recovery_ratio,
        gather_teacher_boost_decay=args.gather_teacher_boost_decay,
        inventory_full_target=args.inventory_full_target,
        memory_size=args.memory_size,
    )

    objectives = [obj.lower() for obj in args.objectives]
    stop_on_fail = not args.no_stop_on_fail

    all_results: list[dict[str, float | str]] = []

    if args.gather_first:
        tuned_config, gather_row, gather_passed, gather_history = optimize_objective(
            base_config,
            objective="gather",
            target_success=args.target_success,
            max_trials=args.gather_max_trials,
            strict_eval_episodes=args.strict_eval_episodes,
            strict_eval_threshold=args.strict_eval_threshold,
        )
        _print_gather_optimization_summary(gather_history)

        all_results.append(gather_row)

        if stop_on_fail and not gather_passed:
            _print_summary(all_results)
            raise SystemExit(2)

        remaining = [obj for obj in objectives if obj != "gather"]
        current_config = tuned_config
        for objective in remaining:
            trial_cap = (
                args.navigation_max_trials
                if objective == "navigation"
                else args.mixed_max_trials
            )
            current_config, objective_row, objective_passed, objective_history = (
                optimize_objective(
                    current_config,
                    objective=objective,
                    target_success=args.target_success,
                    max_trials=trial_cap,
                    strict_eval_episodes=args.strict_eval_episodes,
                    strict_eval_threshold=args.strict_eval_threshold,
                )
            )
            _print_objective_trials_summary(objective, objective_history)
            all_results.append(objective_row)

            if stop_on_fail and not objective_passed:
                _print_summary(all_results)
                raise SystemExit(2)
    else:
        all_results = run_objective_benchmark(
            base_config,
            objectives,
            target_success=args.target_success,
            stop_on_fail=stop_on_fail,
        )

    _print_summary(all_results)

    if stop_on_fail and all_results:
        last_passed = bool(all_results[-1].get("passed", False))
        required_len = len(
            [obj for obj in objectives if (not args.gather_first) or obj != "gather"]
        ) + (1 if args.gather_first else 0)
        ran_all = len(all_results) == required_len
        if not (last_passed and ran_all):
            raise SystemExit(2)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    main()
