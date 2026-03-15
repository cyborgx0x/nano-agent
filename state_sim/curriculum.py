from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    min_success_rate: float = 0.85
    max_death_rate: float = 0.05
    max_avg_return_steps: int = 140
    reward_resource: float = 1.0
    reward_xp: float = 0.0
    reward_bank: float = 1.2
    penalty_damage: float = 0.9
    penalty_idle: float = 0.02
    penalty_death: float = 12.0
    step_penalty: float = 0.003


def default_curriculum() -> list[PhaseConfig]:
    return [
        PhaseConfig(
            name="phase_0_safety_bootstrap",
            reward_resource=0.0,
            reward_xp=0.0,
            reward_bank=0.4,
            penalty_damage=1.4,
            penalty_idle=0.03,
            penalty_death=16.0,
            max_avg_return_steps=120,
        ),
        PhaseConfig(
            name="phase_1_in_zone_navigation",
            reward_resource=0.2,
            reward_xp=0.0,
            reward_bank=0.6,
            penalty_damage=1.2,
            penalty_idle=0.02,
            penalty_death=14.0,
            max_avg_return_steps=120,
        ),
        PhaseConfig(
            name="phase_2_inter_zone_transition",
            reward_resource=0.2,
            reward_xp=0.0,
            reward_bank=0.7,
            penalty_damage=1.1,
            penalty_idle=0.02,
            penalty_death=14.0,
            max_avg_return_steps=130,
        ),
        PhaseConfig(
            name="phase_3_single_gather_loop",
            reward_resource=1.0,
            reward_xp=0.0,
            reward_bank=1.2,
            penalty_damage=1.0,
            penalty_idle=0.02,
            penalty_death=13.0,
            max_avg_return_steps=135,
        ),
        PhaseConfig(
            name="phase_4_gather_with_threat",
            reward_resource=1.1,
            reward_xp=0.0,
            reward_bank=1.3,
            penalty_damage=1.0,
            penalty_idle=0.02,
            penalty_death=12.5,
            max_avg_return_steps=140,
        ),
        PhaseConfig(
            name="phase_5_economy_loop",
            reward_resource=1.2,
            reward_xp=0.1,
            reward_bank=1.4,
            penalty_damage=0.95,
            penalty_idle=0.02,
            penalty_death=12.0,
            max_avg_return_steps=145,
        ),
        PhaseConfig(
            name="phase_6_xp_integration",
            reward_resource=1.0,
            reward_xp=0.8,
            reward_bank=1.3,
            penalty_damage=0.95,
            penalty_idle=0.02,
            penalty_death=12.0,
            max_avg_return_steps=150,
        ),
        PhaseConfig(
            name="phase_7_mixed_random_scenarios",
            reward_resource=1.0,
            reward_xp=1.0,
            reward_bank=1.5,
            penalty_damage=0.9,
            penalty_idle=0.02,
            penalty_death=11.0,
            max_avg_return_steps=150,
        ),
    ]


def should_promote(phase: PhaseConfig, metrics: dict[str, float]) -> bool:
    success_rate = metrics.get("success_rate", 0.0)
    death_rate = metrics.get("death_rate", 1.0)
    avg_return_steps = metrics.get("avg_return_steps", 10_000.0)

    return (
        success_rate >= phase.min_success_rate
        and death_rate <= phase.max_death_rate
        and avg_return_steps <= phase.max_avg_return_steps
    )
