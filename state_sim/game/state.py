"""State definitions for the Albion simulation."""

from dataclasses import dataclass


@dataclass
class SimState:
    zone_id: str
    zone_type: str
    biome: str
    x: float
    y: float
    hp: float
    mounted: bool
    inventory_load: float
    resource_value: float
    xp_value: float
    threat_score: float
    at_city: bool
    alive: bool
    step_count: int
    banked_value: float
    last_return_steps: int
