"""Entity management for the Albion simulation."""

from __future__ import annotations

import math
import random


class EntityManager:
    """Manages spawning and tracking of resources, mobs, and obstacles."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._next_resource_id = 1

    def resource_tier_band(self, zone_type: str) -> tuple[int, int]:
        """Return the resource tier range for a zone type."""
        if zone_type == "blue":
            return (2, 5)
        if zone_type == "yellow":
            return (3, 6)
        if zone_type == "red":
            return (4, 7)
        if zone_type == "black":
            return (2, 8)
        return (1, 2)

    def sample_resource_type(
        self, biome: str, biome_resource_profile: dict[str, tuple[str, str, str]]
    ) -> str:
        """Sample a resource type based on biome."""
        primary, secondary, tertiary = biome_resource_profile[biome]
        roll = self.rng.random()
        if roll < 0.55:
            return primary
        if roll < 0.85:
            return secondary
        return tertiary

    def spawn_resource_node(
        self,
        zone_type: str,
        biome: str,
        biome_resource_profile: dict[str, tuple[str, str, str]],
        sample_point_func,
    ) -> dict[str, object]:
        """Spawn a resource node."""
        min_tier, max_tier = self.resource_tier_band(zone_type)
        if self.rng.random() < 0.16:
            resource_type = self.rng.choice(["rough_stone", "rough_logs"])
            tier = 1
            charges = self.rng.randint(6, 12)
        else:
            resource_type = self.sample_resource_type(biome, biome_resource_profile)
            tier = self.rng.randint(min_tier, max_tier)
            charges = (
                30
                if self.rng.random() < 0.10 and tier >= 4
                else self.rng.randint(3, 11)
            )

        enchant = 0
        if tier >= 4:
            eroll = self.rng.random()
            if eroll < 0.13:
                enchant = 1
            elif eroll < 0.16:
                enchant = 2
            elif eroll < 0.168:
                enchant = 3
            elif eroll < 0.170:
                enchant = 4

        node = {
            "id": self._next_resource_id,
            "pos": sample_point_func(margin=0.06),
            "resource_type": resource_type,
            "tier": tier,
            "enchant": enchant,
            "charges": charges,
            "plentiful": charges >= 30,
        }
        self._next_resource_id += 1
        return node

    def spawn_obstacle(self, biome: str, sample_point_func) -> dict[str, object]:
        """Spawn an obstacle."""
        kind_pool = {
            "forest": ["large_tree", "thick_tree", "boulder"],
            "highland": ["boulder", "rock_cluster", "rough_stone_patch"],
            "mountain": ["rock_cluster", "cliff_chunk", "rough_stone_patch"],
            "steppe": ["dry_tree", "boulder", "rock_cluster"],
            "swamp": ["swamp_tree", "mud_rock", "large_root"],
        }
        kind = self.rng.choice(kind_pool[biome])
        return {
            "pos": sample_point_func(margin=0.05),
            "radius": self.rng.uniform(0.045, 0.09),
            "kind": kind,
            "blocks_vision": kind not in {"mud_rock", "rough_stone_patch"},
        }

    def spawn_obstacle_with_clearance(
        self,
        biome: str,
        resource_points: list[tuple[float, float]],
        mob_points: list[tuple[float, float]],
        gate_points: list[tuple[float, float]],
        existing_obstacles: list[dict[str, object]],
        sample_point_func,
    ) -> dict[str, object] | None:
        """Spawn an obstacle with clearance from other entities."""
        kind_pool = {
            "forest": ["large_tree", "thick_tree", "boulder"],
            "highland": ["boulder", "rock_cluster", "rough_stone_patch"],
            "mountain": ["rock_cluster", "cliff_chunk", "rough_stone_patch"],
            "steppe": ["dry_tree", "boulder", "rock_cluster"],
            "swamp": ["swamp_tree", "mud_rock", "large_root"],
        }

        for _ in range(240):
            kind = self.rng.choice(kind_pool[biome])
            radius = self.rng.uniform(0.045, 0.09)
            x, y = sample_point_func(margin=0.05)

            clear_of_resources = all(
                math.hypot(x - px, y - py) >= radius + 0.050
                for px, py in resource_points
            )
            if not clear_of_resources:
                continue

            clear_of_mobs = all(
                math.hypot(x - px, y - py) >= radius + 0.060 for px, py in mob_points
            )
            if not clear_of_mobs:
                continue

            clear_of_gates = all(
                math.hypot(x - px, y - py) >= radius + 0.095 for px, py in gate_points
            )
            if not clear_of_gates:
                continue

            clear_of_obstacles = all(
                math.hypot(x - float(obs["pos"][0]), y - float(obs["pos"][1]))
                >= radius + float(obs["radius"]) + 0.020
                for obs in existing_obstacles
            )
            if not clear_of_obstacles:
                continue

            return {
                "pos": (x, y),
                "radius": radius,
                "kind": kind,
                "blocks_vision": kind not in {"mud_rock", "rough_stone_patch"},
            }

        return None

    def resource_value_per_charge(
        self, resource_type: str, tier: int, enchant: int
    ) -> float:
        """Calculate the value per charge of a resource."""
        base = {
            "fiber": 3.0,
            "wood": 2.8,
            "stone": 2.6,
            "ore": 3.2,
            "hide": 3.0,
            "rough_stone": 1.0,
            "rough_logs": 1.0,
        }.get(resource_type, 2.4)
        return base * (1.0 + tier * 0.34) * (1.0 + 0.55 * enchant)

    @staticmethod
    def resource_color(resource_type: str) -> tuple[int, int, int]:
        """Return RGB color for a resource type."""
        return {
            "fiber": (80, 220, 80),
            "wood": (30, 140, 70),
            "stone": (130, 130, 130),
            "ore": (170, 120, 80),
            "hide": (60, 120, 180),
            "rough_stone": (160, 160, 160),
            "rough_logs": (50, 110, 60),
        }.get(resource_type, (80, 200, 80))
