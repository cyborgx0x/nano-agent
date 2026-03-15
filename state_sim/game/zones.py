"""Zone management for the Albion simulation."""

from __future__ import annotations

import math
import random


class ZoneManager:
    """Manages zone graph, types, biomes, and layouts."""

    def __init__(self, rng: random.Random, map_l1_radius: float = 0.47):
        self.rng = rng
        self.map_l1_radius = map_l1_radius

        self.zone_graph = {
            "lymhurst": ["forest_cross", "forest_gate"],
            "bridgewatch": ["desert_gate"],
            "martlock": ["mountain_gate"],
            "thetford": ["swamp_gate"],
            "fort_sterling": ["snow_gate"],
            "forest_cross": ["lymhurst", "green_forest_1", "yellow_forest_1"],
            "green_forest_1": ["forest_cross", "yellow_forest_1"],
            "yellow_forest_1": ["forest_cross", "green_forest_1", "yellow_forest_2"],
            "yellow_forest_2": ["yellow_forest_1", "forest_gate"],
            "forest_gate": ["yellow_forest_2", "lymhurst"],
            "desert_gate": ["bridgewatch"],
            "mountain_gate": ["martlock"],
            "swamp_gate": ["thetford"],
            "snow_gate": ["fort_sterling"],
        }

        self.zone_types = {
            "lymhurst": "city",
            "bridgewatch": "city",
            "martlock": "city",
            "thetford": "city",
            "fort_sterling": "city",
            "forest_cross": "blue",
            "green_forest_1": "blue",
            "yellow_forest_1": "yellow",
            "yellow_forest_2": "yellow",
            "forest_gate": "blue",
            "desert_gate": "blue",
            "mountain_gate": "blue",
            "swamp_gate": "blue",
            "snow_gate": "blue",
        }

        self.zone_biomes = {
            "lymhurst": "forest",
            "bridgewatch": "steppe",
            "martlock": "highland",
            "thetford": "swamp",
            "fort_sterling": "mountain",
            "forest_cross": "forest",
            "green_forest_1": "forest",
            "yellow_forest_1": "forest",
            "yellow_forest_2": "forest",
            "forest_gate": "forest",
            "desert_gate": "steppe",
            "mountain_gate": "mountain",
            "swamp_gate": "swamp",
            "snow_gate": "mountain",
        }

        self.biome_resource_profile = {
            "forest": ("wood", "hide", "stone"),
            "highland": ("stone", "ore", "wood"),
            "mountain": ("ore", "stone", "fiber"),
            "steppe": ("hide", "fiber", "ore"),
            "swamp": ("fiber", "wood", "hide"),
        }

        self.zone_layout = {
            "lymhurst": (0.42, 0.16),
            "bridgewatch": (0.84, 0.34),
            "martlock": (0.70, 0.78),
            "thetford": (0.22, 0.78),
            "fort_sterling": (0.12, 0.36),
            "forest_cross": (0.46, 0.34),
            "green_forest_1": (0.38, 0.48),
            "yellow_forest_1": (0.54, 0.46),
            "yellow_forest_2": (0.62, 0.56),
            "forest_gate": (0.58, 0.26),
            "desert_gate": (0.78, 0.24),
            "mountain_gate": (0.78, 0.68),
            "swamp_gate": (0.28, 0.70),
            "snow_gate": (0.20, 0.24),
        }

        self._extend_gather_zone_pool()

    def register_zone(
        self,
        zone_id: str,
        neighbors: list[str],
        zone_type: str,
        biome: str,
        layout: tuple[float, float],
    ) -> None:
        """Register a new zone and its connections."""
        self.zone_graph.setdefault(zone_id, [])
        for neighbor in neighbors:
            self.zone_graph.setdefault(neighbor, [])
            if neighbor not in self.zone_graph[zone_id]:
                self.zone_graph[zone_id].append(neighbor)
            if zone_id not in self.zone_graph[neighbor]:
                self.zone_graph[neighbor].append(zone_id)

        self.zone_types[zone_id] = zone_type
        self.zone_biomes[zone_id] = biome
        self.zone_layout[zone_id] = layout

    def _extend_gather_zone_pool(self) -> None:
        """Add gathering zones to the zone graph."""
        gather_zone_specs: list[
            tuple[str, list[str], str, str, tuple[float, float]]
        ] = [
            (
                "forest_gather_1",
                ["forest_cross", "forest_gather_2"],
                "blue",
                "forest",
                (0.44, 0.28),
            ),
            (
                "forest_gather_2",
                ["forest_gather_1", "forest_gather_3"],
                "yellow",
                "forest",
                (0.49, 0.30),
            ),
            (
                "forest_gather_3",
                ["forest_gather_2", "forest_gather_4", "yellow_forest_2"],
                "yellow",
                "forest",
                (0.55, 0.34),
            ),
            (
                "forest_gather_4",
                ["forest_gather_3", "forest_gather_5", "forest_gate"],
                "red",
                "forest",
                (0.61, 0.32),
            ),
            (
                "forest_gather_5",
                ["forest_gather_4", "forest_gather_6"],
                "red",
                "forest",
                (0.66, 0.36),
            ),
            (
                "forest_gather_6",
                ["forest_gather_5"],
                "black",
                "forest",
                (0.70, 0.40),
            ),
            (
                "steppe_gather_1",
                ["desert_gate", "steppe_gather_2"],
                "blue",
                "steppe",
                (0.82, 0.28),
            ),
            (
                "steppe_gather_2",
                ["steppe_gather_1", "steppe_gather_3"],
                "yellow",
                "steppe",
                (0.87, 0.31),
            ),
            (
                "steppe_gather_3",
                ["steppe_gather_2", "steppe_gather_4"],
                "red",
                "steppe",
                (0.90, 0.36),
            ),
            (
                "steppe_gather_4",
                ["steppe_gather_3"],
                "black",
                "steppe",
                (0.92, 0.42),
            ),
            (
                "swamp_gather_1",
                ["swamp_gate", "swamp_gather_2"],
                "blue",
                "swamp",
                (0.30, 0.74),
            ),
            (
                "swamp_gather_2",
                ["swamp_gather_1", "swamp_gather_3"],
                "yellow",
                "swamp",
                (0.25, 0.78),
            ),
            (
                "swamp_gather_3",
                ["swamp_gather_2", "swamp_gather_4"],
                "red",
                "swamp",
                (0.19, 0.80),
            ),
            (
                "swamp_gather_4",
                ["swamp_gather_3"],
                "black",
                "swamp",
                (0.14, 0.76),
            ),
            (
                "mountain_gather_1",
                ["mountain_gate", "mountain_gather_2"],
                "blue",
                "mountain",
                (0.74, 0.71),
            ),
            (
                "mountain_gather_2",
                ["mountain_gather_1", "mountain_gather_3"],
                "yellow",
                "mountain",
                (0.78, 0.75),
            ),
            (
                "mountain_gather_3",
                ["mountain_gather_2", "mountain_gather_4"],
                "red",
                "highland",
                (0.73, 0.82),
            ),
            (
                "mountain_gather_4",
                ["mountain_gather_3"],
                "black",
                "highland",
                (0.67, 0.86),
            ),
            (
                "snow_gather_1",
                ["snow_gate", "snow_gather_2"],
                "blue",
                "mountain",
                (0.16, 0.28),
            ),
            (
                "snow_gather_2",
                ["snow_gather_1", "snow_gather_3"],
                "yellow",
                "mountain",
                (0.12, 0.22),
            ),
            (
                "snow_gather_3",
                ["snow_gather_2", "snow_gather_4"],
                "red",
                "highland",
                (0.10, 0.17),
            ),
            (
                "snow_gather_4",
                ["snow_gather_3"],
                "black",
                "highland",
                (0.08, 0.12),
            ),
        ]

        for zone_id, neighbors, zone_type, biome, layout in gather_zone_specs:
            self.register_zone(zone_id, neighbors, zone_type, biome, layout)

    def rotated_square_gate_slots(self) -> list[tuple[float, float]]:
        """Return positions for gates in a rotated square layout."""
        half_edge = self.map_l1_radius / 2.0
        return [
            (0.5 + half_edge, 0.5 - half_edge),
            (0.5 + half_edge, 0.5 + half_edge),
            (0.5 - half_edge, 0.5 + half_edge),
            (0.5 - half_edge, 0.5 - half_edge),
        ]

    def in_diamond(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is within the diamond-shaped zone boundary."""
        radius = max(0.02, self.map_l1_radius - margin)
        return abs(x - 0.5) + abs(y - 0.5) <= radius

    def project_to_diamond(self, x: float, y: float) -> tuple[float, float]:
        """Project a point to be within the diamond boundary."""
        dx = x - 0.5
        dy = y - 0.5
        l1 = abs(dx) + abs(dy)
        if l1 <= self.map_l1_radius:
            return x, y
        scale = self.map_l1_radius / max(1e-6, l1)
        nx = 0.5 + dx * scale
        ny = 0.5 + dy * scale
        return min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))

    def sample_point_in_diamond(self, margin: float = 0.08) -> tuple[float, float]:
        """Sample a random point within the diamond boundary."""
        for _ in range(64):
            x = self.rng.uniform(0.08, 0.92)
            y = self.rng.uniform(0.08, 0.92)
            if self.in_diamond(x, y, margin=margin):
                return x, y
        return self.project_to_diamond(
            self.rng.uniform(0.08, 0.92), self.rng.uniform(0.08, 0.92)
        )

    def zone_risk(self, zone_type: str) -> float:
        """Return the risk level for a zone type."""
        if zone_type == "city":
            return 0.0
        if zone_type == "blue":
            return 0.02
        if zone_type == "yellow":
            return 0.04
        if zone_type == "red":
            return 0.06
        return 0.10

    def zone_color(self, zone_type: str) -> tuple[int, int, int]:
        """Return RGB color for a zone type."""
        if zone_type == "city":
            return (60, 80, 130)
        if zone_type == "blue":
            return (80, 50, 30)
        if zone_type == "yellow":
            return (40, 120, 120)
        if zone_type == "red":
            return (50, 70, 140)
        return (90, 90, 90)

    def spawn_at_zone_gate(
        self,
        zone_id: str,
        zone_entities: dict[str, dict[str, object]],
        preferred_neighbor: str | None = None,
    ) -> tuple[float, float]:
        """Spawn player near a zone gate."""
        entities = zone_entities.get(zone_id, {})
        gates = entities.get("gates", [])
        if not gates:
            return (0.5, 0.5)

        selected_gate: tuple[float, float] | None = None

        if preferred_neighbor is not None:
            neighbors = self.zone_graph.get(zone_id, [])
            if preferred_neighbor in neighbors:
                idx = neighbors.index(preferred_neighbor)
                gate_idx = idx % len(gates)
                gx, gy = gates[gate_idx]
                selected_gate = (float(gx), float(gy))

        if selected_gate is None:
            gx, gy = gates[0]
            selected_gate = (float(gx), float(gy))

        gx, gy = selected_gate
        center_x, center_y = 0.5, 0.5
        vec_x = center_x - gx
        vec_y = center_y - gy
        length = math.hypot(vec_x, vec_y)
        if length < 1e-6:
            return gx, gy

        inward_offset = 0.040
        tangent_jitter = self.rng.uniform(-0.012, 0.012)
        dir_x = vec_x / length
        dir_y = vec_y / length
        tan_x = -dir_y
        tan_y = dir_x

        sx = gx + dir_x * inward_offset + tan_x * tangent_jitter
        sy = gy + dir_y * inward_offset + tan_y * tangent_jitter
        sx, sy = self.project_to_diamond(sx, sy)
        return sx, sy
