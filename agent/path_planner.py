"""
Path Planning for Fiber Gathering
A* pathfinding + TSP optimization for multi-target gathering routes
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PathNode:
    """Node in the pathfinding graph."""
    position: Tuple[int, int]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Optional['PathNode'] = None

    @property
    def f_cost(self):
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost


class PathPlanner:
    """
    A* pathfinding for navigating to fiber resources.
    """

    def __init__(self, move_cost=1.0, diagonal_cost=1.414):
        """
        Args:
            move_cost: Cost of moving one pixel
            diagonal_cost: Cost of diagonal movement (sqrt(2))
        """
        self.move_cost = move_cost
        self.diagonal_cost = diagonal_cost

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Euclidean distance heuristic.
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return np.sqrt(dx**2 + dy**2)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 8-directional neighbors.
        """
        x, y = pos
        neighbors = [
            (x+1, y),   (x-1, y),   (x, y+1),   (x, y-1),  # Cardinal
            (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)  # Diagonal
        ]
        return neighbors

    def is_diagonal(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if movement is diagonal."""
        return abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1]) == 1

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  max_iterations: int = 10000) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding from start to goal.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            max_iterations: Maximum search iterations

        Returns:
            List of waypoints from start to goal, or None if no path found
        """
        # Early exit if start == goal
        if start == goal:
            return [start]

        # Priority queue: (f_cost, node)
        open_set = []
        start_node = PathNode(
            position=start,
            g_cost=0,
            h_cost=self.heuristic(start, goal)
        )
        heapq.heappush(open_set, start_node)

        # Track visited nodes
        closed_set = set()
        open_dict = {start: start_node}  # Fast lookup

        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Remove from open_dict
            if current.position in open_dict:
                del open_dict[current.position]

            # Goal reached?
            if current.position == goal:
                return self._reconstruct_path(current)

            closed_set.add(current.position)

            # Check neighbors
            for neighbor_pos in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue

                # Calculate cost
                move_cost = self.diagonal_cost if self.is_diagonal(current.position, neighbor_pos) else self.move_cost
                tentative_g = current.g_cost + move_cost

                # Check if this path is better
                if neighbor_pos in open_dict:
                    neighbor_node = open_dict[neighbor_pos]
                    if tentative_g < neighbor_node.g_cost:
                        # Better path found
                        neighbor_node.g_cost = tentative_g
                        neighbor_node.parent = current
                        heapq.heapify(open_set)  # Re-sort
                else:
                    # New node
                    neighbor_node = PathNode(
                        position=neighbor_pos,
                        g_cost=tentative_g,
                        h_cost=self.heuristic(neighbor_pos, goal),
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node

        # No path found
        return None

    def _reconstruct_path(self, node: PathNode) -> List[Tuple[int, int]]:
        """Reconstruct path from goal node to start."""
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

    def simplify_path(self, path: List[Tuple[int, int]],
                     tolerance: float = 5.0) -> List[Tuple[int, int]]:
        """
        Simplify path using Ramer-Douglas-Peucker algorithm.
        Reduces number of waypoints while maintaining path shape.

        Args:
            path: Full path
            tolerance: Maximum distance from simplified line

        Returns:
            Simplified path with fewer waypoints
        """
        if len(path) <= 2:
            return path

        # Find point with maximum distance from line
        start = np.array(path[0])
        end = np.array(path[-1])

        max_dist = 0
        max_index = 0

        for i in range(1, len(path) - 1):
            point = np.array(path[i])

            # Distance from point to line
            dist = self._point_to_line_distance(point, start, end)

            if dist > max_dist:
                max_dist = dist
                max_index = i

        # If max distance > tolerance, split and recurse
        if max_dist > tolerance:
            left = self.simplify_path(path[:max_index+1], tolerance)
            right = self.simplify_path(path[max_index:], tolerance)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]

    def _point_to_line_distance(self, point: np.ndarray,
                                line_start: np.ndarray,
                                line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line."""
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return np.linalg.norm(point - line_start)

        # Project point onto line
        t = np.dot(point - line_start, line_vec) / (line_len ** 2)
        t = np.clip(t, 0, 1)

        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)


class GatheringRouter:
    """
    Plan optimal routes for gathering multiple fiber resources.
    Solves Traveling Salesman Problem (TSP) for efficient collection.
    """

    def __init__(self, path_planner: PathPlanner):
        self.planner = path_planner

    def plan_gathering_route(self, current_pos: Tuple[int, int],
                           fiber_positions: List[Tuple[int, int]],
                           max_fibers: int = 10) -> List[Tuple[int, int]]:
        """
        Plan optimal route to gather multiple fibers.

        Args:
            current_pos: Current character position
            fiber_positions: List of fiber locations
            max_fibers: Maximum number of fibers to collect in one route

        Returns:
            Ordered list of fiber positions to visit
        """
        if not fiber_positions:
            return []

        # Limit to nearest fibers if too many
        if len(fiber_positions) > max_fibers:
            fiber_positions = self._get_nearest_fibers(current_pos, fiber_positions, max_fibers)

        # Use nearest neighbor heuristic for TSP
        route = [current_pos]
        remaining = set(fiber_positions)

        current = current_pos
        while remaining:
            # Find nearest unvisited fiber
            nearest = min(remaining, key=lambda pos: self.planner.heuristic(current, pos))
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest

        return route[1:]  # Exclude starting position

    def _get_nearest_fibers(self, pos: Tuple[int, int],
                           fibers: List[Tuple[int, int]],
                           count: int) -> List[Tuple[int, int]]:
        """Get N nearest fibers to position."""
        distances = [(f, self.planner.heuristic(pos, f)) for f in fibers]
        distances.sort(key=lambda x: x[1])
        return [f for f, _ in distances[:count]]

    def calculate_route_length(self, route: List[Tuple[int, int]]) -> float:
        """Calculate total route distance."""
        if len(route) < 2:
            return 0

        total = 0
        for i in range(len(route) - 1):
            total += self.planner.heuristic(route[i], route[i+1])
        return total


def visualize_path(path: List[Tuple[int, int]],
                  map_img: np.ndarray,
                  color=(255, 0, 255)) -> np.ndarray:
    """
    Draw path on map image.

    Args:
        path: List of waypoints
        map_img: Background map image
        color: Path color (BGR)

    Returns:
        Map with path drawn
    """
    import cv2

    result = map_img.copy()

    if len(path) < 2:
        return result

    # Draw path lines
    for i in range(len(path) - 1):
        pt1 = path[i]
        pt2 = path[i+1]
        cv2.line(result, pt1, pt2, color, 3)

    # Draw waypoints
    for i, point in enumerate(path):
        if i == 0:
            # Start (green)
            cv2.circle(result, point, 8, (0, 255, 0), -1)
        elif i == len(path) - 1:
            # Goal (red)
            cv2.circle(result, point, 8, (0, 0, 255), -1)
        else:
            # Waypoint (yellow)
            cv2.circle(result, point, 5, (0, 255, 255), -1)

    return result
