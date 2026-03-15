"""Rendering logic for the Albion simulation."""

from __future__ import annotations

import numpy as np


class Renderer:
    """Handles rendering of the simulation environment."""

    def __init__(self):
        self._mpl_figure = None
        self._mpl_axes = None
        self._mpl_image = None

    def draw_circle(
        self,
        canvas: np.ndarray,
        pos: tuple[float, float],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a circle on the canvas."""
        try:
            import cv2

            h, w = canvas.shape[:2]
            x = int(pos[0] * (w - 1))
            y = int(pos[1] * (h - 1))
            cv2.circle(canvas, (x, y), radius, color, -1)
        except Exception:
            pass

    def crop_agent_view(
        self, world: np.ndarray, view_size: int, agent_x: float, agent_y: float
    ) -> np.ndarray:
        """Crop the world view centered on the agent."""
        half = view_size // 2
        h, w = world.shape[:2]

        agent_pixel_x = int(agent_x * (w - 1))
        agent_pixel_y = int(agent_y * (h - 1))

        padded = np.pad(world, ((half, half), (half, half), (0, 0)), mode="edge")
        center_x = agent_pixel_x + half
        center_y = agent_pixel_y + half

        y1 = center_y - half
        y2 = y1 + view_size
        x1 = center_x - half
        x2 = x1 + view_size
        return padded[y1:y2, x1:x2].copy()

    def apply_visibility_mask(
        self, frame: np.ndarray, radius_ratio: float = 0.40
    ) -> np.ndarray:
        """Apply a circular visibility mask to the frame."""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        radius = int(min(h, w) * radius_ratio)

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
        mask = dist2 <= radius * radius

        out = frame.copy()
        out[~mask] = (out[~mask] * 0.12).astype(np.uint8)
        return out

    def world_to_view_point(
        self,
        pos: tuple[float, float],
        view_size: int,
        world_size: int,
        agent_x: float,
        agent_y: float,
    ) -> tuple[int, int]:
        """Convert world coordinates to view coordinates."""
        scale = world_size - 1
        dx = (float(pos[0]) - float(agent_x)) * scale
        dy = (float(pos[1]) - float(agent_y)) * scale
        vx = int(view_size / 2 + dx)
        vy = int(view_size / 2 + dy)
        return vx, vy

    def draw_label(
        self,
        canvas: np.ndarray,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw a text label on the canvas."""
        try:
            import cv2

            h, w = canvas.shape[:2]
            x = min(w - 4, max(4, pos[0]))
            y = min(h - 4, max(12, pos[1]))
            cv2.putText(
                canvas,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    def render_with_matplotlib(self, canvas: np.ndarray) -> None:
        """Render using matplotlib as fallback."""
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "Human render requires GUI OpenCV or matplotlib. Install matplotlib and retry."
            ) from exc

        rgb_canvas = canvas[:, :, ::-1]
        if self._mpl_figure is None:
            plt.ion()
            self._mpl_figure, self._mpl_axes = plt.subplots(figsize=(5, 5))
            assert self._mpl_axes is not None
            self._mpl_axes.set_title("AlbionStateSim")
            self._mpl_axes.axis("off")
            self._mpl_image = self._mpl_axes.imshow(rgb_canvas)
        else:
            assert self._mpl_image is not None
            self._mpl_image.set_data(rgb_canvas)

        plt.pause(0.001)

    def close(self) -> None:
        """Clean up rendering resources."""
        try:
            import cv2

            cv2.destroyWindow("AlbionStateSim")
        except Exception:
            pass

        try:
            import matplotlib.pyplot as plt

            if self._mpl_figure is not None:
                plt.close(self._mpl_figure)
        except Exception:
            pass
