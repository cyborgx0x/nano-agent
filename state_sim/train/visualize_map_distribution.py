from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from state_sim.environment import AlbionStateSim


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a zone resource distribution map with a random viewport"
    )
    parser.add_argument("--zone-id", type=str, default="forest_cross")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-size", type=int, default=1200)
    parser.add_argument("--viewport-size", type=int, default=360)
    parser.add_argument(
        "--output",
        type=str,
        default="runs/state_sim/map_distribution_viewport.png",
    )
    parser.add_argument(
        "--full-output",
        type=str,
        default="runs/state_sim/map_distribution_full.png",
    )
    parser.add_argument(
        "--viewport-output",
        type=str,
        default="runs/state_sim/map_distribution_viewport_only.png",
    )
    return parser.parse_args()


def _resource_color(resource_type: str) -> tuple[int, int, int]:
    return {
        "fiber": (80, 220, 80),
        "wood": (30, 140, 70),
        "stone": (130, 130, 130),
        "ore": (170, 120, 80),
        "hide": (60, 120, 180),
        "rough_stone": (160, 160, 160),
        "rough_logs": (50, 110, 60),
    }.get(resource_type, (80, 200, 80))


def _zone_color(zone_type: str) -> tuple[int, int, int]:
    if zone_type == "city":
        return (60, 80, 130)
    if zone_type == "blue":
        return (80, 50, 30)
    if zone_type == "yellow":
        return (40, 120, 120)
    if zone_type == "red":
        return (50, 70, 140)
    return (90, 90, 90)


def _to_px(pos: tuple[float, float], size: int) -> tuple[int, int]:
    x = int(float(pos[0]) * (size - 1))
    y = int(float(pos[1]) * (size - 1))
    return x, y


def _put_label(
    canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    import cv2

    h, w = canvas.shape[:2]
    x = max(6, min(w - 160, x))
    y = max(14, min(h - 6, y))
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        color,
        1,
        cv2.LINE_AA,
    )


def _safe_crop_with_pad(image: np.ndarray, x1: int, y1: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x2 = x1 + size
    y2 = y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        x1 += pad_left
        y1 += pad_top

    return image[y1 : y1 + size, x1 : x1 + size].copy()


def main() -> None:
    args = _parse_args()

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for this script") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = AlbionStateSim(seed=args.seed)
    env.set_task_type("gather")
    env.set_navigation_task("forest_cross", "yellow_forest_2", enabled=False)
    env.reset()

    if args.zone_id not in env.zone_entities:
        available = sorted(env.zone_entities.keys())
        env.close()
        raise ValueError(f"unknown zone-id={args.zone_id}, available={available}")

    zone_id = args.zone_id
    zone_type = env.zone_types[zone_id]
    biome = env.zone_biomes[zone_id]

    entities = env.zone_entities[zone_id]
    gates = entities.get("gates", [])
    resources = entities.get("resources", [])
    mobs = entities.get("mobs", [])
    obstacles = entities.get("obstacles", [])

    map_size = int(args.map_size)
    viewport_size = int(args.viewport_size)

    full = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    full[:] = _zone_color(zone_type)

    yy, xx = np.ogrid[:map_size, :map_size]
    nx = xx / max(1, map_size - 1)
    ny = yy / max(1, map_size - 1)
    mask = (np.abs(nx - 0.5) + np.abs(ny - 0.5)) <= env.map_l1_radius
    full[~mask] = (0, 0, 0)

    for obs in obstacles:
        ox, oy = obs["pos"]
        radius = int(float(obs["radius"]) * map_size)
        px, py = _to_px((float(ox), float(oy)), map_size)
        cv2.circle(full, (px, py), max(7, radius), (95, 95, 95), -1)
        _put_label(
            full, f"OBS:{str(obs.get('kind', 'rock'))}", px + 8, py - 8, (180, 180, 180)
        )

    for mob in mobs:
        mx, my = mob
        px, py = _to_px((float(mx), float(my)), map_size)
        cv2.circle(full, (px, py), 10, (50, 50, 220), -1)
        _put_label(full, "MOB", px + 8, py - 8, (50, 50, 220))

    for gate in gates:
        gx, gy = gate
        px, py = _to_px((float(gx), float(gy)), map_size)
        cv2.circle(full, (px, py), 11, (220, 180, 70), -1)
        _put_label(full, "GATE", px + 8, py - 8, (220, 180, 70))

    for res in resources:
        rx, ry = res["pos"]
        px, py = _to_px((float(rx), float(ry)), map_size)
        r_color = _resource_color(str(res["resource_type"]))
        r_radius = 11 if bool(res.get("plentiful", False)) else 8
        cv2.circle(full, (px, py), r_radius, r_color, -1)
        _put_label(
            full,
            f"{str(res['resource_type'])}:T{int(res['tier'])}",
            px + 8,
            py - 8,
            r_color,
        )

    player_pos = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
    ppx, ppy = _to_px(player_pos, map_size)
    cv2.circle(full, (ppx, ppy), 12, (245, 245, 245), -1)
    _put_label(full, "PLAYER", ppx + 8, ppy - 8, (255, 255, 255))

    half = viewport_size // 2
    x1 = ppx - half
    y1 = ppy - half
    x2 = x1 + viewport_size
    y2 = y1 + viewport_size

    cv2.rectangle(
        full,
        (max(0, x1), max(0, y1)),
        (min(map_size - 1, x2), min(map_size - 1, y2)),
        (0, 255, 255),
        2,
    )
    _put_label(full, "VIEWPORT", max(0, x1) + 8, max(0, y1) + 16, (0, 255, 255))

    viewport = _safe_crop_with_pad(full, x1, y1, viewport_size)

    cv2.putText(
        full,
        f"zone={zone_id} type={zone_type} biome={biome}",
        (10, map_size - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    separator = np.zeros((max(map_size, viewport_size), 16, 3), dtype=np.uint8)
    full_pad = np.pad(
        full, ((0, max(0, viewport_size - map_size)), (0, 0), (0, 0)), mode="constant"
    )
    vp_pad = np.pad(
        viewport,
        ((0, max(0, map_size - viewport_size)), (0, 0), (0, 0)),
        mode="constant",
    )
    combined = np.concatenate([full_pad, separator, vp_pad], axis=1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), combined)
    cv2.imwrite(str(Path(args.full_output)), full)
    cv2.imwrite(str(Path(args.viewport_output)), viewport)

    env.close()
    print(f"saved combined={out}")
    print(f"saved full={args.full_output}")
    print(f"saved viewport={args.viewport_output}")


if __name__ == "__main__":
    main()
