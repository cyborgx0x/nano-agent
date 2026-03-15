"""Action definitions for the Albion simulation."""

from enum import IntEnum


class Action(IntEnum):
    INTERACT = 0
    ATTACK = 1
    MOUNT_TOGGLE = 2
    IDLE = 3
