# Game Module

This module contains the refactored components of the Albion simulation environment, organized into focused, maintainable files.

## Structure

```
game/
├── __init__.py       # Module exports
├── actions.py        # Action enum definitions
├── state.py          # SimState dataclass
├── zones.py          # Zone management (graph, types, biomes, spawning)
├── entities.py       # Entity spawning (resources, mobs, obstacles)
└── renderer.py       # Rendering logic
```

## Components

### actions.py
- `Action`: IntEnum defining player actions (INTERACT, ATTACK, MOUNT_TOGGLE, IDLE)

### state.py
- `SimState`: Dataclass containing all simulation state variables (position, HP, inventory, etc.)

### zones.py
- `ZoneManager`: Manages zone graph, zone types, biomes, layouts, and zone-related operations
  - Zone graph and connections
  - Zone types (city, blue, yellow, red, black)
  - Biome definitions and resource profiles
  - Diamond-shaped map boundary logic
  - Gate positioning and spawning

### entities.py
- `EntityManager`: Handles spawning and management of game entities
  - Resource node spawning with tier/enchant system
  - Obstacle spawning with clearance checking
  - Resource value calculations
  - Mob spawning logic

### renderer.py
- `Renderer`: Handles all rendering operations
  - Drawing circles and labels
  - Agent view cropping
  - Visibility masking
  - Matplotlib fallback rendering

## Usage

```python
from state_sim.environment import AlbionStateSim

# Create environment (uses modular components internally)
env = AlbionStateSim(seed=42)
obs = env.reset()

# The environment now uses:
# - ZoneManager for zone operations
# - EntityManager for entity spawning
# - Renderer for visualization
```

## Benefits of Refactoring

1. **Separation of Concerns**: Each module has a clear, single responsibility
2. **Easier Testing**: Individual components can be tested in isolation
3. **Better Maintainability**: Changes to zones, entities, or rendering are localized
4. **Code Reusability**: Managers can be reused in other contexts
5. **Reduced Complexity**: Main environment file reduced from ~1900 to ~1270 lines
