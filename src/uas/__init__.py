"""Universe Agent System research prototype."""

from uas.adapters import MultiverseAdapter
from uas.bridge import Episode, MemoryBridge
from uas.envs import ContinuousWorldEnv, make_env

__all__ = ["ContinuousWorldEnv", "Episode", "MemoryBridge", "MultiverseAdapter", "make_env"]
__version__ = "0.2.0"

