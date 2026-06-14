"""Model-based agents and the common agent interface."""

from .base import Agent
from .factory import make_agent

__all__ = ["Agent", "make_agent"]
