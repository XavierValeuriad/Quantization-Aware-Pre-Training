# ===================================================================
# src/tasks/__init__.py
#
# v1.0 : Entry point for SYSTEM v4.3 architecture (sub-modules).
#        Exposes only the base, the factory, and allows access to sub-modules.
# ===================================================================

# --- Base Interface ---
from .base import BaseTaskSpecification

# --- Factory ---
# Exposing the factory remains essential
from .factory import get_task_specification, TASK_REGISTRY 

# --- Allows imports like 'from src.tasks import glue' ---
# These lines are not strictly necessary if folders contain __init__.py,
# but they make the intent more explicit.
from . import drbenchmark
from . import francophone

# --- Export Control (Public Interface of 'tasks' module) ---
__all__ = [
    "BaseTaskSpecification",
    "get_task_specification",
    "TASK_REGISTRY",
    # Sub-modules are accessible but not their contents directly
    "drbenchmark",
    "francophone",
]