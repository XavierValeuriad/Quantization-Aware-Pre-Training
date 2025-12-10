# ===================================================================
# src/tasks/drbenchmark/__init__.py
#
# v1.0 : Exposes TaskSpecification classes for DrBenchmark.
# ===================================================================

# from .morfitt import MorfittTask
from .ner import QuaeroTask
from .pos import CasTask, EssaiTask

__all__ = [
    "QuaeroTask", 
    "CasTask", 
    "EssaiTask"
]