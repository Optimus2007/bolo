"""The "do it" step: command -> plan of OS steps (Windows, Linux, macOS) -> carried out only when asked."""
from .plans import plan
from .run import describe, find, perform, status
from .scan import current_os, scan_system

__all__ = ["plan", "perform", "describe", "status", "find", "scan_system", "current_os"]
