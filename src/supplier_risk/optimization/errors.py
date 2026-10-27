"""Project-level errors for the Module 7 optimization and evaluation layer.

Module 7 sits on top of the Module 6 decision system. Its failures are kept
distinguishable from the Module 6 decision errors so a caller can tell an
optimizer problem from a verdict problem:

- :class:`ExampleConstructionError`  a dataset record could not be turned into a
  usable, correctly labeled DSPy example.
- :class:`OptimizationConfigError`   the optimizer settings or the optimizer
  inputs are unusable.
- :class:`CompilationError`          the DSPy optimizer itself failed.
- :class:`ArtifactError`             a compiled student could not be saved,
  loaded, or validated.
- :class:`ComparisonError`           two evaluation reports cannot be compared
  because they do not cover the same partition and the same suppliers.

Compilation and artifact errors carry only sanitized text: provider libraries
can embed request details, including credentials, in their messages.
"""

from __future__ import annotations


class OptimizationError(Exception):
    """Base class for every Module 7 optimization/evaluation failure."""


class ExampleConstructionError(OptimizationError):
    """Raised when a dataset record cannot become a usable labeled example."""


class OptimizationConfigError(OptimizationError):
    """Raised when optimizer settings or optimizer inputs are unusable."""


class CompilationError(OptimizationError):
    """Raised when the DSPy optimizer fails to compile the student."""


class ArtifactError(OptimizationError):
    """Raised when a compiled student artifact cannot be written or reloaded."""


class ComparisonError(OptimizationError):
    """Raised when two evaluation reports do not cover the same examples."""
