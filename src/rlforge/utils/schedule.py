"""Value schedules for learning rates, epsilon, etc."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Schedule(ABC):
    """Base class for timestep-dependent scalar schedules."""

    @abstractmethod
    def value(self, progress: float) -> float:
        """Return the scheduled value.

        Args:
            progress: Fraction of training remaining in ``[0, 1]``.
                ``1.0`` means training just started; ``0.0`` means finished.
        """

    def __call__(self, progress: float) -> float:
        return self.value(progress)


class ConstantSchedule(Schedule):
    """Always returns the same value."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self, progress: float) -> float:
        return self._value


class LinearSchedule(Schedule):
    """Linearly interpolate from ``start`` to ``end`` as progress goes 1 → 0."""

    def __init__(self, start: float, end: float, end_fraction: float = 1.0) -> None:
        if not 0.0 < end_fraction <= 1.0:
            raise ValueError("end_fraction must be in (0, 1].")
        self.start = float(start)
        self.end = float(end)
        self.end_fraction = float(end_fraction)

    def value(self, progress: float) -> float:
        # progress: 1 at start, 0 at end of training
        fraction_done = 1.0 - progress
        if fraction_done >= self.end_fraction:
            return self.end
        if self.end_fraction <= 0.0:
            return self.end
        t = fraction_done / self.end_fraction
        return self.start + t * (self.end - self.start)
