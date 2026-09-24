"""Pure output-time envelope and speech rate-map calculations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._validation import validate_finite, validate_integer
from .exceptions import InvalidParameterError

ControlPoints = Iterable[tuple[float, float]]


def _validate_points(
    points: ControlPoints,
    *,
    name: str,
    positive_values: bool = False,
    allow_empty: bool = True,
) -> tuple[tuple[float, float], ...]:
    """Validate ordered ``(output_seconds, value)`` control points."""
    try:
        raw_points = tuple(points)
    except TypeError as exc:
        raise InvalidParameterError(f"{name} must be an iterable of control points") from exc
    if not raw_points and not allow_empty:
        raise InvalidParameterError(f"{name} must contain at least one control point")

    validated: list[tuple[float, float]] = []
    for index, raw_point in enumerate(raw_points):
        try:
            point = tuple(raw_point)
        except TypeError as exc:
            raise InvalidParameterError(f"{name}[{index}] must contain a time and value") from exc
        if len(point) != 2:
            raise InvalidParameterError(f"{name}[{index}] must contain a time and value")
        try:
            time = validate_finite(point[0], f"{name}[{index}] time")
            value = validate_finite(point[1], f"{name}[{index}] value")
        except InvalidParameterError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidParameterError(
                f"{name}[{index}] time and value must be finite numbers"
            ) from exc
        if time < 0.0:
            raise InvalidParameterError(f"{name}[{index}] time must be non-negative")
        if positive_values and value <= 0.0:
            raise InvalidParameterError(f"{name}[{index}] value must be positive")
        if index == 0 and time != 0.0:
            raise InvalidParameterError(f"the first {name} point must be at 0.0 seconds")
        if validated and time <= validated[-1][0]:
            raise InvalidParameterError(f"{name} times must be strictly increasing")
        validated.append((time, value))
    return tuple(validated)


class _LinearEnvelope:
    """Validated piecewise-linear envelope with a held final value."""

    __slots__ = ("_default", "_times", "_values")

    def __init__(
        self,
        points: ControlPoints = (),
        *,
        default: float = 0.0,
        name: str = "control_points",
        positive_values: bool = False,
    ) -> None:
        validated = _validate_points(
            points, name=name, positive_values=positive_values, allow_empty=True
        )
        self._default = validate_finite(default, f"{name} default")
        self._times = np.asarray([point[0] for point in validated], dtype=np.float64)
        self._values = np.asarray([point[1] for point in validated], dtype=np.float64)

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        """Return the validated points as immutable Python floats."""
        return tuple(zip(self._times.tolist(), self._values.tolist(), strict=True))

    def value_at(self, output_seconds: float) -> float:
        """Evaluate the envelope at a non-negative output time."""
        time = validate_finite(output_seconds, "output_seconds")
        if time < 0.0:
            raise InvalidParameterError("output_seconds must be non-negative")
        if self._times.size == 0:
            return self._default
        index = int(np.searchsorted(self._times, time, side="right")) - 1
        if index < 0:
            return float(self._values[0])
        if index >= self._times.size - 1:
            return float(self._values[-1])
        fraction = (time - self._times[index]) / (self._times[index + 1] - self._times[index])
        return float((1.0 - fraction) * self._values[index] + fraction * self._values[index + 1])


class _RateMap:
    """Map output seconds to source seconds using a linear rate envelope."""

    __slots__ = ("_rates", "_source_knots", "_times")

    def __init__(self, rate_points: ControlPoints = ()) -> None:
        points = _validate_points(
            rate_points,
            name="rate_points",
            positive_values=True,
            allow_empty=True,
        )
        if not points:
            points = ((0.0, 1.0),)
        self._times = np.asarray([point[0] for point in points], dtype=np.float64)
        self._rates = np.asarray([point[1] for point in points], dtype=np.float64)
        self._source_knots = np.zeros(self._times.size, dtype=np.float64)
        for index in range(self._times.size - 1):
            duration = self._times[index + 1] - self._times[index]
            average_rate = 0.5 * self._rates[index] + 0.5 * self._rates[index + 1]
            source_duration = duration * average_rate
            source_time = self._source_knots[index] + source_duration
            if not np.isfinite(source_time):
                raise InvalidParameterError("rate map exceeds the finite timing range")
            self._source_knots[index + 1] = source_time

    @property
    def rate_points(self) -> tuple[tuple[float, float], ...]:
        """Return the effective validated rate points."""
        return tuple(zip(self._times.tolist(), self._rates.tolist(), strict=True))

    def rate_at(self, output_seconds: float) -> float:
        """Evaluate the rate factor at an output time."""
        time = validate_finite(output_seconds, "output_seconds")
        if time < 0.0:
            raise InvalidParameterError("output_seconds must be non-negative")
        index = int(np.searchsorted(self._times, time, side="right")) - 1
        if index < 0:
            return float(self._rates[0])
        if index >= self._times.size - 1:
            return float(self._rates[-1])
        fraction = (time - self._times[index]) / (self._times[index + 1] - self._times[index])
        return float((1.0 - fraction) * self._rates[index] + fraction * self._rates[index + 1])

    def source_time_at(self, output_seconds: float) -> float:
        """Return the source time consumed by an output time."""
        time = validate_finite(output_seconds, "output_seconds")
        if time < 0.0:
            raise InvalidParameterError("output_seconds must be non-negative")
        index = int(np.searchsorted(self._times, time, side="right")) - 1
        if index >= self._times.size - 1:
            source_time = self._source_knots[-1] + (time - self._times[-1]) * self._rates[-1]
        else:
            duration = self._times[index + 1] - self._times[index]
            elapsed = time - self._times[index]
            fraction = elapsed / duration
            weighted_rate = (1.0 - 0.5 * fraction) * self._rates[
                index
            ] + 0.5 * fraction * self._rates[index + 1]
            source_time = self._source_knots[index] + elapsed * weighted_rate
        if not np.isfinite(source_time):
            raise InvalidParameterError("rate map exceeds the finite timing range")
        return float(source_time)

    def output_time_for_source_time(self, source_seconds: float) -> float:
        """Invert the rate map for a non-negative source time."""
        source_time = validate_finite(source_seconds, "source_seconds")
        if source_time < 0.0:
            raise InvalidParameterError("source_seconds must be non-negative")
        index = int(np.searchsorted(self._source_knots, source_time, side="right")) - 1
        if index >= self._times.size - 1:
            output_time = self._times[-1] + (source_time - self._source_knots[-1]) / self._rates[-1]
        else:
            duration = self._times[index + 1] - self._times[index]
            source_delta = source_time - self._source_knots[index]
            scale = max(self._rates[index], self._rates[index + 1])
            normalized_start = self._rates[index] / scale
            normalized_end = self._rates[index + 1] / scale
            normalized_delta = (source_delta / scale) / duration
            discriminant = (
                normalized_start * normalized_start
                + 2.0 * (normalized_end - normalized_start) * normalized_delta
            )
            if discriminant < 0.0 or not np.isfinite(discriminant):
                raise InvalidParameterError("rate map inverse exceeds the finite timing range")
            denominator = normalized_start + float(np.sqrt(discriminant))
            fraction = 2.0 * normalized_delta / denominator
            output_time = self._times[index] + duration * fraction
        if not np.isfinite(output_time):
            raise InvalidParameterError("rate map inverse exceeds the finite timing range")
        return float(output_time)

    def output_frames_for_input_frames(self, input_frames: int, sample_rate: int) -> int:
        """Return the deterministic rounded output-frame count."""
        frames = validate_integer(input_frames, "input_frames", minimum=0)
        rate = validate_integer(sample_rate, "sample_rate")
        output_seconds = self.output_time_for_source_time(frames / rate)
        output_frames = output_seconds * rate
        if not np.isfinite(output_frames) or output_frames > np.iinfo(np.intp).max:
            raise InvalidParameterError("requested output length is too large")
        return round(output_frames)
