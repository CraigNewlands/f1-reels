from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GpsFix:
    """One raw GPS position update from pos_data (Source=='pos' rows only)."""
    x: float
    y: float
    z: float
    distance_m: float
    norm_dist: float


@dataclass
class CarPacket:
    """One car telemetry sample. Speed is from the wheel sensor (reliable).
    session_time_s is lap-relative so drivers can be compared on the same axis."""
    speed_kph: float
    session_time_s: float


@dataclass
class OdometryPoint:
    """Wheel-speed-integrated cumulative distance at one point in time."""
    distance_m: float
    session_time_s: float


@dataclass
class NormalisedPoint:
    """Fraction of lap completed (0→1) and when."""
    norm_dist: float
    session_time_s: float


@dataclass
class TrackShape:
    """Circuit outline built from the median of all Q3 GPS fixes,
    re-parameterised by arc length so points are uniformly spaced visually."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def lookup(self, norm_dist: float) -> tuple[float, float, float]:
        t = np.linspace(0, 1, len(self.x))
        return (
            float(np.interp(norm_dist, t, self.x)),
            float(np.interp(norm_dist, t, self.y)),
            float(np.interp(norm_dist, t, self.z)),
        )

    def all_points(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


@dataclass
class CarPositionAtTime:
    """Plot-ready position: where the car is on the track, how far around the lap,
    and when — ready for a renderer to consume."""
    x: float
    y: float
    z: float
    norm_dist: float
    session_time_s: float


@dataclass
class DriverFrames:
    """Everything a renderer needs for one driver: 30 Hz positions (with norm_dist)
    and native-rate packets (for speed readout)."""
    abbr: str
    color: str          # hex e.g. "#FF8000"
    positions: list[CarPositionAtTime]
    packets: list[CarPacket]

    # Fast numpy arrays built once for interpolation — not part of the public model
    _t: np.ndarray = field(init=False, repr=False)
    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)
    _nd: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._t  = np.array([p.session_time_s for p in self.positions])
        self._x  = np.array([p.x              for p in self.positions])
        self._y  = np.array([p.y              for p in self.positions])
        self._nd = np.array([p.norm_dist      for p in self.positions])

    @property
    def lap_time_s(self) -> float:
        return self._t[-1]

    def at(self, t: float) -> tuple[float, float, float]:
        """Return (x, y, norm_dist) at lap time t seconds."""
        return (
            float(np.interp(t, self._t, self._x)),
            float(np.interp(t, self._t, self._y)),
            float(np.interp(t, self._t, self._nd)),
        )
