"""Geometry helpers for WCTE PMT lookup and wall-distance observables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import DEFAULT_PMT_POSITION_BASE, DEFAULT_WALL_AXIS


@dataclass
class PmtGeometry:
    """
    PMT positions and orientations keyed directly by readout slot/position.

    The geofile stores slot and PMT-position columns next to the PMT coordinates.
    The nTagging pipeline only needs those pairs: readout hits provide
    hit_mpmt_slot_ids and hit_pmt_position_ids, and this class uses them to look
    up PMT x/y/z and direction directly.
    """

    path: str
    scale_to_cm: float
    tube_id: np.ndarray
    slot: np.ndarray
    pos0: np.ndarray
    xyz_cm: np.ndarray
    direction: np.ndarray
    slot_pos_to_index: np.ndarray

    @classmethod
    def from_geofile(cls, path: Path, scale_to_cm: float = 1.0) -> "PmtGeometry":
        data = np.loadtxt(path, skiprows=5)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 9:
            raise ValueError(f"Geometry file {path} does not contain PMT position/orientation columns")

        tube_id = data[:, 0].astype(np.int64)
        slot = data[:, 1].astype(np.int64)

        # The geofile PMT position is one-based (1..19), while readout
        # hit_pmt_position_ids are zero-based (0..18). Store geofile positions in
        # readout convention so later lookups can compare slot/pos directly.
        pos0 = data[:, 2].astype(np.int64) - 1
        xyz_cm = data[:, 3:6].astype(float) * scale_to_cm
        direction = data[:, 6:9].astype(float)

        if np.any(slot < 0) or np.any(pos0 < 0):
            raise ValueError(f"Geometry file {path} contains negative slot or PMT-position ids")

        slot_pos_lookup = np.full((int(np.max(slot)) + 1, int(np.max(pos0)) + 1), -1, dtype=np.int64)
        for i, (s, p) in enumerate(zip(slot, pos0)):
            if slot_pos_lookup[s, p] >= 0:
                raise ValueError(f"Duplicate geometry entry for slot={s}, position={p}")
            slot_pos_lookup[s, p] = i

        return cls(
            path=str(path),
            scale_to_cm=float(scale_to_cm),
            tube_id=tube_id,
            slot=slot,
            pos0=pos0,
            xyz_cm=xyz_cm,
            direction=direction,
            slot_pos_to_index=slot_pos_lookup,
        )

    def _position_ids_for_lookup(self, pos_ids: np.ndarray, position_base: str) -> np.ndarray:
        pos_ids = np.asarray(pos_ids, dtype=np.int64)
        if position_base == "one":
            return pos_ids - 1
        if position_base == "auto":
            has_zero = np.any(pos_ids == 0)
            has_geofile_max = np.any(pos_ids == self.slot_pos_to_index.shape[1])
            return pos_ids - 1 if has_geofile_max and not has_zero else pos_ids
        if position_base == "zero":
            return pos_ids
        raise ValueError(f"Unknown PMT position base: {position_base}")

    def _lookup_indices(self, slot_ids: np.ndarray, pos_ids: np.ndarray) -> np.ndarray:
        slot_ids = np.asarray(slot_ids, dtype=np.int64)
        pos_ids = np.asarray(pos_ids, dtype=np.int64)
        idx = np.full(len(slot_ids), -1, dtype=np.int64)
        valid = (
            (slot_ids >= 0)
            & (slot_ids < self.slot_pos_to_index.shape[0])
            & (pos_ids >= 0)
            & (pos_ids < self.slot_pos_to_index.shape[1])
        )
        idx[valid] = self.slot_pos_to_index[slot_ids[valid], pos_ids[valid]]
        return idx

    def lookup_slotpos(
        self,
        slot_ids: np.ndarray,
        pos_ids: np.ndarray,
        position_base: str = DEFAULT_PMT_POSITION_BASE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return PMT positions, PMT directions, and a boolean success mask."""
        slot_ids = np.asarray(slot_ids, dtype=np.int64)
        pos_lookup = self._position_ids_for_lookup(pos_ids, position_base)
        idx = self._lookup_indices(slot_ids, pos_lookup)

        found = idx >= 0
        xyz = np.full((len(slot_ids), 3), np.nan, dtype=float)
        direction = np.full((len(slot_ids), 3), np.nan, dtype=float)
        xyz[found] = self.xyz_cm[idx[found]]
        direction[found] = self.direction[idx[found]]
        return xyz, direction, found

    def lookup_cable_ids(
        self,
        slot_ids: np.ndarray,
        pos_ids: np.ndarray,
        position_base: str = DEFAULT_PMT_POSITION_BASE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return one-based WCSim tube IDs for BONSAI and a success mask."""
        slot_ids = np.asarray(slot_ids, dtype=np.int64)
        pos_lookup = self._position_ids_for_lookup(pos_ids, position_base)
        idx = self._lookup_indices(slot_ids, pos_lookup)
        found = idx >= 0
        cable_ids = np.full(len(slot_ids), -1, dtype=np.int32)
        cable_ids[found] = self.tube_id[idx[found]].astype(np.int32)
        return cable_ids, found

    def summary(self) -> Dict[str, object]:
        ax = {"x": 0, "y": 1, "z": 2}
        rho_xz = np.sqrt(self.xyz_cm[:, ax["x"]] ** 2 + self.xyz_cm[:, ax["z"]] ** 2)
        return {
            "path": self.path,
            "scale_to_cm": self.scale_to_cm,
            "n_pmts": int(len(self.xyz_cm)),
            "x_min_cm": float(np.nanmin(self.xyz_cm[:, 0])),
            "x_max_cm": float(np.nanmax(self.xyz_cm[:, 0])),
            "y_min_cm": float(np.nanmin(self.xyz_cm[:, 1])),
            "y_max_cm": float(np.nanmax(self.xyz_cm[:, 1])),
            "z_min_cm": float(np.nanmin(self.xyz_cm[:, 2])),
            "z_max_cm": float(np.nanmax(self.xyz_cm[:, 2])),
            "rho_xz_max_cm": float(np.nanmax(rho_xz)),
        }


@dataclass
class WallEstimator:
    """
    Simple cylindrical wall proxy for WCTE/HK-like geometries.

    axis is the cylinder axis. For WCTE with y vertical, use axis='y'. The
    cylinder dimensions are estimated from hit PMT positions. For a more
    precise analysis, replace this with a true detector geometry distance-to-wall.
    """

    axis: str = "y"
    radius_cm: float = np.nan
    axis_min_cm: float = np.nan
    axis_max_cm: float = np.nan

    def contains(self, xyz_cm: np.ndarray, margin_cm: float = 0.0) -> bool:
        return self.distance_to_wall(xyz_cm) >= margin_cm

    def distance_to_wall(self, xyz_cm: np.ndarray) -> float:
        """Signed distance: positive inside the cylinder, negative outside."""
        p = np.asarray(xyz_cm, dtype=float)
        if not np.all(np.isfinite(p)):
            return np.nan
        ax = {"x": 0, "y": 1, "z": 2}[self.axis]
        other = [i for i in range(3) if i != ax]
        rho = float(np.sqrt(p[other[0]] ** 2 + p[other[1]] ** 2))
        d_side = self.radius_cm - rho
        d_low = p[ax] - self.axis_min_cm
        d_high = self.axis_max_cm - p[ax]
        return float(min(d_side, d_low, d_high))

    def absolute_distance_to_wall(self, xyz_cm: np.ndarray) -> float:
        """Shortest non-negative distance to the finite cylindrical boundary."""
        p = np.asarray(xyz_cm, dtype=float)
        if not np.all(np.isfinite(p)):
            return np.nan

        ax = {"x": 0, "y": 1, "z": 2}[self.axis]
        other = [i for i in range(3) if i != ax]
        rho = float(np.hypot(p[other[0]], p[other[1]]))

        radial_outside = max(rho - self.radius_cm, 0.0)
        axial_outside = max(self.axis_min_cm - p[ax], p[ax] - self.axis_max_cm, 0.0)
        if radial_outside > 0.0 or axial_outside > 0.0:
            return float(np.hypot(radial_outside, axial_outside))
        return self.distance_to_wall(p)

    @classmethod
    def from_points(
        cls,
        points_cm: np.ndarray,
        axis: str = "y",
        percentile: float = 99.5,
    ) -> "WallEstimator":
        points = np.asarray(points_cm, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            raise ValueError("Cannot build WallEstimator from zero finite points")
        ax = {"x": 0, "y": 1, "z": 2}[axis]
        other = [i for i in range(3) if i != ax]
        rho = np.sqrt(points[:, other[0]] ** 2 + points[:, other[1]] ** 2)
        radius = float(np.nanpercentile(rho, percentile))
        amin = float(np.nanpercentile(points[:, ax], 100.0 - percentile))
        amax = float(np.nanpercentile(points[:, ax], percentile))
        return cls(axis=axis, radius_cm=radius, axis_min_cm=amin, axis_max_cm=amax)

    @classmethod
    def from_geometry(cls, geometry: PmtGeometry, axis: str = DEFAULT_WALL_AXIS) -> "WallEstimator":
        return cls.from_points(geometry.xyz_cm, axis=axis, percentile=100.0)
