"""Minimal hk-BONSAI adapter for WCTE delayed-candidate vertex fits."""

from __future__ import annotations

import array
import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .geometry import PmtGeometry


def flush_process_output() -> None:
    """
    Flush Python, C stdio, and C++ iostream buffers.

    Batch stdout is normally block-buffered. Without this synchronization,
    BONSAI's printf output can remain buffered while later Python progress
    messages are written, splitting or reordering initialization lines.
    """
    sys.stdout.flush()
    sys.stderr.flush()

    # fflush(NULL) flushes every open C stdio output stream in this process.
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass

    # ROOT and WCSim also use C++ streams for some initialization messages.
    try:
        import cppyy

        cppyy.gbl.std.cout.flush()
        cppyy.gbl.std.cerr.flush()
    except (AttributeError, ImportError):
        pass


@dataclass
class BonsaiFitResult:
    """Result of one BONSAI fit, expressed in WCTE coordinates."""

    success: bool
    vertex_cm: np.ndarray
    n_fit: int = 0
    n_selected: int = 0
    n_window: int = 0
    fit_goodness: float = np.nan
    time_goodness: float = np.nan
    failure_reason: str = ""

    @classmethod
    def failed(cls, reason: str) -> "BonsaiFitResult":
        return cls(False, np.full(3, np.nan, dtype=float), failure_reason=reason)


class BonsaiVertexFitter:
    """
    Own the WCSim geometry carrier and the process-global hk-BONSAI state.

    BONSAI expects one-based WCSim tube IDs, calibrated hit times, and charges.
    The WCTE build swaps y and z internally because its cylinder axis is y; the
    output is swapped back before it is returned.
    """

    _active_instance = None

    def __init__(
        self,
        bonsai_dir: Path,
        wcsim_dir: Path,
        fit_param: Path,
        geometry_root: Path,
    ) -> None:
        if BonsaiVertexFitter._active_instance is not None:
            raise RuntimeError("Reuse the existing BonsaiVertexFitter; hk-BONSAI uses global C++ state")

        self.bonsai_dir = Path(bonsai_dir).resolve()
        self.wcsim_dir = Path(wcsim_dir).resolve()
        self.fit_param = Path(fit_param).resolve()
        self.geometry_root = Path(geometry_root).resolve()
        self._check_files()

        os.environ["BONSAIDIR"] = str(self.bonsai_dir)
        os.environ["BONSAIPARAM"] = str(self.fit_param)
        os.environ["WCSIM_BUILD_DIR"] = str(self.wcsim_dir)

        try:
            import ROOT
            import cppyy
        except ImportError as exc:
            raise RuntimeError(
                "BONSAI needs ROOT/cppyy. Activate the WCTE environment before extraction."
            ) from exc

        cppyy.add_include_path(str(self.wcsim_dir / "include" / "WCSim"))
        cppyy.load_library(str(self.wcsim_dir / "lib" / "libWCSimRoot.so"))
        cppyy.add_include_path(str(self.bonsai_dir / "bonsai"))
        cppyy.include("WCSimBonsai.hh")
        cppyy.load_library(str(self.bonsai_dir / "libWCSimBonsai.so"))

        self._geometry_file = ROOT.TFile.Open(str(self.geometry_root))
        if not self._geometry_file or self._geometry_file.IsZombie():
            raise OSError(f"Could not open BONSAI geometry ROOT file: {self.geometry_root}")
        geometry_tree = self._geometry_file.Get("wcsimGeoT")
        if not geometry_tree or geometry_tree.GetEntries() < 1:
            raise KeyError(f"{self.geometry_root} does not contain a populated wcsimGeoT")
        geometry_tree.GetEntry(0)
        self._geometry = geometry_tree.wcsimrootgeom
        self.n_pmts = int(self._geometry.GetWCNumPMT())

        self._bonsai = cppyy.gbl.WCSimBonsai()
        flush_process_output()
        init_status = int(self._bonsai.Init(self._geometry))
        flush_process_output()
        if init_status != 0:
            raise RuntimeError(f"WCSimBonsai.Init failed with status {init_status}")

        BonsaiVertexFitter._active_instance = self

    def _check_files(self) -> None:
        required = [
            self.fit_param,
            self.geometry_root,
            self.bonsai_dir / "data" / "like.bin",
            self.bonsai_dir / "libWCSimBonsai.so",
            self.wcsim_dir / "lib" / "libWCSimRoot.so",
            self.wcsim_dir / "include" / "WCSim" / "WCSimRootGeom.hh",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing BONSAI/WCSim files:\n  " + "\n  ".join(missing))

    def validate_geometry(self, geometry: PmtGeometry, tolerance_cm: float = 1e-3) -> Dict[str, float]:
        """Verify that the text readout map and BONSAI's ROOT geometry agree."""
        if len(geometry.tube_id) != self.n_pmts:
            raise ValueError(
                f"Geometry PMT count mismatch: geofile={len(geometry.tube_id)}, "
                f"BONSAI ROOT={self.n_pmts}"
            )

        root_positions = {}
        for i in range(self.n_pmts):
            pmt = self._geometry.GetPMTPtr(i)
            root_positions[int(pmt.GetTubeNo())] = np.array(
                [pmt.GetPosition(0), pmt.GetPosition(1), pmt.GetPosition(2)],
                dtype=float,
            )

        differences = []
        for tube_id, xyz_cm in zip(geometry.tube_id, geometry.xyz_cm):
            root_xyz = root_positions.get(int(tube_id))
            if root_xyz is None:
                raise ValueError(f"Tube ID {tube_id} is absent from BONSAI ROOT geometry")
            differences.append(float(np.linalg.norm(root_xyz - xyz_cm)))

        max_difference = max(differences, default=np.inf)
        if max_difference > tolerance_cm:
            raise ValueError(
                f"BONSAI ROOT geometry differs from {geometry.path}: "
                f"maximum PMT displacement is {max_difference:.6g} cm"
            )
        return {
            "n_pmts": int(self.n_pmts),
            "max_position_difference_cm": float(max_difference),
        }

    def fit(
        self,
        cable_ids: np.ndarray,
        times_ns: np.ndarray,
        charges_pe: np.ndarray,
    ) -> BonsaiFitResult:
        """Fit one candidate hit neighborhood."""
        cable_ids = np.asarray(cable_ids, dtype=np.int32)
        times_ns = np.asarray(times_ns, dtype=float)
        charges_pe = np.asarray(charges_pe, dtype=float)
        if not (len(cable_ids) == len(times_ns) == len(charges_pe)):
            raise ValueError("BONSAI cable, time, and charge arrays must have equal length")

        valid = (
            (cable_ids >= 1)
            & (cable_ids <= self.n_pmts)
            & np.isfinite(times_ns)
            & np.isfinite(charges_pe)
        )
        cable_ids = cable_ids[valid]
        times_ns = times_ns[valid]
        charges_pe = charges_pe[valid]
        if len(cable_ids) < 4:
            return BonsaiFitResult.failed("fewer than four valid hits")

        # Match the established WCTE BONSAI wrapper and keep the values close to
        # the likelihood table's local time origin.
        times_for_fit = (times_ns - np.min(times_ns) + 200.0).astype(np.float32)
        charges_for_fit = charges_pe.astype(np.float32)

        vertex = array.array("f", [0.0] * 4)
        direction = array.array("f", [0.0] * 6)
        goodness = array.array("f", [0.0] * 3)
        n_selected = array.array("i", [0, 0])
        n_hit = array.array("i", [len(cable_ids)])

        try:
            n_fit = int(
                self._bonsai.BonsaiFit(
                    vertex,
                    direction,
                    goodness,
                    n_selected,
                    n_hit,
                    array.array("i", cable_ids.tolist()),
                    array.array("f", times_for_fit.tolist()),
                    array.array("f", charges_for_fit.tolist()),
                )
            )
        except Exception as exc:
            return BonsaiFitResult.failed(f"BONSAI exception: {exc}")

        if n_fit <= 0:
            result = BonsaiFitResult.failed("fit did not converge")
            result.n_selected = int(n_selected[0])
            result.n_window = int(n_selected[1])
            return result

        # Init(..., notNuPrism=False) stores WCTE as (x, z, y) internally.
        internal = np.asarray(vertex[:3], dtype=float)
        vertex_wcte = internal[[0, 2, 1]]
        if not np.all(np.isfinite(vertex_wcte)):
            return BonsaiFitResult.failed("fit returned a non-finite vertex")

        return BonsaiFitResult(
            success=True,
            vertex_cm=vertex_wcte,
            n_fit=n_fit,
            n_selected=int(n_selected[0]),
            n_window=int(n_selected[1]),
            fit_goodness=float(goodness[2]),
            time_goodness=float(goodness[1]),
        )

    def summary(self) -> Dict[str, object]:
        return {
            "bonsai_dir": str(self.bonsai_dir),
            "wcsim_dir": str(self.wcsim_dir),
            "fit_param": str(self.fit_param),
            "geometry_root": str(self.geometry_root),
            "n_pmts": int(self.n_pmts),
        }
