#!/usr/bin/env python3
"""Shared helpers for closed-loop TMM / SMM / aniso / RCWA pytest + notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("SAVE_TO_FILE", "0")

try:
    from py_core_plugins import viz_io
except ImportError:
    import viz_io

viz_io.SAVE_TO_FILE = False

import simulation as sim

WL_UM = 0.55
ANGLE_N = 41
# Default gate: most closed-loop residuals ≤~2e-6; hard default 1e-6.
RT_TOL = 1e-6
E_TOL = 1e-6
# >1e-6: wrap-aware |Δφ| (rad); float/arg near zeros can exceed 1e-6
EPHASE_TOL = 1e-5
# >1e-6: diag-ε aniso vs TMM; p-channel peak ~7e-6 (modal t / q-branch)
ANISO_TOL = 1e-5
# >1e-6: n_G=1 RCWA specular vs TMM (FP), peak |Δr|~1.03e-6
RCWA_RT_TOL = 2e-6
# Default panels: R,T are total Sz flux. Optional cross-pol when r_x/t_x present.
PANEL_KEYS = ("R", "T", "r_amp", "r_phase", "t_amp", "t_phase", "Eamp", "Ephase")
PANEL_KEYS_CROSS = (
    "R",
    "T",
    "r_amp",
    "r_phase",
    "t_amp",
    "t_phase",
    "rx_amp",
    "rx_phase",
    "tx_amp",
    "tx_phase",
    "Eamp",
    "Ephase",
)
PANEL_YLABELS = {
    "R": "R (flux)",
    "T": "T (flux)",
    "r_amp": "|r| co-pol",
    "r_phase": "∠r co-pol (rad)",
    "t_amp": "|t| co-pol",
    "t_phase": "∠t co-pol (rad)",
    "rx_amp": "|r| cross",
    "rx_phase": "∠r cross (rad)",
    "tx_amp": "|t| cross",
    "tx_phase": "∠t cross (rad)",
    "Eamp": "|E|=√(|Ex|²+|Ey|²+|Ez|²)",
    "Ephase": "∠E (dom. Ex/Ey)",
}
STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
_PHASE_KEYS = frozenset(("r_phase", "t_phase", "rx_phase", "tx_phase", "Ephase"))


def require_rcwa_triad() -> None:
    """Skip when RCWA TMM-aligned triad is missing from the artifact."""
    import pytest

    if not hasattr(sim, "RCWA_get_r_t_coeff"):
        pytest.skip("RCWA_get_r_t_coeff API missing")


@dataclass
class SweepResult:
    name: str
    angles_deg: np.ndarray
    r: np.ndarray
    t: np.ndarray
    R: np.ndarray
    T: np.ndarray
    Eamp: np.ndarray
    Ephase: np.ndarray
    r_x: np.ndarray | None = None
    t_x: np.ndarray | None = None

    def has_cross(self) -> bool:
        return self.r_x is not None and self.t_x is not None


def angle_grid_rad(n: int = ANGLE_N) -> np.ndarray:
    # Start slightly off normal so u=0 Jones/modal branch is not the only sample.
    return np.linspace(1e-3, 0.5 * np.pi, n, endpoint=False)


def make_iso_layer(n: complex, depth: float, name: str = "") -> Any:
    return sim.make_layer_from_nk_s(complex(n), float(depth), name)


def make_aniso_diag_layer(n: complex, depth: float, name: str = "") -> Any:
    eps = complex(n) * complex(n)
    mat = sim.material_s.from_anisotropic_nk(
        [eps, 0j, 0j, eps, eps], name or "aniso_diag"
    )
    lyr = sim.layer_s()
    lyr.background_material = sim.share_material_s(mat)
    lyr.depth = float(depth)
    return lyr


def layers_to_d_nk(layers_s: Sequence[Any], wl: float = WL_UM) -> list[Any]:
    out = []
    for i, lyr in enumerate(layers_s):
        mat = lyr.background_material
        n = complex(mat.nk_at_wavelength_um(float(wl)))
        out.append(sim.make_layer_from_nk_d(n, float(lyr.depth), f"L{i}"))
    return out


def e_amp_phase(ex, ey, ez) -> tuple[float, float]:
    """|E|=√(|Ex|²+|Ey|²+|Ez|²); phase from dominant in-plane (Ex/Ey)."""
    comps = [complex(ex), complex(ey), complex(ez)]
    amp = float(np.sqrt(sum(abs(c) ** 2 for c in comps)))
    primary = comps[1] if abs(comps[1]) >= abs(comps[0]) else comps[0]
    if abs(primary) > 1e-14:
        return amp, float(np.angle(primary))
    if abs(comps[2]) > 1e-14:
        return amp, float(np.angle(comps[2]))
    return amp, 0.0


def assert_close_arr(a, b, tol: float, label: str) -> None:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    err = float(np.max(np.abs(a - b)))
    if err > tol:
        raise AssertionError(f"{label}: max|a-b|={err} > tol={tol}")


def assert_sweep_vs_ref(
    ref: SweepResult,
    other: SweepResult,
    *,
    rt_tol=RT_TOL,
    e_tol=E_TOL,
    ephase_tol: float = EPHASE_TOL,
    cross_tol: float | None = None,
) -> None:
    """Compare against baseline. Overrides >1e-6 need a call-site comment."""
    assert_close_arr(ref.r, other.r, rt_tol, f"{other.name}.r")
    assert_close_arr(ref.t, other.t, rt_tol, f"{other.name}.t")
    assert_close_arr(ref.R, other.R, rt_tol, f"{other.name}.R")
    assert_close_arr(ref.T, other.T, rt_tol, f"{other.name}.T")
    if cross_tol is not None and ref.has_cross() and other.has_cross():
        assert_close_arr(ref.r_x, other.r_x, cross_tol, f"{other.name}.r_x")
        assert_close_arr(ref.t_x, other.t_x, cross_tol, f"{other.name}.t_x")
    mask = np.real(np.asarray(ref.T, dtype=np.complex128)) > 0.05
    if np.any(mask):
        assert_close_arr(ref.Eamp[mask], other.Eamp[mask], e_tol, f"{other.name}.Eamp")
        dphi = np.angle(
            np.exp(
                1j
                * (
                    np.asarray(ref.Ephase, dtype=float)[mask]
                    - np.asarray(other.Ephase, dtype=float)[mask]
                )
            )
        )
        if float(np.max(np.abs(dphi))) > ephase_tol:
            raise AssertionError(
                f"{other.name}.Ephase: wrap-diff {float(np.max(np.abs(dphi)))} > tol={ephase_tol}"
            )


def _cross_series(res: SweepResult, which: str) -> np.ndarray | None:
    return res.r_x if which == "r" else res.t_x


def _panel_y(res: SweepResult, key: str) -> np.ndarray:
    if key == "R":
        return np.real(np.asarray(res.R, dtype=np.complex128))
    if key == "T":
        return np.real(np.asarray(res.T, dtype=np.complex128))
    if key == "r_amp":
        return np.abs(np.asarray(res.r, dtype=np.complex128))
    if key == "r_phase":
        return np.unwrap(np.angle(np.asarray(res.r, dtype=np.complex128)))
    if key == "t_amp":
        return np.abs(np.asarray(res.t, dtype=np.complex128))
    if key == "t_phase":
        return np.unwrap(np.angle(np.asarray(res.t, dtype=np.complex128)))
    if key in ("rx_amp", "rx_phase", "tx_amp", "tx_phase"):
        series = _cross_series(res, "r" if key.startswith("rx") else "t")
        if series is None:
            return np.zeros_like(res.angles_deg, dtype=float)
        arr = np.asarray(series, dtype=np.complex128)
        return np.abs(arr) if key.endswith("amp") else np.unwrap(np.angle(arr))
    if key == "Eamp":
        return np.asarray(res.Eamp, dtype=float)
    if key == "Ephase":
        return np.unwrap(np.asarray(res.Ephase, dtype=float))
    raise KeyError(key)


def _panel_delta(ref: SweepResult, other: SweepResult, key: str) -> np.ndarray:
    if key in _PHASE_KEYS:
        return np.angle(np.exp(1j * (_panel_y(other, key) - _panel_y(ref, key))))
    return _panel_y(other, key) - _panel_y(ref, key)


def _find_baseline(results: Sequence[SweepResult]) -> SweepResult | None:
    for res in results:
        if str(res.name).upper() == "TMM" or str(res.name).endswith("TMM"):
            return res
    for res in results:
        if "TMM" in str(res.name).upper() and not str(res.name).lower().startswith("diff"):
            return res
    return results[0] if results else None


def _format_delta_axis(ax, delta_series: Sequence[np.ndarray]) -> None:
    """Make machine-eps / 1e-6 residuals readable in dense 2×N grids.

    Default ScalarFormatter puts the decade in a tiny offset label that
    ``tight_layout`` often clips; tick text then looks like ``0.00`` / ``±1``.
    """
    from matplotlib.ticker import ScalarFormatter

    chunks = [np.asarray(y, dtype=float).ravel() for y in delta_series]
    y = np.concatenate(chunks) if chunks else np.zeros(1)
    peak = float(np.nanmax(np.abs(y))) if y.size else 0.0
    if not np.isfinite(peak):
        peak = 0.0

    ax.text(
        0.02,
        0.96,
        f"max|Δ|={peak:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.12", "fc": "w", "ec": "none", "alpha": 0.85},
    )
    if peak == 0.0:
        # Exact 0: still pin a 1e-7 frame so all Δ panels share a readable scale.
        ax.set_ylim(-1e-7, 1e-7)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(7)


def plot_solver_suite(results: Sequence[SweepResult], *, pol: str, title_prefix: str) -> None:
    """2×N subplots: row0 overlay vs solvers, row1 Δ vs baseline."""
    abs_rows = [r for r in results if not str(r.name).lower().startswith("diff")]
    if not abs_rows:
        return
    baseline = _find_baseline(abs_rows)
    others = [r for r in abs_rows if r is not baseline] if baseline is not None else []
    keys = PANEL_KEYS_CROSS if any(r.has_cross() for r in abs_rows) else PANEL_KEYS

    n_col = len(keys)
    fig, axes = plt.subplots(2, n_col, figsize=(2.2 * n_col, 5.2), squeeze=False)
    fig.suptitle(f"{title_prefix} {pol}", fontsize=12)

    for j, key in enumerate(keys):
        ax0 = axes[0, j]
        for i, res in enumerate(abs_rows):
            ax0.plot(
                np.asarray(res.angles_deg, dtype=float),
                _panel_y(res, key),
                STYLES[i % len(STYLES)],
                label=res.name,
                lw=1.2,
            )
        ax0.set_title(PANEL_YLABELS[key], fontsize=8)
        ax0.grid(True, alpha=0.35)
        if j == 0:
            ax0.set_ylabel("overlay")
        if j == n_col - 1:
            ax0.legend(loc="best", fontsize=7)

        ax1 = axes[1, j]
        if baseline is None or not others:
            ax1.set_visible(False)
            continue
        deltas = []
        for i, res in enumerate(others):
            d = _panel_delta(baseline, res, key)
            deltas.append(d)
            ax1.plot(
                np.asarray(res.angles_deg, dtype=float),
                d,
                STYLES[(i + 1) % len(STYLES)],
                label=f"Δ {res.name}",
                lw=1.2,
            )
        ax1.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax1.grid(True, alpha=0.35)
        ax1.set_xlabel("angle (deg)")
        _format_delta_axis(ax1, deltas)
        if j == 0:
            ax1.set_ylabel(f"Δ vs {baseline.name}")
        if j == n_col - 1:
            ax1.legend(loc="best", fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    viz_io.save_or_show_fig(fig, title=f"{title_prefix} {pol}")
    plt.close(fig)


def _empty_rt_arrays(n: int, *, with_cross: bool):
    r = np.zeros(n, dtype=np.complex128)
    t = np.zeros(n, dtype=np.complex128)
    r_x = np.zeros(n, dtype=np.complex128) if with_cross else None
    t_x = np.zeros(n, dtype=np.complex128) if with_cross else None
    R = np.zeros(n, dtype=float)
    T = np.zeros(n, dtype=float)
    Eamp = np.zeros(n, dtype=float)
    Ephase = np.zeros(n, dtype=float)
    return r, t, r_x, t_x, R, T, Eamp, Ephase


def sweep_tmm_smm(
    layers: Sequence[Any],
    *,
    pol: str,
    angles_rad: np.ndarray,
    wl: float = WL_UM,
    layer_index: int = 0,
    depth_in_layer: float = 0.0,
    name_prefix: str = "",
) -> tuple[SweepResult, SweepResult]:
    """TMM/SMM sweep. R,T from get_r_t_power (= total/co-pol flux; no cross-pol)."""
    get_power = sim.TMM_get_r_t_power_s if pol == "s" else sim.TMM_get_r_t_power_p
    iface = (
        sim.TMM_interface_transfer_matrix_with_thickness_s
        if pol == "s"
        else sim.TMM_interface_transfer_matrix_with_thickness_p
    )
    get_e = sim.TMM_get_e_field_at_depth_s if pol == "s" else sim.TMM_get_e_field_at_depth_p
    smm_power = sim.SMM_get_r_t_power_s if pol == "s" else sim.SMM_get_r_t_power_p
    smm_coeff = sim.SMM_get_r_t_coeff_s if pol == "s" else sim.SMM_get_r_t_coeff_p
    smm_e = sim.SMM_get_e_field_at_depth_s if pol == "s" else sim.SMM_get_e_field_at_depth_p

    layers_list = list(layers)
    ang_deg = np.degrees(angles_rad)
    n = len(angles_rad)
    tmm_r, tmm_t, _, _, tmm_R, tmm_T, tmm_Eamp, tmm_Ephase = _empty_rt_arrays(n, with_cross=False)
    smm_r, smm_t, _, _, smm_R, smm_T, smm_Eamp, smm_Ephase = _empty_rt_arrays(n, with_cross=False)

    for i, th in enumerate(angles_rad):
        # propagate_direction is polarization-independent (Snell / u = n sinθ).
        dirs = sim.TMM_propagate_direction_s(layers_list, float(th), float(wl))
        M = iface(layers_list, dirs, float(wl))
        r, t = sim.TMM_get_r_t_from_tmm(M[-1])
        R, T = get_power(layers_list, dirs, float(wl))
        ea, ep = e_amp_phase(
            *get_e(layers_list, float(wl), dirs, int(layer_index), float(depth_in_layer))
        )
        tmm_r[i], tmm_t[i] = complex(r), complex(t)
        tmm_R[i], tmm_T[i] = float(R), float(T)
        tmm_Eamp[i], tmm_Ephase[i] = ea, ep

        rc, tc = smm_coeff(layers_list, dirs, float(wl))
        Rs, Ts = smm_power(layers_list, dirs, float(wl))
        ea, ep = e_amp_phase(
            *smm_e(layers_list, float(wl), dirs, int(layer_index), float(depth_in_layer))
        )
        smm_r[i], smm_t[i] = complex(rc), complex(tc)
        smm_R[i], smm_T[i] = float(Rs), float(Ts)
        smm_Eamp[i], smm_Ephase[i] = ea, ep

    pref = name_prefix or ""
    return (
        SweepResult(f"{pref}TMM", ang_deg, tmm_r, tmm_t, tmm_R, tmm_T, tmm_Eamp, tmm_Ephase),
        SweepResult(f"{pref}SMM", ang_deg, smm_r, smm_t, smm_R, smm_T, smm_Eamp, smm_Ephase),
    )


def sweep_smm_aniso(
    layers: Sequence[Any],
    *,
    pol: str,
    angles_rad: np.ndarray,
    wl: float = WL_UM,
    layer_index: int = 0,
    depth_in_layer: float = 0.0,
    name: str = "SMM_aniso",
    with_cross: bool = False,
) -> SweepResult:
    """Sweep SMM_aniso. angles_rad = incidence in top-layer material (TMM convention)."""
    layers_list = list(layers)
    ang_deg = np.degrees(angles_rad)
    n = len(angles_rad)
    r, t, r_x, t_x, R, T, Eamp, Ephase = _empty_rt_arrays(n, with_cross=with_cross)

    for i, th in enumerate(angles_rad):
        th_c = complex(float(th), 0.0)
        rc, tc, rcx, tcx = sim.SMM_aniso_get_r_t_coeff(layers_list, th_c, float(wl), pol)
        r[i], t[i] = complex(rc), complex(tc)
        if with_cross:
            r_x[i], t_x[i] = complex(rcx), complex(tcx)
        _rs, _rp, _ts, _tp, Rtot, Ttot = sim.SMM_aniso_get_r_t_power(
            layers_list, th_c, float(wl), pol
        )
        R[i], T[i] = float(Rtot), float(Ttot)
        Eamp[i], Ephase[i] = e_amp_phase(
            *sim.SMM_aniso_get_e_field_at_depth(
                layers_list, th_c, float(wl), int(layer_index), float(depth_in_layer), pol
            )
        )

    return SweepResult(name, ang_deg, r, t, R, T, Eamp, Ephase, r_x=r_x, t_x=t_x)


def sweep_rcwa_per_angle(
    layers_d: Sequence[Any],
    *,
    pol: str,
    angles_rad: np.ndarray,
    wl: float = WL_UM,
    n_G: int = 1,
    which_layer: int = 0,
    z_offset: float = 0.0,
    name: str = "RCWA",
    with_cross: bool = False,
) -> SweepResult:
    """Sweep RCWA via TMM-aligned triad: get_r_t_coeff / get_r_t_power / get_e_field_at_depth."""
    layers_list = list(layers_d)
    ang_deg = np.degrees(angles_rad)
    n = len(angles_rad)
    r, t, r_x, t_x, R, T, Eamp, Ephase = _empty_rt_arrays(n, with_cross=with_cross)

    for i, th in enumerate(angles_rad):
        th_deg = float(np.degrees(th))
        rc, tc, rcx, tcx = sim.RCWA_get_r_t_coeff(
            layers_list, float(wl), n_G=n_G, theta_deg=th_deg, pol=pol
        )
        r[i], t[i] = complex(rc), complex(tc)
        if with_cross:
            r_x[i], t_x[i] = complex(rcx), complex(tcx)
        Rp, Tp = sim.RCWA_get_r_t_power(
            layers_list, float(wl), n_G=n_G, theta_deg=th_deg, pol=pol
        )
        R[i], T[i] = float(Rp), float(Tp)
        Eamp[i], Ephase[i] = e_amp_phase(
            *sim.RCWA_get_e_field_at_depth(
                layers_list,
                float(wl),
                n_G=n_G,
                theta_deg=th_deg,
                pol=pol,
                which_layer=which_layer,
                z_offset=float(z_offset),
            )
        )

    return SweepResult(name, ang_deg, r, t, R, T, Eamp, Ephase, r_x=r_x, t_x=t_x)


def _stamp_notebook_refresh_markdown(nb: dict[str, Any], ts: str) -> None:
    """Ensure the notebook opens with a visible refresh timestamp (first markdown cell)."""
    import uuid

    stamp = f"**刷新时间:** `{ts}`\n"
    cells = nb.setdefault("cells", [])
    md_idx = next(
        (i for i, c in enumerate(cells) if c.get("cell_type") == "markdown"), None
    )
    if md_idx is None:
        cells.insert(
            0,
            {
                "cell_type": "markdown",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": [stamp + "\n"],
            },
        )
        return

    cell = cells[md_idx]
    src = cell.get("source", [])
    text = "".join(src) if isinstance(src, list) else str(src)
    lines = text.splitlines(keepends=True)
    lines = [ln for ln in lines if not ln.startswith("**刷新时间:**")]
    # Drop a single leading blank line left by a previous stamp.
    if lines and lines[0].strip() == "":
        lines = lines[1:]
    cell["source"] = [stamp + "\n", *lines]


def refresh_closed_loop_notebooks(assets_dir: str | os.PathLike[str] | None = None) -> None:
    """Re-execute ``*.ipynb`` in the closed_loop dir and rewrite outputs (no jupyter dep).

    Cell sources are exec'd in order; ``viz_io.show_figure`` is patched so Agg figures
    become ``display_data`` PNG outputs (same shape as interactive Run All).
    Each notebook is stamped with a refresh timestamp (markdown + first-cell print).
    """
    import base64
    import contextlib
    import io
    import json
    import sys
    import uuid
    from datetime import datetime
    from pathlib import Path

    root = Path(assets_dir) if assets_dir is not None else Path(__file__).resolve().parent
    notebooks = sorted(root.glob("*.ipynb"))
    if not notebooks:
        print(f"refresh_closed_loop_notebooks: no ipynb under {root}")
        return

    batch_ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"closed_loop notebook refresh start: {batch_ts}")

    os.environ["SAVE_TO_FILE"] = "0"
    viz_io.SAVE_TO_FILE = False

    for nb_path in notebooks:
        ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        print(f"refreshing {nb_path.name} at {ts}")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        _stamp_notebook_refresh_markdown(nb, ts)
        for cell in nb.get("cells", []):
            if "id" not in cell:
                cell["id"] = uuid.uuid4().hex[:8]
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None

        ns: dict[str, Any] = {"__name__": "__main__", "__file__": str(nb_path)}
        exec_count = 0
        cwd_prev = Path.cwd()
        first_code = True
        try:
            os.chdir(root)
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                exec_count += 1
                cell["execution_count"] = exec_count
                src = cell.get("source", [])
                code = "".join(src) if isinstance(src, list) else str(src)
                # Notebook magic is not available outside IPython.
                code = "\n".join(
                    ln for ln in code.splitlines() if not ln.strip().startswith("%")
                )
                if first_code:
                    code = f'print("notebook refreshed at {ts}")\n' + code
                    first_code = False
                outputs: list[dict[str, Any]] = []
                stdout_buf = io.StringIO()
                stderr_buf = io.StringIO()

                def _flush_stream(buf: io.StringIO, name: str) -> None:
                    text = buf.getvalue()
                    if not text:
                        return
                    buf.seek(0)
                    buf.truncate(0)
                    outputs.append({"output_type": "stream", "name": name, "text": text})

                def _show_figure_capture(fig=None):
                    _flush_stream(stdout_buf, "stdout")
                    _flush_stream(stderr_buf, "stderr")
                    if fig is None:
                        fig = plt.gcf()
                    bio = io.BytesIO()
                    fig.savefig(bio, format="png", bbox_inches="tight")
                    plt.close(fig)
                    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
                    outputs.append(
                        {
                            "output_type": "display_data",
                            "metadata": {},
                            "data": {
                                "image/png": b64,
                                "text/plain": "<IPython.core.display.Image object>",
                            },
                        }
                    )

                old_show = viz_io.show_figure
                viz_io.show_figure = _show_figure_capture  # type: ignore[assignment]
                try:
                    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
                        stderr_buf
                    ):
                        try:
                            exec(compile(code, str(nb_path), "exec"), ns, ns)
                        except SystemExit as exc:
                            # pytest.main may sys.exit
                            code_rc = int(getattr(exc, "code", 0) or 0)
                            if code_rc != 0:
                                raise RuntimeError(
                                    f"{nb_path.name}: cell exit code {code_rc}"
                                ) from exc
                        except Exception as exc:
                            _flush_stream(stdout_buf, "stdout")
                            _flush_stream(stderr_buf, "stderr")
                            outputs.append(
                                {
                                    "output_type": "error",
                                    "ename": type(exc).__name__,
                                    "evalue": str(exc),
                                    "traceback": [str(exc)],
                                }
                            )
                            cell["outputs"] = outputs
                            nb_path.write_text(
                                json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                            raise
                finally:
                    viz_io.show_figure = old_show  # type: ignore[assignment]
                _flush_stream(stdout_buf, "stdout")
                _flush_stream(stderr_buf, "stderr")
                cell["outputs"] = outputs
        finally:
            os.chdir(cwd_prev)

        nb_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        n_png = sum(
            1
            for c in nb.get("cells", [])
            for o in c.get("outputs", [])
            if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})
        )
        print(f"refreshed {nb_path.name}: {n_png} figure(s) (stamp {ts})")
