"""Trajectory analysis + Parseval subband decomposition of the refinement curve.

Reads scan_dir.py CSVs that were produced with --bands (extra
``L{lvl}_{band}_step_k`` columns). Keeps the familiar total-PSNR trajectory
(delta / slope, real vs synthetic) and ADDS a decomposition of where the
reconstruction error lives, per subband, over the refinement steps:

  * energy ratio   — band_energy / sum(bands) = share of MSE in each subband
                     (sums to 1; scale-invariant; the "subband ratio" view)
  * band-PSNR    — -10*log10(band_energy): a per-band PSNR with its own
                     delta/slope (the "delta-PSNR-LL" extension). Offset from the
                     logged total PSNR by a constant (decomposition is in model
                     space, not the unsigned [0,1] space), which cancels in
                     delta/slope.
  * raw energy     — band_energy on a log axis; additive (sums to pixel MSE).

The decomposition is exact only for an orthonormal wavelet + mode='zero'
(how scan_dir --bands logs it); ratios then sum to 1 and band-PSNRs are
consistent with the total.

Usage:
    python src/plot_wavelet_curve.py results/synth_bands.csv results/real_bands.csv \
        --out results/wavelet_decomp.png --metric ratio psnr energy
    python src/plot_wavelet_curve.py results/combined_bands.csv --metric ratio --auc
"""

import argparse
import csv
import json
import math
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# slope metric's step-1 column has no window yet (all-NaN) -> nanmean empty-slice
warnings.filterwarnings("ignore", message="Mean of empty slice")

_BAND_RE = re.compile(r"^(L\d+_(?:LL|LH|HL|HH))_step_(\d+)$")
_PALETTE = ["steelblue", "darkorange", "seagreen", "crimson", "mediumpurple", "brown"]


def load_csv(path: Path, tag: str | None) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if tag:
        for r in rows:
            r["label"] = r["label"] if r.get("label") == tag else f"{tag}/{r['label']}"
    return rows


def discover(records: list[dict]) -> tuple[list[str], int]:
    """Return (ordered band keys, n_steps) from the band columns present."""
    keys, steps = [], 0
    for col in records[0]:
        m = _BAND_RE.match(col)
        if m:
            if m.group(1) not in keys:
                keys.append(m.group(1))
            steps = max(steps, int(m.group(2)))
    return keys, steps


def _matrix(group: list[dict], prefix: str, steps: int) -> np.ndarray:
    return np.array([[float(r[f"{prefix}{k}"]) for k in range(1, steps + 1)]
                     for r in group], dtype=float)


def _band_matrix(group: list[dict], key: str, steps: int) -> np.ndarray:
    return _matrix(group, f"{key}_step_", steps)


# --- baseline-aware x-axis helpers (mirrors plot_curves.py) ---------------
# PSNR panels plot the baseline at x=0 so the curve fills the axis from the
# start; old CSVs where baseline == step_1 drop the duplicate step_1.

def _baseline_value(row: dict) -> float:
    return float(row.get("baseline", row.get("step_1")))


def _has_duplicate_baseline_step(records: list[dict]) -> bool:
    if "baseline" not in records[0] or "step_1" not in records[0]:
        return False
    return all(np.isclose(_baseline_value(r), float(r["step_1"])) for r in records)


def _psnr_step_ids(steps: int, drop_dup: bool) -> np.ndarray:
    return np.array([0] + list(range(2 if drop_dup else 1, steps + 1)))


def _psnr_value(row: dict, step: int) -> float:
    return _baseline_value(row) if step == 0 else float(row[f"step_{step}"])


def _psnr_matrix(group: list[dict], step_ids: np.ndarray) -> np.ndarray:
    return np.array([[_psnr_value(r, int(k)) for k in step_ids] for r in group])


def compute_auc(records, steps, score_fn):
    """Per-step AUC (synthetic-positive) for an arbitrary per-record score_fn(record, k)."""
    from sklearn.metrics import roc_auc_score
    labels = [0 if r["label"].startswith("real") else 1 for r in records]
    if len(set(labels)) < 2:
        return [float("nan")] * steps
    aucs = []
    for k in range(1, steps + 1):
        scores = [score_fn(r, k) for r in records]
        try:
            aucs.append(roc_auc_score(labels, scores))
        except ValueError:
            aucs.append(float("nan"))
    return aucs


def band_values(group, keys, steps, metric):
    """Return {band: [n_images, steps] matrix} for the chosen metric + the total."""
    energy = {k: _band_matrix(group, k, steps) for k in keys}
    total = sum(energy.values())  # == pixel MSE per image per step
    if metric == "energy":
        return energy, total
    if metric == "ratio":
        return {k: energy[k] / np.clip(total, 1e-30, None) for k in keys}, total
    if metric == "psnr":
        return {k: -10.0 * np.log10(np.clip(energy[k], 1e-30, None)) for k in keys}, total
    if metric in ("dpsnr", "psnr_delta", "centered_psnr"):
        pp = {k: -10.0 * np.log10(np.clip(energy[k], 1e-30, None)) for k in keys}
        return {k: pp[k] - pp[k][:, [0]] for k in keys}, total
    if metric == "slope":
        # trailing-window-10 OLS slope of each band's band-PSNR at every step
        # (sign-anchored: real degrades <0 / synth compresses >0). Vectorized:
        # slope = sum((x-xbar)*y) / sum((x-xbar)^2) over the trailing window.
        W = 10
        out = {}
        for k, e in energy.items():
            pp = -10.0 * np.log10(np.clip(e, 1e-30, None))   # [n, steps]
            n, S = pp.shape
            sl = np.full((n, S), np.nan)
            for j in range(S):
                lo = max(0, j - W + 1)
                if j - lo >= 1:
                    x = np.arange(lo, j + 1, dtype=float)
                    xc = x - x.mean()
                    sl[:, j] = (pp[:, lo:j + 1] * xc).sum(1) / (xc * xc).sum()
            out[k] = sl
        return out, total
    if metric == "gainperz":
        # Per-band analog of gain/z: each band's band-PSNR gain since step1, per
        # unit latent travel = (pp_b(k) - pp_b(1)) / zd_k. Needs zd_k columns.
        if "zd_1" not in group[0]:
            raise SystemExit("--metric gainperz needs zd_k columns — rescan with the updated scan_dir.py")
        zd = _matrix(group, "zd_", steps)                       # [n, steps], zd_k
        pp = {k: -10.0 * np.log10(np.clip(energy[k], 1e-30, None)) for k in keys}
        return ({k: (pp[k] - pp[k][:, [0]]) / np.clip(zd, 1e-9, None) for k in keys}, total)
    raise ValueError(metric)


_METRIC_LABEL = {
    "ratio":  "energy ratio (share of MSE)",
    "psnr":   "band-PSNR  -10·log10(energy)",
    "dpsnr":  "Δ band-PSNR from step 1 (dB)",
    "psnr_delta": "Δ band-PSNR from step 1 (dB)",
    "centered_psnr": "Δ band-PSNR from step 1 (dB)",
    "energy": "band energy (log)",
    "gainperz": "band-PSNR gain / z-travel",
    "slope": "band-PSNR slope (w10, dB/step)",
}


def _per_step_auc(vals: np.ndarray, labels: np.ndarray) -> list[float]:
    """Direction-agnostic separability AUC per step for a [n_images, steps] matrix."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for k in range(vals.shape[1]):
        try:
            a = roc_auc_score(labels, vals[:, k])
            aucs.append(max(a, 1 - a))
        except ValueError:
            aucs.append(float("nan"))
    return aucs


def _feature_fn(token, keys):
    """Resolve a feature token to (score(record, step), psnr_domain, label).

    Tokens: 'psnr' (raw PSNR@step), 'dpsnr'/'delta' (PSNR-baseline),
    'dropfrac:BAND' ((e1-ek)/e1), 'energy:BAND', 'ratio:BAND'. psnr_domain marks
    features defined on the [0..N] PSNR axis (baseline at 0) vs the [1..N] band axis.
    """
    if token == "psnr":
        return (lambda r, k: _psnr_value(r, int(k)), True, "PSNR")
    if token in ("dpsnr", "delta"):
        return (lambda r, k: _psnr_value(r, int(k)) - _baseline_value(r), True, "ΔPSNR")
    if token == "gainperz":
        # PSNR gained per unit latent travel = manifold efficiency: synth >> 0
        # (cheap PSNR, on-manifold), real <= 0 (travels far for little/neg gain).
        # Needs zd_k columns (scan_dir logs them). Band-domain (k>=1; zd_0=0).
        def f(r, k):
            if f"zd_{int(k)}" not in r:
                raise SystemExit("gainperz needs zd_k columns — rescan with the updated scan_dir.py")
            zd = float(r[f"zd_{int(k)}"])
            return (_psnr_value(r, int(k)) - _baseline_value(r)) / (zd if abs(zd) > 1e-9 else 1e-9)
        return (f, False, "ΔPSNR / latent-travel")
    if ":" not in token:
        raise SystemExit(f"unknown feature token: {token!r} (psnr/dpsnr/<kind>:BAND)")
    kind, band = token.split(":", 1)
    if "+" in band:
        # '+'-joined bands average the feature (e.g. slope:L3_LH+L3_HL = off-diagonal)
        subs = [_feature_fn(f"{kind}:{b}", keys) for b in band.split("+")]
        psnr_dom = subs[0][1]

        def f(r, k):
            vs = [fn(r, k) for fn, _, _ in subs]
            vs = [v for v in vs if v == v]
            return sum(vs) / len(vs) if vs else float("nan")
        return (f, psnr_dom, f"{band} {kind}")
    if kind == "dropfrac":
        def f(r, k):
            e1 = float(r[f"{band}_step_1"]); ek = float(r[f"{band}_step_{int(k)}"])
            return (e1 - ek) / (e1 if e1 > 0 else 1e-30)
    elif kind == "energy":
        f = lambda r, k: float(r[f"{band}_step_{int(k)}"])
    elif kind == "ratio":
        def f(r, k):
            tot = sum(float(r[f"{b}_step_{int(k)}"]) for b in keys)
            return float(r[f"{band}_step_{int(k)}"]) / (tot if tot > 0 else 1e-30)
    elif kind == "gainperz":
        # Per-band-PSNR gain since step 1 per unit latent travel:
        # (-10log10(e_b(k)) - same@1) / zd_k. Per-band analog of gain/z (synth>0, real<=0).
        def f(r, k):
            if f"zd_{int(k)}" not in r:
                raise SystemExit("gainperz:BAND needs zd_k columns — rescan with scan_dir.py")
            e1 = max(float(r[f"{band}_step_1"]), 1e-30)
            ek = max(float(r[f"{band}_step_{int(k)}"]), 1e-30)
            zd = float(r[f"zd_{int(k)}"])
            return (-10 * np.log10(ek) + 10 * np.log10(e1)) / (zd if abs(zd) > 1e-9 else 1e-9)
        return (f, False, f"band-PSNR gain/z {band}")
    elif kind in ("slope", "slope10"):
        # Trailing-window-10 OLS slope of the band-PSNR ending at step k.
        # Sign-anchored: real degrades band (<0) / synth compresses (>0). The
        # calibration-free detector feature; animating k shows the sign-flip emerge
        # as the window clears the warmup transient.
        def f(r, k):
            k = int(k); lo = max(1, k - 9)
            pp = [-10.0 * np.log10(max(float(r[f"{band}_step_{j}"]), 1e-30))
                  for j in range(lo, k + 1)]
            if len(pp) < 2:
                return float("nan")
            return float(np.polyfit(np.arange(lo, k + 1), pp, 1)[0])
        return (f, False, f"{band} band-PSNR slope (w10)")
    else:
        raise SystemExit(
            f"unknown feature kind: {kind!r} "
            "(use dropfrac/energy/ratio/gainperz/slope)"
        )
    return (f, False, token)


def _maybe_kde_correct(token, fn, lab, slope_offsets, slope_offset_band, enable: bool):
    """Apply a saved slope KDE offset when the feature token matches the table."""
    if not enable or not slope_offsets:
        return fn, lab
    if ":" not in token:
        return fn, lab
    kind, band = token.split(":", 1)
    if kind not in ("slope", "slope10"):
        return fn, lab
    if slope_offset_band and band != slope_offset_band:
        return fn, lab

    def f(r, k):
        return fn(r, k) - slope_offsets.get(int(k), 0.0)

    return f, f"{lab} (KDE-corrected)"


def _auc_curve(records, token, keys, steps, psnr_xs,
               slope_offsets=None, slope_offset_band=None, kde_correction: bool = False):
    """Per-step direction-agnostic separability AUC for a feature token."""
    from sklearn.metrics import roc_auc_score
    y = np.array([0 if r["label"].startswith("real") else 1 for r in records])
    f, psnr_dom, lab = _feature_fn(token, keys)
    f, lab = _maybe_kde_correct(token, f, lab, slope_offsets, slope_offset_band, kde_correction)
    xs = psnr_xs if psnr_dom else np.arange(1, steps + 1)

    def sep(scores):
        try:
            a = roc_auc_score(y, scores)
            return max(a, 1 - a)
        except ValueError:
            return float("nan")

    return np.asarray(xs), [sep([f(r, int(k)) for r in records]) for k in xs], lab


def _plot_hist(ax, records, label_names, color_map, token, keys, step,
               slope_offsets=None, slope_offset_band=None, kde_correction: bool = False):
    """Distribution of a feature token at a chosen step, per label (hist + KDE)."""
    if token == "confusion":
        from sklearn.metrics import confusion_matrix
        f, _, lab = _feature_fn("dpsnr", keys)
        f, lab = _maybe_kde_correct("dpsnr", f, lab, slope_offsets, slope_offset_band, kde_correction)
        y_true = np.array([0 if r["label"].startswith("real") else 1 for r in records])
        scores = np.array([f(r, int(step)) for r in records], dtype=float)
        y_pred = (scores > 0).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=12,
                    color="white" if v > cm.max() * 0.5 else "black")
        ax.set_xticks([0, 1], ["pred real", "pred synth"])
        ax.set_yticks([0, 1], ["true real", "true synth"])
        ax.set_xlabel(f"{lab} @ step {step}")
        ax.set_title("Confusion matrix", fontsize=10)
        return
    from scipy.stats import gaussian_kde
    f, _, lab = _feature_fn(token, keys)
    f, lab = _maybe_kde_correct(token, f, lab, slope_offsets, slope_offset_band, kde_correction)

    def vals(grp):
        return np.array([f(r, int(step)) for r in grp])

    allv = vals(records)
    lo, hi = float(allv.min()), float(allv.max())
    edges = np.linspace(lo, hi, 25) if hi > lo else 25
    kx = np.linspace(lo, hi, 300) if hi > lo else np.array([lo])
    for name in label_names:
        grp = [r for r in records if r["label"] == name]
        v = vals(grp)
        ax.hist(v, bins=edges, alpha=0.35, color=color_map[name], density=True)
        if len(v) >= 4 and v.std() > 1e-9:
            ax.plot(kx, gaussian_kde(v)(kx), color=color_map[name], linewidth=1.6,
                    label=f"{name} (n={len(v)})")
        else:
            ax.plot([], [], color=color_map[name], label=f"{name} (n={len(v)})")
    # decision threshold divider at 0 (the sign-detector boundary: real<0 / synth>0)
    if hi > lo and lo <= 0 <= hi:
        ax.axvline(0, color="k", linestyle="--", linewidth=1.1, alpha=0.8,
                   label="threshold (0)")
    ax.set(xlabel=f"{lab} @ step {step}", ylabel="density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)




def _plot_confusion(ax, records, label_names, color_map, token, keys, step,
                    slope_offsets=None, slope_offset_band=None, kde_correction: bool = False):
    """Plot a 2x2 confusion matrix for a binary split at threshold 0."""
    from sklearn.metrics import confusion_matrix
    f, _, lab = _feature_fn(token, keys)
    f, lab = _maybe_kde_correct(token, f, lab, slope_offsets, slope_offset_band, kde_correction)
    y_true = np.array([0 if r["label"].startswith("real") else 1 for r in records])
    scores = np.array([f(r, int(step)) for r in records], dtype=float)
    y_pred = (scores > 0).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=11,
                color="white" if v > cm.max() * 0.5 else "black")
    ax.set_xticks([0, 1], ["pred real", "pred synth"])
    ax.set_yticks([0, 1], ["true real", "true synth"])
    ax.set_xlabel(f"{lab} @ step {step}")
    ax.set_title("Confusion matrix", fontsize=10)
    return im
def plot(records, keys, steps, out: Path, metrics: list[str],
         title: str, ncols: int, show_bands: bool,
         auc_curves: list[str], panel_auc: bool = False,
         panel_auc_mode: str = "drop",
         hist_token: str = "dpsnr", hist_step: int | None = None,
         traj: str = "psnr", n_std: float = 1.0, objective: str = "loss",
         band_pct: tuple[float, float] | None = None,
         slope_offsets: dict[int, float] | None = None,
         slope_offset_band: str | None = None,
         kde_correction: bool = False,
         view: str = "all",
         top_panel: str = "hist") -> None:
    def _bounds(arr, mu):
        """Shaded-ribbon (lower, upper): percentile band if --band-pct set (never
        extends past real data; honest for skewed dists), else mean ± n_std·σ."""
        if band_pct is not None:
            return (np.percentile(arr, band_pct[0], axis=0),
                    np.percentile(arr, band_pct[1], axis=0))
        sd = n_std * arr.std(0)
        return mu - sd, mu + sd
    xs = np.arange(1, steps + 1)              # band-metric steps (no baseline column)
    label_names = list(dict.fromkeys(r["label"] for r in records))
    color_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(label_names)}
    has_labels = any(n.startswith("real") for n in label_names) and len(label_names) >= 2
    drop_dup = _has_duplicate_baseline_step(records)
    psnr_xs = _psnr_step_ids(steps, drop_dup)
    has_loss = all(f"loss_{k}" in records[0] for k in range(1, steps + 1))
    show_top = view in ("all", "top", "traj", "objective", "auc", "hist")
    show_band_grid = view in ("all", "bands")
    show_ribbons = show_bands
    single_panel = view in ("traj", "objective", "auc", "hist")

    # per-band grid: one ROW per decomposition level, columns = LH | HL | HH | LL.
    # LL is present only where logged (coarsest level by default; higher levels if
    # the CSV has them, e.g. all_ll); absent panels are left blank.
    _BAND_COLS = ["LH", "HL", "HH", "LL"]
    max_level = max(int(re.match(r"L(\d+)_", k).group(1)) for k in keys)
    ncols_band = len(_BAND_COLS)
    nrows_band = max_level
    bot_rows = nrows_band * len(metrics)

    if single_panel:
        fig = plt.figure(figsize=(8.6, 6.4), constrained_layout=True)
        gs_top = fig.add_gridspec(2, 2)
        gs_bot = None
    elif show_top and show_band_grid:
        fig = plt.figure(figsize=(4.2 * ncols_band, 3.0 * (2 + bot_rows) + 1),
                         constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[2, bot_rows])
        gs_top = outer[0].subgridspec(2, 2)
        gs_bot = outer[1].subgridspec(bot_rows, ncols_band)
    elif show_top:
        fig = plt.figure(figsize=(8.6, 6.4), constrained_layout=True)
        gs_top = fig.add_gridspec(2, 2)
        gs_bot = None
    else:
        fig = plt.figure(figsize=(4.2 * ncols_band, 3.0 * bot_rows + 1),
                         constrained_layout=True)
        gs_top = None
        gs_bot = fig.add_gridspec(bot_rows, ncols_band)

    # Top summary: full 2x2 block.
    if view in ("all", "top"):
        if traj == "gainperz":
            tf = _feature_fn("gainperz", keys)[0]
            traj_xs = np.arange(1, steps + 1)
            def _traj_mat(grp):
                return np.array([[tf(r, k) for k in traj_xs] for r in grp])
            traj_ylabel, traj_title, traj_unit = "gain/z (ΔPSNR per latent travel)", "gain/z trajectory", "/step"
        else:
            traj_xs = psnr_xs
            def _traj_mat(grp):
                return _psnr_matrix(grp, traj_xs)
            traj_ylabel, traj_title, traj_unit = "total PSNR (dB)", "PSNR trajectory", " dB/step"

        axp = fig.add_subplot(gs_top[0, 0])
        for name in label_names:
            grp = [r for r in records if r["label"] == name]
            arr = _traj_mat(grp); mu = arr.mean(0)
            slope = np.polyfit(traj_xs, mu, 1)[0]
            axp.plot(traj_xs, mu, color=color_map[name],
                     label=f"{name} (n={len(grp)}, slope={slope:+.3g}{traj_unit})")
            if show_ribbons:
                lo, hi = _bounds(arr, mu)
                axp.fill_between(traj_xs, lo, hi, color=color_map[name], alpha=0.18)
        axp.set(ylabel=traj_ylabel, xlabel="step")
        axp.set_title(title or traj_title, fontsize=10)
        axp.grid(True, alpha=0.3); axp.margins(x=0); axp.set_xlim(traj_xs[0], traj_xs[-1])
        axp.legend(fontsize=7, loc="upper right")

        axl = fig.add_subplot(gs_top[0, 1])
        if objective == "dpsnr":
            tf = _feature_fn("dpsnr", keys)[0]
            for name in label_names:
                grp = [r for r in records if r["label"] == name]
                arr = np.array([[tf(r, k) for k in xs] for r in grp]); mu = arr.mean(0)
                axl.plot(xs, mu, color=color_map[name], label=name)
                if show_ribbons:
                    lo, hi = _bounds(arr, mu)
                    axl.fill_between(xs, lo, hi, color=color_map[name], alpha=0.18)
            axl.axhline(0, color="k", lw=0.6, alpha=0.5)
            axl.set(ylabel="ΔPSNR = PSNR - baseline (dB)", xlabel="step")
            axl.grid(True, alpha=0.3); axl.margins(x=0); axl.set_xlim(xs[0], xs[-1]); axl.legend(fontsize=7, loc="upper right")
            axl.set_title("ΔPSNR trajectory", fontsize=10)
        elif objective == "psnr":
            for name in label_names:
                grp = [r for r in records if r["label"] == name]
                arr = _psnr_matrix(grp, psnr_xs); mu = arr.mean(0)
                axl.plot(psnr_xs, mu, color=color_map[name], label=name)
                if show_ribbons:
                    lo, hi = _bounds(arr, mu)
                    axl.fill_between(psnr_xs, lo, hi, color=color_map[name], alpha=0.18)
            axl.set(ylabel="PSNR (dB)", xlabel="step")
            axl.grid(True, alpha=0.3); axl.margins(x=0); axl.set_xlim(psnr_xs[0], psnr_xs[-1])
            axl.legend(fontsize=7, loc="upper right")
            axl.set_title("PSNR trajectory (raw)", fontsize=10)
        elif has_loss:
            for name in label_names:
                grp = [r for r in records if r["label"] == name]
                arr = _matrix(grp, "loss_", steps); mu = arr.mean(0)
                axl.plot(xs, mu, color=color_map[name], label=name)
                if show_ribbons:
                    lo, hi = _bounds(arr, mu)
                    axl.fill_between(xs, lo, hi, color=color_map[name], alpha=0.18)
            axl.set_yscale("log"); axl.set(ylabel="refinement loss", xlabel="step")
            axl.grid(True, alpha=0.3); axl.margins(x=0); axl.set_xlim(xs[0], xs[-1]); axl.legend(fontsize=7, loc="upper right")
            axl.set_title("Refinement objective", fontsize=10)
        else:
            axl.text(0.5, 0.5, "no loss_* columns", ha="center", va="center", fontsize=9)
            axl.set_xticks([]); axl.set_yticks([])
            axl.set_title("Refinement objective", fontsize=10)

        axa = fig.add_subplot(gs_top[1, 0])
        if has_labels and auc_curves:
            for tok in auc_curves:
                cx, ca, lab = _auc_curve(records, tok, keys, steps, psnr_xs,
                                         slope_offsets, slope_offset_band, kde_correction)
                line = axa.plot(cx, ca, linewidth=1.4, label=lab)[0]
                fin = [(x, a) for x, a in zip(cx, ca) if not np.isnan(a)]
                if fin:
                    bx, ba = max(fin, key=lambda t: t[1])
                    axa.scatter([bx], [ba], s=14, color=line.get_color(), zorder=5)
            axa.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
            axa.set_ylim(0, 1.05); axa.margins(x=0); axa.set_xlim(psnr_xs[0], psnr_xs[-1])
            axa.set(ylabel="separability AUC", xlabel="step")
            axa.grid(True, alpha=0.3); axa.legend(fontsize=6, ncol=2, loc="lower right")
        else:
            axa.text(0.5, 0.5, "AUC needs real + synthetic labels", ha="center", va="center", fontsize=9)
            axa.set_xticks([]); axa.set_yticks([])
        axa.set_title("Detection AUC", fontsize=10)

        if top_panel == "hist":
            axh = fig.add_subplot(gs_top[1, 1])
            h_step = steps if hist_step is None else hist_step
            _plot_hist(axh, records, label_names, color_map, hist_token, keys, h_step,
                       slope_offsets, slope_offset_band, kde_correction)
            axh.set_title("Distribution", fontsize=10)
        else:
            axc = fig.add_subplot(gs_top[1, 1])
            c_step = steps if hist_step is None else hist_step
            _plot_confusion(axc, records, label_names, color_map, hist_token, keys, c_step,
                            slope_offsets, slope_offset_band, kde_correction)

    elif view == "traj":
        if traj == "gainperz":
            tf = _feature_fn("gainperz", keys)[0]
            traj_xs = np.arange(1, steps + 1)
            def _traj_mat(grp):
                return np.array([[tf(r, k) for k in traj_xs] for r in grp])
            traj_ylabel, traj_title, traj_unit = "gain/z (ΔPSNR per latent travel)", "gain/z trajectory", "/step"
        else:
            traj_xs = psnr_xs
            def _traj_mat(grp):
                return _psnr_matrix(grp, traj_xs)
            traj_ylabel, traj_title, traj_unit = "total PSNR (dB)", "PSNR trajectory", " dB/step"
        ax = fig.add_subplot(111)
        for name in label_names:
            grp = [r for r in records if r["label"] == name]
            arr = _traj_mat(grp); mu = arr.mean(0)
            slope = np.polyfit(traj_xs, mu, 1)[0]
            ax.plot(traj_xs, mu, color=color_map[name],
                    label=f"{name} (n={len(grp)}, slope={slope:+.3g}{traj_unit})")
            if show_ribbons:
                lo, hi = _bounds(arr, mu)
                ax.fill_between(traj_xs, lo, hi, color=color_map[name], alpha=0.18)
        ax.set(ylabel=traj_ylabel, xlabel="step")
        ax.set_title(title or traj_title, fontsize=10)
        ax.grid(True, alpha=0.3); ax.margins(x=0); ax.set_xlim(traj_xs[0], traj_xs[-1])
        ax.legend(fontsize=7, loc="upper right")

    elif view == "objective":
        axl = fig.add_subplot(111)
        if objective == "dpsnr":
            tf = _feature_fn("dpsnr", keys)[0]
            for name in label_names:
                grp = [r for r in records if r["label"] == name]
                arr = np.array([[tf(r, k) for k in xs] for r in grp]); mu = arr.mean(0)
                axl.plot(xs, mu, color=color_map[name], label=name)
                if show_ribbons:
                    lo, hi = _bounds(arr, mu)
                    axl.fill_between(xs, lo, hi, color=color_map[name], alpha=0.18)
            axl.axhline(0, color="k", lw=0.6, alpha=0.5)
            axl.set(ylabel="ΔPSNR = PSNR - baseline (dB)", xlabel="step")
            axl.grid(True, alpha=0.3); axl.margins(x=0); axl.set_xlim(xs[0], xs[-1]); axl.legend(fontsize=7, loc="upper right")
            axl.set_title("ΔPSNR trajectory", fontsize=10)
        elif objective == "psnr":
            for name in label_names:
                grp = [r for r in records if r["label"] == name]
                arr = _psnr_matrix(grp, psnr_xs); mu = arr.mean(0)
                axl.plot(psnr_xs, mu, color=color_map[name], label=name)
                if show_ribbons:
                    lo, hi = _bounds(arr, mu)
                    axl.fill_between(psnr_xs, lo, hi, color=color_map[name], alpha=0.18)
            axl.set(ylabel="PSNR (dB)", xlabel="step")
            axl.grid(True, alpha=0.3); axl.margins(x=0); axl.set_xlim(psnr_xs[0], psnr_xs[-1])
            axl.legend(fontsize=7, loc="upper right")
            axl.set_title("PSNR trajectory (raw)", fontsize=10)
        elif has_loss:
            for name in label_names:
                grp = [r for r in records if r["label"] == name]
                arr = _matrix(grp, "loss_", steps); mu = arr.mean(0)
                axl.plot(xs, mu, color=color_map[name], label=name)
                if show_ribbons:
                    lo, hi = _bounds(arr, mu)
                    axl.fill_between(xs, lo, hi, color=color_map[name], alpha=0.18)
            axl.set_yscale("log"); axl.set(ylabel="refinement loss", xlabel="step")
            axl.grid(True, alpha=0.3); axl.margins(x=0); axl.set_xlim(xs[0], xs[-1]); axl.legend(fontsize=7, loc="upper right")
            axl.set_title("Refinement objective", fontsize=10)
        else:
            axl.text(0.5, 0.5, "no loss_* columns", ha="center", va="center", fontsize=9)
            axl.set_xticks([]); axl.set_yticks([])
            axl.set_title("Refinement objective", fontsize=10)

    elif view == "auc":
        axa = fig.add_subplot(111)
        if has_labels and auc_curves:
            for tok in auc_curves:
                cx, ca, lab = _auc_curve(records, tok, keys, steps, psnr_xs,
                                         slope_offsets, slope_offset_band, kde_correction)
                line = axa.plot(cx, ca, linewidth=1.4, label=lab)[0]
                fin = [(x, a) for x, a in zip(cx, ca) if not np.isnan(a)]
                if fin:
                    bx, ba = max(fin, key=lambda t: t[1])
                    axa.scatter([bx], [ba], s=14, color=line.get_color(), zorder=5)
            axa.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
            axa.set_ylim(0, 1.05); axa.margins(x=0); axa.set_xlim(psnr_xs[0], psnr_xs[-1])
            axa.set(ylabel="separability AUC", xlabel="step")
            axa.grid(True, alpha=0.3); axa.legend(fontsize=6, ncol=2, loc="lower right")
        else:
            axa.text(0.5, 0.5, "AUC needs real + synthetic labels", ha="center", va="center", fontsize=9)
            axa.set_xticks([]); axa.set_yticks([])
        axa.set_title("Detection AUC", fontsize=10)

    elif view == "hist":
        axh = fig.add_subplot(111)
        h_step = steps if hist_step is None else hist_step
        _plot_hist(axh, records, label_names, color_map, hist_token, keys, h_step,
                   slope_offsets, slope_offset_band, kde_correction)
        axh.set_title("Distribution", fontsize=10)
    # ---- per-band decomposition grids, one block per metric ----
    if show_band_grid:
        overlay_auc = panel_auc and has_labels
        auc_labels = (np.array([0 if r["label"].startswith("real") else 1 for r in records])
                      if overlay_auc else None)
        r0 = 0
        for metric in metrics:
            per_label = {name: band_values([r for r in records if r["label"] == name],
                                           keys, steps, metric)[0]
                         for name in label_names}
            all_vals = band_values(records, keys, steps, metric)[0] if overlay_auc else None
            for lvl in range(1, max_level + 1):
                for cc, bandname in enumerate(_BAND_COLS):
                    key = f"L{lvl}_{bandname}"
                    rr = r0 + (lvl - 1)
                    ax = fig.add_subplot(gs_bot[rr, cc])
                    if key not in keys:
                        ax.axis("off")
                        continue
                    _plot_band_panel(ax, key, lvl, bandname, cc, metric, per_label,
                                     all_vals, overlay_auc, panel_auc_mode, auc_labels,
                                     label_names, color_map, xs, show_ribbons, _bounds,
                                     ncols_band)
            r0 += nrows_band

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_band_panel(ax, key, lvl, bandname, cc, metric, per_label, all_vals,
                     overlay_auc, panel_auc_mode, auc_labels, label_names,
                     color_map, xs, show_ribbons, _bounds, ncols_band):
    """Draw one per-band panel (titled 'Level X BAND') into ax."""
    for name in label_names:
        arr = per_label[name][key]
        mu = np.nanmean(arr, 0)
        if metric == "slope":
            lbl = f"{name} (end={np.nan_to_num(mu)[-1]:+.2g})"
        else:
            slope = np.polyfit(xs, mu, 1)[0]   # per-step rate over the trajectory
            lbl = f"{name} (slope={slope:+.2g}/step)"
        ax.plot(xs, mu, color=color_map[name], linewidth=1.4, label=lbl)
        if show_ribbons and metric != "energy":
            lo, hi = _bounds(arr, mu)
            ax.fill_between(xs, lo, hi, color=color_map[name], alpha=0.15)
    if metric == "energy":
        ax.set_yscale("log")
    if metric in ("dpsnr", "psnr_delta", "centered_psnr", "gainperz", "slope"):
        ax.axhline(0, color="k", lw=0.6, alpha=0.45)
    title_txt = f"Level {lvl} {bandname}"
    if overlay_auc:
        # 'drop' = separability of the change since step 1 (compressibility);
        # 'absolute' = separability of the band's value at each step.
        vals = all_vals[key]
        if panel_auc_mode == "drop":
            vals = vals - vals[:, [0]]
        aucs = _per_step_auc(vals, auc_labels)
        ax2 = ax.twinx()
        ax2.plot(xs, aucs, color="black", linestyle=":", linewidth=1.1, alpha=0.85)
        ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.35)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=6)
        if cc == ncols_band - 1:
            ax2.set_ylabel(f"AUC ({panel_auc_mode})", fontsize=7)
        else:
            ax2.set_yticklabels([])
        finite = [a for a in aucs if not np.isnan(a)]
        if finite:
            title_txt = f"Level {lvl} {bandname}  AUC*={max(finite):.2f}"
    ax.set_title(title_txt, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.margins(x=0)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel("step", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    if cc == 0:
        ax.set_ylabel(_METRIC_LABEL[metric], fontsize=8)


def _load_kde_offsets(path: Path | None, vae: str | None):
    """Load scan_dir/gainz KDE offset JSONs.

    Supports either a single-table shape:
      {"band": "...", "offsets": {"30": ...}}
    or the consolidated per-VAE shape:
      {"band": "...", "per_vae": {"FLUX.1-dev": {"30": ...}}}
    """
    if path is None:
        return None, None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    band = data.get("band")

    if "per_vae" in data:
        per_vae = data["per_vae"]
        if not vae:
            if len(per_vae) == 1:
                vae = next(iter(per_vae))
            else:
                names = ", ".join(sorted(per_vae))
                raise SystemExit(f"--kde-offset-vae required; table contains: {names}")
        if vae not in per_vae:
            names = ", ".join(sorted(per_vae))
            raise SystemExit(f"VAE {vae!r} not found in {path}; available: {names}")
        offsets = per_vae[vae]
        source = vae
    elif "offsets" in data:
        offsets = data["offsets"]
        source = vae or data.get("vae") or "single"
    else:
        offsets = data
        source = vae or "raw"

    return {int(k): float(v) for k, v in offsets.items()}, band, source


def _slope_band_list(token: str) -> list[str]:
    """Flatten a slope token like ``slope:L2_LH+L2_HH`` to a list of bands."""
    if ":" not in token:
        return []
    kind, band = token.split(":", 1)
    if kind not in ("slope", "slope10"):
        return []
    return [b for b in band.split("+") if b]


def _offdiag_slope_row(rec: dict, steps: int, window: int = 10, bands: tuple[str, ...] = ("L3_LH", "L3_HL")) -> float:
    """Mean trailing-window OLS slope of the selected bands' band-PSNR."""
    lo = max(1, steps - window + 1)
    out = []
    for band in bands:
        try:
            pp = [-10.0 * np.log10(max(float(rec[f"{band}_step_{j}"]), 1e-30))
                  for j in range(lo, steps + 1)]
        except (KeyError, TypeError, ValueError):
            continue
        if len(pp) >= 2:
            out.append(float(np.polyfit(np.arange(lo, steps + 1), pp, 1)[0]))
    return sum(out) / len(out) if out else float("nan")


def _kde_crossover_offset(real_vals, synth_vals):
    """Bayes-ish boundary: KDE density crossover between real and synth scores."""
    r = np.array([float(x) for x in real_vals if x == x], dtype=float)
    s = np.array([float(x) for x in synth_vals if x == x], dtype=float)
    if len(r) < 4 or len(s) < 4:
        return float("nan")
    lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
    if not hi > lo:
        return float("nan")
    pad = 0.05 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 512)
    try:
        from scipy.stats import gaussian_kde
        dr = gaussian_kde(r, bw_method="silverman")(xs)
        ds = gaussian_kde(s, bw_method="silverman")(xs)
        diff = dr - ds
        mlo, mhi = sorted((float(np.median(r)), float(np.median(s))))
        candidates = []
        for j in range(1, len(xs)):
            if diff[j - 1] == 0 or diff[j] == 0 or diff[j - 1] * diff[j] < 0:
                x0, x1 = xs[j - 1], xs[j]
                y0, y1 = diff[j - 1], diff[j]
                x = x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
                candidates.append(float(x))
        between = [x for x in candidates if mlo <= x <= mhi]
        if between:
            mid = 0.5 * (mlo + mhi)
            return min(between, key=lambda x: abs(x - mid))
        mask = (xs >= mlo) & (xs <= mhi)
        if mask.any():
            return float(xs[mask][np.argmin(np.abs(diff[mask]))])
    except Exception:
        pass
    return float(0.5 * (np.median(r) + np.median(s)))


def _compute_kde_offsets_from_records(records, steps: int, band: str, window: int = 10):
    """Compute per-step KDE crossover offsets from the loaded CSV rows."""
    if not band:
        return None
    bands = tuple(b for b in band.split("+") if b)
    if not bands:
        return None
    real_recs = [r for r in records if r["label"].startswith("real")]
    synth_recs = [r for r in records if not r["label"].startswith("real")]
    offsets = {}
    for step in range(2, steps + 1):
        rv = [_offdiag_slope_row(r, step, window, bands) for r in real_recs]
        sv = [_offdiag_slope_row(r, step, window, bands) for r in synth_recs]
        off = _kde_crossover_offset(rv, sv)
        if off == off:
            offsets[step] = float(off)
    return offsets


def plot_ribbon(records, keys, steps, out: Path, ribbon_height, title: str) -> None:
    """PSNR mean line with a stacked ribbon whose slice thicknesses = subband ratios.

    One panel per label. The ribbon is centered on the PSNR mean; its total
    half-height is a fixed dB value or the per-step PSNR std (ribbon_height=="std").
    Slices stack low→high frequency (L1 detail at the bottom, coarse/LL on top)
    and morph along the curve as the residual energy redistributes.
    """
    xs = np.arange(1, steps + 1)
    label_names = list(dict.fromkeys(r["label"] for r in records))
    fig, axes = plt.subplots(1, len(label_names),
                             figsize=(7 * len(label_names), 5), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("viridis")
    band_colors = {k: cmap(i / max(len(keys) - 1, 1)) for i, k in enumerate(keys)}

    for ax, name in zip(axes, label_names):
        grp = [r for r in records if r["label"] == name]
        psnr = _matrix(grp, "step_", steps).mean(0)
        ratios, _ = band_values(grp, keys, steps, "ratio")
        ratio_mean = {k: ratios[k].mean(0) for k in keys}  # sum over k == 1 per step
        if ribbon_height == "std":
            half = _matrix(grp, "step_", steps).std(0)
        else:
            half = np.full(steps, float(ribbon_height))
        cum = np.zeros(steps)
        for k in keys:
            y0 = psnr - half + 2 * half * cum
            cum = cum + ratio_mean[k]
            y1 = psnr - half + 2 * half * cum
            ax.fill_between(xs, y0, y1, color=band_colors[k], alpha=0.85,
                            linewidth=0, label=k)
        ax.plot(xs, psnr, color="black", linewidth=1.6, label="PSNR mean")
        ax.set_title(f"{name}  (n={len(grp)})")
        ax.set_xlabel("step")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        ax.set_xlim(xs[0], xs[-1])
        if ax is axes[0]:
            ax.legend(fontsize=7, ncol=2, loc="upper right")
    hdesc = "±std" if ribbon_height == "std" else f"±{ribbon_height} dB"
    fig.suptitle(title or f"PSNR curve — subband energy-ratio ribbon ({hdesc})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tags", nargs="+", default=None,
                   help="Label prefix per CSV (default: file stem when multiple)")
    p.add_argument("--objective", choices=["loss", "dpsnr", "psnr"], default="dpsnr",
                   help="Top-right panel: 'dpsnr' = ΔPSNR trajectory (default; real<0 / "
                        "synth>0 about a zero line), 'loss' = refinement objective curve, "
                        "'psnr' = raw PSNR trajectory (with baseline at step 0).")
    p.add_argument("--traj", choices=["psnr", "gainperz"], default="psnr",
                   help="Top trajectory panel: 'psnr' (default) or 'gainperz' "
                        "(= (PSNR-baseline)/zd per step; needs zd_k cols).")
    p.add_argument("--metric", nargs="+", default=["psnr"],
                   choices=["ratio", "psnr", "dpsnr", "psnr_delta", "centered_psnr", "energy", "gainperz", "slope"],
                   help="Per-band quantities to decompose (one grid block each; default psnr). "
                        "dpsnr/centered_psnr = band-PSNR minus step-1 value; "
                        "gainperz = (band-PSNR gain since step1)/zd_k per band (needs zd_k cols).")
    p.add_argument("--title", default="")
    p.add_argument("--ncols", type=int, default=5, help="Columns in the per-band grid")
    p.add_argument("--auc-curves", nargs="+", dest="auc_curves",
                   default=["psnr", "slope:L2_LH+L2_HH+L3_LL"], metavar="CURVE",
                   help="Which AUC curves to draw in the Detection-AUC panel "
                        "(default: psnr + the KDE-correctable universal slope). Tokens: "
                        "psnr, dpsnr, gainperz (=ΔPSNR/latent-travel; needs zd_k cols), and "
                        "per-band dropfrac:BAND / energy:BAND / ratio:BAND "
                        "(e.g. dpsnr gainperz dropfrac:L1_HH). Direction-agnostic separability.")
    p.add_argument("--auc", action="store_true",
                   help="(deprecated; the Detection-AUC panel is always shown in the 2x2 top)")
    p.add_argument("--hist", dest="hist", default="slope:L2_LH+L2_HH+L3_LL", metavar="FEATURE",
                   help="Feature for the histogram panel (default: the universal slope; "
                        "same tokens as --auc-curves): "
                        "psnr, dpsnr, dropfrac:BAND, energy:BAND, ratio:BAND "
                        "(e.g. --hist dropfrac:L1_HH or --hist energy:L1_HH).")
    p.add_argument("--hist-step", type=int, default=None, dest="hist_step",
                   help="Step for the histogram feature (default: final step).")
    p.add_argument("--kde-offset-table", type=Path, default=None, dest="kde_offset_table",
                   help="Optional KDE offset JSON from scan_dir.py --save-kde-offset; "
                        "applied to matching slope:<band> AUC/histogram features.")
    p.add_argument("--kde-offset-vae", default=None, dest="kde_offset_vae",
                   help="VAE key to read when --kde-offset-table contains per_vae.")
    p.add_argument("--kde-correction", action="store_true", dest="kde_correction",
                   help="Enable KDE offset correction for slope-based AUC/hist panels.")
    p.add_argument("--panel-auc", action="store_true", dest="panel_auc",
                   help="Overlay each per-band panel with that band's per-step detection AUC "
                        "(dotted black line, right axis 0-1; direction-agnostic separability; "
                        "needs real/synthetic labels). Panel title shows peak AUC.")
    p.add_argument("--panel-auc-mode", choices=["absolute", "drop"], default="drop",
                   dest="panel_auc_mode",
                   help="What the per-band AUC scores: 'drop' (default) = change since step 1 "
                        "(trajectory signal, rises with refinement); 'absolute' = the band's "
                        "value at each step (endpoint signal, tends to decay).")
    p.add_argument("--view", choices=["all", "top", "bands", "traj", "objective", "auc", "hist"], default="all",
                   help="Which part of the figure to render: full figure, top 2x2, lower bands, "
                        "or a single top panel (traj/objective/auc/hist).")
    p.add_argument("--top-panel", choices=["hist", "confusion"], default="hist",
                   help="Bottom-right panel in --view top/all: histogram or confusion matrix.")
    p.add_argument("--no-bands", action="store_true", help="Hide ±std shading")
    p.add_argument("--n-std", type=float, default=1.0, dest="n_std",
                   help="Shaded band width in std-devs (1 = ±1σ default, 2 = ±2σ)")
    p.add_argument("--band-pct", type=float, nargs=2, default=None, dest="band_pct",
                   metavar=("LO", "HI"),
                   help="Use a percentile band (e.g. 10 90) for the shaded ribbons "
                        "instead of mean±n·σ. Non-parametric: never extends past real "
                        "data, so it won't over-state overlap for skewed distributions.")
    p.add_argument("--ribbon", action="store_true",
                   help="Also save a composition-ribbon figure: the PSNR mean line with a "
                        "stacked band segmented by subband energy ratio (one panel per label).")
    p.add_argument("--ribbon-height", default="1.0", metavar="DB|std",
                   help="Total half-height of the ribbon: a dB value (default 1.0) or "
                        "'std' to use the per-step PSNR std (the 'segment the std band' view).")
    args = p.parse_args()

    multi = len(args.csvs) > 1
    tags = args.tags or ([p.stem for p in args.csvs] if multi else [None] * len(args.csvs))
    if len(tags) != len(args.csvs):
        sys.exit(f"--tags count ({len(tags)}) must match CSV count ({len(args.csvs)})")

    records = []
    for path, tag in zip(args.csvs, tags):
        if not path.exists():
            sys.exit(f"File not found: {path}")
        batch = load_csv(path, tag)
        records.extend(batch)
        print(f"Loaded {len(batch)} rows from {path}" + (f" [tag={tag}]" if tag else ""))

    keys, steps = discover(records)
    if not keys:
        sys.exit("No L{lvl}_{band}_step_k columns found — rescan with scan_dir.py --bands.")
    print(f"Bands: {keys}  steps={steps}  labels={sorted(set(r['label'] for r in records))}")

    slope_offsets, slope_offset_band, slope_offset_source = _load_kde_offsets(
        args.kde_offset_table, args.kde_offset_vae)
    if args.kde_correction and not slope_offsets:
        hist_bands = _slope_band_list(args.hist)
        if hist_bands:
            slope_offset_band = "+".join(hist_bands)
            slope_offsets = _compute_kde_offsets_from_records(records, steps, slope_offset_band)
            slope_offset_source = "computed-from-csv"
    if args.kde_correction and slope_offsets:
        final_off = slope_offsets.get(steps)
        final_txt = "missing" if final_off is None else f"{final_off:+.5f}"
        band_txt = slope_offset_band or "<any slope band>"
        src_txt = args.kde_offset_table if args.kde_offset_table else "<computed>"
        print(f"KDE offsets: {src_txt} [{slope_offset_source}] "
              f"band={band_txt} step{steps}={final_txt}")

    out = args.out or args.csvs[0].with_name(args.csvs[0].stem + "_decomp.png")
    plot(records, keys, steps, out, args.metric, args.title,
         args.ncols, show_bands=not args.no_bands, auc_curves=args.auc_curves,
         panel_auc=args.panel_auc, panel_auc_mode=args.panel_auc_mode,
         hist_token=args.hist, hist_step=args.hist_step, traj=args.traj, n_std=args.n_std,
         objective=args.objective,
         band_pct=tuple(args.band_pct) if args.band_pct else None,
         slope_offsets=slope_offsets, slope_offset_band=slope_offset_band,
         kde_correction=args.kde_correction, view=args.view)

    if args.ribbon:
        plot_ribbon(records, keys, steps,
                    out.with_name(out.stem + "_ribbon.png"),
                    args.ribbon_height, args.title)


if __name__ == "__main__":
    main()
