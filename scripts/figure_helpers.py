from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from model_deposition import (
    Cunningham__correction,
    MU_AIR,
    RHO_WATER,
    stokes_number,
    v_flow,
)
from sensitivity_helpers import LOBES, parse_strain


MODEL_ORDER = ["chan_lipp", "yeh_schum", "zhang"]
MODEL_LABELS = {
    "chan_lipp": "Chan-Lipp",
    "yeh_schum": "Yeh-Schum",
    "zhang": "Zhang",
}
MODEL_COLORS = {
    "chan_lipp": "#1f77b4",
    "yeh_schum": "#2ca02c",
    "zhang": "#ff7f0e",
}
PARTICLE_SIZES_UM = [0.5, 1.0, 2.0]
MAX_GEN = 25


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _repo_root() / "data"


def _suffix_name(stem: str, suffix: str = "", ext: str = ".png") -> str:
    return f"{stem}_{suffix}{ext}" if suffix else f"{stem}{ext}"


def _size_label(size_um: float) -> str:
    return "05" if np.isclose(size_um, 0.5) else str(int(size_um))


def _parse_particle_size_um(info_text: str) -> float | None:
    match = re.search(r"Particle size:\s*([0-9.]+)", info_text)
    return float(match.group(1)) if match else None


def _read_info(path: Path, mouse_id: str) -> str:
    info_path = path / f"{mouse_id}_info.txt"
    if not info_path.exists():
        info_path = path / f"{mouse_id}_Info.md"
    return info_path.read_text() if info_path.exists() else ""


def load_result_tables(results_path: str | Path, max_gen: int = MAX_GEN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one completed simulation result folder.

    Returns
    -------
    all_deposition, all_ground_truth
        Long concatenations of per-mouse result CSVs. Both include mouse_id,
        strain, and gt_particle_size columns when the mouse info file is present.
    """
    results_path = Path(results_path)
    dep_files = sorted(results_path.glob("m*/m*_all_models_deposition_results.csv"))
    if not dep_files:
        dep_files = sorted(results_path.glob("**/*_all_models_deposition_results.csv"))
    if not dep_files:
        raise FileNotFoundError(f"No *_all_models_deposition_results.csv files in {results_path}")

    dep_frames: list[pd.DataFrame] = []
    gt_frames: list[pd.DataFrame] = []
    for dep_file in dep_files:
        mouse_id = dep_file.parent.name
        info_text = _read_info(dep_file.parent, mouse_id)
        particle_um = _parse_particle_size_um(info_text)
        try:
            strain = parse_strain(info_text) if info_text else "unknown"
        except Exception:
            strain = "unknown"

        dep = pd.read_csv(dep_file)
        dep["mouse_id"] = mouse_id
        dep["strain"] = strain
        dep["gt_particle_size"] = particle_um
        if "generation" in dep:
            dep = dep[dep["generation"].notna() & (dep["generation"] <= max_gen)]
        dep_frames.append(dep)

        gt_file = dep_file.parent / f"{mouse_id}_ground_truth.csv"
        if gt_file.exists():
            gt = pd.read_csv(gt_file)
            gt["mouse_id"] = mouse_id
            gt["strain"] = strain
            gt["gt_particle_size"] = particle_um
            if "generation" in gt:
                gt = gt[gt["generation"].notna() & (gt["generation"] <= max_gen)]
            gt_frames.append(gt)

    all_dep = pd.concat(dep_frames, ignore_index=True)
    all_gt = pd.concat(gt_frames, ignore_index=True) if gt_frames else pd.DataFrame()
    return all_dep, all_gt


def _style_axes(ax, *, legend: bool = False) -> None:
    ax.set_xlim(1, MAX_GEN)
    ax.set_xticks([1] + list(range(5, MAX_GEN + 1, 5)))
    if ax.get_yscale() == "log":
        ymin, _ = ax.get_ylim()
        ax.set_ylim(bottom=max(ymin, 1e-12))
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=11, framealpha=0.9)


def _save(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / name,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",
        edgecolor="none",
        transparent=True,
    )
    plt.close(fig)


def plot_mechanism_comparison(all_dep: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    for size_um in PARTICLE_SIZES_UM:
        size_m = size_um * 1e-6
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        for model in MODEL_ORDER:
            col = f"{model}_impaction_{size_m:.2e}"
            if col in all_dep:
                gen = all_dep.groupby("generation")[col].mean()
                ax.plot(gen.index, gen.values, "o-", color=MODEL_COLORS[model],
                        label=f"Impaction ({MODEL_LABELS[model]})", linewidth=1.8)

        for col, label, color, marker in [
            (f"sedimentation_{size_m:.2e}", "Sedimentation", "#2ecc71", "s-"),
            (f"diffusion_{size_m:.2e}", "Diffusion", "#3498db", "^-"),
        ]:
            if col in all_dep:
                gen = all_dep.groupby("generation")[col].mean()
                ax.plot(gen.index, gen.values, marker, color=color, label=label, linewidth=1.8)

        _style_axes(ax, legend=np.isclose(size_um, 0.5))
        fig.tight_layout()
        _save(fig, output_dir, _suffix_name(f"comparison_hq_{_size_label(size_um)}m", suffix))


def plot_ground_truth_by_generation(all_gt: pd.DataFrame, output_dir: Path) -> None:
    if all_gt.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for size_um in sorted(all_gt["gt_particle_size"].dropna().unique()):
        gt_ps = all_gt[all_gt["gt_particle_size"] == size_um]
        per_mouse = gt_ps.groupby(["mouse_id", "generation"])["probability"].sum().unstack(fill_value=0)
        mean = per_mouse.mean()
        ax.plot(mean.index, mean.values, "o-", label=f"{size_um:g} um")
    _style_axes(ax, legend=True)
    fig.tight_layout()
    _save(fig, output_dir, "dep_per_gen_gt.png")


def plot_ground_truth_overview(all_gt: pd.DataFrame, output_dir: Path) -> None:
    if all_gt.empty:
        return
    for density in [False, True]:
        fig, ax = plt.subplots(figsize=(8, 8))
        value_col = "probability"
        work = all_gt.copy()
        if density and "area" in work:
            area = work["area"].replace(0, np.nan)
            work["deposition_density"] = work["probability"] / area
            value_col = "deposition_density"
        summary = work.groupby(["gt_particle_size", "generation"])[value_col].mean().reset_index()
        pivot = summary.pivot(index="generation", columns="gt_particle_size", values=value_col).fillna(0)
        im = ax.imshow(pivot.T, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels([f"{c:g} um" for c in pivot.columns])
        ax.set_xticks(range(0, len(pivot.index), max(1, len(pivot.index) // 6)))
        ax.set_xticklabels([int(pivot.index[i]) for i in ax.get_xticks()])
        ax.set_xlabel("Generation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        name = "deposition_gt_density.png" if density else "deposition_gt.png"
        fig.tight_layout()
        _save(fig, output_dir, name)


def plot_ground_truth_by_strain(all_gt: pd.DataFrame, output_dir: Path) -> None:
    if all_gt.empty or "strain" not in all_gt:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, size_um in zip(axes, PARTICLE_SIZES_UM):
        gt_ps = all_gt[all_gt["gt_particle_size"] == size_um]
        if gt_ps.empty:
            continue
        for strain, group in gt_ps.groupby("strain"):
            per_mouse = group.groupby(["mouse_id", "generation"])["probability"].sum().unstack(fill_value=0)
            mean = per_mouse.mean()
            ax.plot(mean.index, mean.values, "o-", label=strain, linewidth=1.4, markersize=3)
        _style_axes(ax, legend=np.isclose(size_um, 0.5))
        ax.set_title(f"{size_um:g} um")
    fig.tight_layout()
    _save(fig, output_dir, "dep_per_gen_gt_by_strain.png")


def _normalized_generation_curves(
    all_dep: pd.DataFrame,
    all_gt: pd.DataFrame,
    size_um: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    size_m = size_um * 1e-6
    dep_ps = all_dep[all_dep["gt_particle_size"] == size_um]
    gt_ps = all_gt[all_gt["gt_particle_size"] == size_um]
    model_curves: dict[str, dict[str, pd.Series]] = {m: {} for m in MODEL_ORDER}
    gt_curves: dict[str, pd.Series] = {}

    for mouse_id, group in gt_ps.groupby("mouse_id"):
        per_gen = group.groupby("generation")["probability"].sum()
        total = per_gen.sum()
        if total > 0:
            gt_curves[mouse_id] = per_gen / total

    for mouse_id, group in dep_ps.groupby("mouse_id"):
        for model in MODEL_ORDER:
            col = f"{model}_deposition_{size_m:.2e}"
            if col not in group:
                continue
            per_gen = group.groupby("generation")[col].sum()
            total = per_gen.sum()
            if total > 0:
                model_curves[model][mouse_id] = per_gen / total

    return {m: pd.DataFrame(v).T for m, v in model_curves.items()}, pd.DataFrame(gt_curves).T


def plot_generation_variability(
    all_dep: pd.DataFrame,
    all_gt: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
) -> None:
    if all_gt.empty:
        return
    for size_um in PARTICLE_SIZES_UM:
        model_dfs, gt_df = _normalized_generation_curves(all_dep, all_gt, size_um)
        if gt_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for model, df in model_dfs.items():
            if df.empty:
                continue
            mean = df.mean()
            sd = df.std()
            ax.plot(mean.index, mean.values, "o-", color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model], linewidth=1.8)
            ax.fill_between(mean.index, (mean - sd).values, (mean + sd).values,
                            color=MODEL_COLORS[model], alpha=0.15)
        gt_mean = gt_df.mean()
        gt_sd = gt_df.std()
        ax.plot(gt_mean.index, gt_mean.values, "k--", label="Ground Truth", linewidth=2)
        ax.fill_between(gt_mean.index, (gt_mean - gt_sd).values, (gt_mean + gt_sd).values,
                        color="gray", alpha=0.2)
        _style_axes(ax, legend=np.isclose(size_um, 0.5))
        fig.tight_layout()
        _save(fig, output_dir, _suffix_name(f"dep_per_gen_{_size_label(size_um)}_variability", suffix))


def _child_labels(row: pd.Series) -> list[int]:
    out = []
    for col in ["child_1", "child_2", "child_3"]:
        if col in row and pd.notna(row[col]):
            out.append(int(row[col]))
    return out


def _sum_descendants(df: pd.DataFrame, root_label: int, value_col: str) -> float:
    by_label = df.set_index("label", drop=False)
    stack = [int(root_label)]
    total = 0.0
    while stack:
        label = stack.pop()
        if label not in by_label.index:
            continue
        row = by_label.loc[label]
        value = row.get(value_col, 0.0)
        total += float(value) if pd.notna(value) else 0.0
        stack.extend(_child_labels(row))
    return total


def lobe_fraction_tables(all_dep: pd.DataFrame, all_gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    gt_rows = []
    for mouse_id, dep_mouse in all_dep.groupby("mouse_id"):
        root_rows = []
        for lobe in LOBES:
            candidates = dep_mouse[dep_mouse["name"] == lobe]
            if not candidates.empty:
                root_rows.append((lobe, int(candidates["label"].min())))
        if not root_rows:
            continue
        gt_mouse = all_gt[all_gt["mouse_id"] == mouse_id] if not all_gt.empty else pd.DataFrame()
        gt_size = dep_mouse["gt_particle_size"].dropna().iloc[0] if dep_mouse["gt_particle_size"].notna().any() else None

        for size_um in PARTICLE_SIZES_UM:
            size_m = size_um * 1e-6
            for model in MODEL_ORDER:
                col = f"{model}_deposition_{size_m:.2e}"
                if col not in dep_mouse:
                    continue
                vals = np.array([_sum_descendants(dep_mouse, label, col) for _, label in root_rows], dtype=float)
                total = vals.sum()
                if total > 0:
                    vals = vals / total
                for (lobe, _), val in zip(root_rows, vals):
                    model_rows.append({
                        "mouse_id": mouse_id,
                        "particle_size": size_um,
                        "model": model,
                        "lobe": lobe,
                        "fraction": val,
                    })

            if gt_size is not None and np.isclose(gt_size, size_um) and not gt_mouse.empty:
                vals = np.array([_sum_descendants(gt_mouse, label, "probability") for _, label in root_rows], dtype=float)
                total = vals.sum()
                if total > 0:
                    vals = vals / total
                for (lobe, _), val in zip(root_rows, vals):
                    gt_rows.append({
                        "mouse_id": mouse_id,
                        "particle_size": size_um,
                        "lobe": lobe,
                        "fraction": val,
                    })
    return pd.DataFrame(model_rows), pd.DataFrame(gt_rows)


def plot_lobe_bars(all_dep: pd.DataFrame, all_gt: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    model_lobes, gt_lobes = lobe_fraction_tables(all_dep, all_gt)
    if model_lobes.empty:
        return
    for size_um in PARTICLE_SIZES_UM:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(LOBES))
        n_bars = len(MODEL_ORDER) + (0 if gt_lobes.empty else 1)
        width = 0.8 / n_bars
        for i, model in enumerate(MODEL_ORDER):
            sub = model_lobes[(model_lobes["particle_size"] == size_um) & (model_lobes["model"] == model)]
            if sub.empty:
                continue
            stats_df = sub.groupby("lobe")["fraction"].agg(["mean", "std"]).reindex(LOBES).fillna(0)
            ax.bar(x + i * width, stats_df["mean"], width=width, yerr=stats_df["std"],
                   capsize=3, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        if not gt_lobes.empty:
            sub = gt_lobes[gt_lobes["particle_size"] == size_um]
            if not sub.empty:
                stats_df = sub.groupby("lobe")["fraction"].agg(["mean", "std"]).reindex(LOBES).fillna(0)
                ax.bar(x + len(MODEL_ORDER) * width, stats_df["mean"], width=width,
                       yerr=stats_df["std"], capsize=3, label="Ground Truth", color="#572F30")
        ax.set_xticks(x + width * (n_bars - 1) / 2)
        ax.set_xticklabels(LOBES, rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if np.isclose(size_um, 0.5) and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=10)
        fig.tight_layout()
        _save(fig, output_dir, _suffix_name(f"dep_per_lobe_{_size_label(size_um)}", suffix))


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    out = np.full_like(p_values, np.nan)
    valid = ~np.isnan(p_values)
    p = p_values[valid]
    if len(p) == 0:
        return out
    order = np.argsort(p)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, len(p) + 1)
    adj = np.minimum(p * len(p) / ranked, 1.0)
    adj_sorted = np.minimum.accumulate(adj[order[::-1]])[::-1]
    restored = np.empty(len(p))
    restored[order] = adj_sorted
    out[valid] = restored
    return out


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_lobe_error_heatmaps(all_dep: pd.DataFrame, all_gt: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    model_lobes, gt_lobes = lobe_fraction_tables(all_dep, all_gt)
    if model_lobes.empty or gt_lobes.empty:
        return
    for size_um in [1.0, 2.0]:
        records = []
        for model in MODEL_ORDER:
            for lobe in LOBES:
                pred = model_lobes[
                    (model_lobes["particle_size"] == size_um)
                    & (model_lobes["model"] == model)
                    & (model_lobes["lobe"] == lobe)
                ][["mouse_id", "fraction"]]
                gt = gt_lobes[
                    (gt_lobes["particle_size"] == size_um)
                    & (gt_lobes["lobe"] == lobe)
                ][["mouse_id", "fraction"]]
                merged = pred.merge(gt, on="mouse_id", suffixes=("_pred", "_gt"))
                if merged.empty:
                    mean_err = np.nan
                    p_val = np.nan
                else:
                    diff = merged["fraction_pred"] - merged["fraction_gt"]
                    mean_err = float(diff.mean() * 100.0)
                    p_val = stats.ttest_rel(merged["fraction_pred"], merged["fraction_gt"]).pvalue if len(merged) > 1 else np.nan
                records.append({"model": model, "lobe": lobe, "mean_err": mean_err, "p": p_val})
        df = pd.DataFrame(records)
        if df["mean_err"].notna().sum() == 0:
            continue
        df["q"] = _bh_adjust(df["p"].to_numpy())
        matrix = df.pivot(index="model", columns="lobe", values="mean_err").reindex(index=MODEL_ORDER, columns=LOBES)
        qmat = df.pivot(index="model", columns="lobe", values="q").reindex(index=MODEL_ORDER, columns=LOBES)
        vmax = max(1.0, np.nanmax(np.abs(matrix.to_numpy())))

        fig, ax = plt.subplots(figsize=(5.2, 2.9))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(LOBES)))
        ax.set_xticklabels(LOBES, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
        for r, model in enumerate(MODEL_ORDER):
            for c, lobe in enumerate(LOBES):
                val = matrix.loc[model, lobe]
                q = qmat.loc[model, lobe]
                if pd.isna(val):
                    continue
                if not pd.isna(q) and q >= 0.05:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="lightgray", zorder=2))
                star = _stars(float(q))
                label = f"{val:+.1f}" if star in ("", "ns") else f"{val:+.1f}\n{star}"
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(c, r, label, ha="center", va="center", fontsize=8, color=color, zorder=3)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        base = "stat_heatmap_bh" if suffix else "heatmap_bh"
        _save(fig, output_dir, _suffix_name(f"{base}_{_size_label(size_um)}", suffix))


def plot_stk_impaction(all_dep: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    size_m = 1e-6
    df = all_dep[all_dep["gt_particle_size"] == 1.0].copy()
    if df.empty:
        df = all_dep.copy()
    if df.empty:
        return
    d = 2.0 * df["radius"].to_numpy(dtype=float)
    q = df["flow"].to_numpy(dtype=float)
    cc = Cunningham__correction(size_m)
    df["stk"] = stokes_number(v_flow(q, d), RHO_WATER, size_m, cc, MU_AIR, d)
    gen_stk = df.groupby("generation")["stk"].mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gen_stk.index, gen_stk.values, "o-", color="#3498db", linewidth=2, label="Stokes number")
    for model in MODEL_ORDER:
        col = f"{model}_impaction_{size_m:.2e}"
        if col in df:
            gen = df.groupby("generation")[col].mean()
            ax.plot(gen.index, gen.values, "o--", color=MODEL_COLORS[model],
                    linewidth=1.5, markersize=4, label=f"P_i {MODEL_LABELS[model]}")
    _style_axes(ax, legend=True)
    fig.tight_layout()
    _save(fig, output_dir, _suffix_name("stk_impaction_vs_generation", suffix))


def plot_morphometry(all_dep: pd.DataFrame, output_dir: Path) -> None:
    sys.path.insert(0, str(_data_dir()))
    from morphometry_references import ISLAM_BALBC_2017, WEIBEL_HUMAN_1963

    morph = all_dep.groupby("generation").agg(radius=("radius", "mean")).reset_index()
    morph["diameter_mm"] = morph["radius"] * 2 * 1000
    weibel_gen = WEIBEL_HUMAN_1963["generation"] + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(morph["generation"], morph["diameter_mm"], "o-", label="LAPDMouse", color="#2ecc71")
    ax.plot(ISLAM_BALBC_2017["generation"], ISLAM_BALBC_2017["diameter_mm"],
            "o-", label="Islam 2017", color="#e67e22")
    ax.plot(weibel_gen, WEIBEL_HUMAN_1963["diameter_mm"],
            "o-", label="Weibel 1963", color="#3498db")
    ax.set_yscale("log")
    _style_axes(ax, legend=True)
    fig.tight_layout()
    _save(fig, output_dir, "morphometry_absolute_diameter.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(morph["generation"], morph["diameter_mm"] / morph["diameter_mm"].iloc[0],
            "o-", label="LAPDMouse", color="#2ecc71")
    ax.plot(ISLAM_BALBC_2017["generation"], ISLAM_BALBC_2017["diameter_mm"] / ISLAM_BALBC_2017["diameter_mm"].iloc[0],
            "o-", label="Islam 2017", color="#e67e22")
    ax.plot(weibel_gen, WEIBEL_HUMAN_1963["diameter_mm"] / WEIBEL_HUMAN_1963["diameter_mm"].iloc[0],
            "o-", label="Weibel 1963", color="#3498db")
    ax.set_yscale("log")
    _style_axes(ax, legend=False)
    fig.tight_layout()
    _save(fig, output_dir, "morphometry_normalized_diameter.png")


def plot_deposition_transition(output_dir: Path, csv_path: str | Path | None = None) -> None:
    if csv_path is None:
        csv_path = _data_dir() / "heyder_deposition_transition.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    required = {"Particle diameter (um)", "Region (normalized)", "Deposition fraction"}
    if "Particle diameter (um)" not in df and "Particle diameter (µm)" in df:
        df = df.rename(columns={"Particle diameter (µm)": "Particle diameter (um)"})
    if not required.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for region, group in df.groupby("Region (normalized)"):
        if region not in {"Total", "Laryngeal", "Bronchial", "Alveolar"}:
            continue
        summary = group.groupby("Particle diameter (um)")["Deposition fraction"].mean().sort_index()
        ax.plot(summary.index, summary.values, "o-", label=region)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, output_dir, "deposition_transition.png")


def run_all_figures(
    results_path: str | Path,
    output_dir: str | Path = "outputs",
    suffix: str = "",
    *,
    include_static: bool | None = None,
) -> None:
    """
    Generate the publication figure set from a result folder.

    To regenerate variants, keep the notebook/code unchanged and only change
    `results_path` plus `suffix`, for example suffix="area_scaled_gen4".
    """
    output_dir = Path(output_dir)
    all_dep, all_gt = load_result_tables(results_path)

    plot_mechanism_comparison(all_dep, output_dir, suffix=suffix)
    plot_generation_variability(all_dep, all_gt, output_dir, suffix=suffix)
    plot_lobe_bars(all_dep, all_gt, output_dir, suffix=suffix)
    plot_lobe_error_heatmaps(all_dep, all_gt, output_dir, suffix=suffix)
    plot_stk_impaction(all_dep, output_dir, suffix=suffix)

    if include_static is None:
        include_static = suffix == ""
    if include_static:
        plot_ground_truth_by_generation(all_gt, output_dir)
        plot_ground_truth_overview(all_gt, output_dir)
        plot_ground_truth_by_strain(all_gt, output_dir)
        plot_morphometry(all_dep, output_dir)
        plot_deposition_transition(output_dir)
