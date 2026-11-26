"""论文式 (17) 的逐网格岩石物理拟合与图件。"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case import DV_KEYS, DV_SCALE, PCS_KEYS, PCS_LIMITS, pcs_to_dv, save_arrays
from results import difference, panel_grid


def fit_pcs(target, initial, active):
    """有界逐点最小二乘，残差按特征物理量归一化。"""
    target = np.stack([target[key] for key in DV_KEYS], axis=-1)
    scale = np.asarray(DV_SCALE)
    index = [PCS_KEYS.index(key) for key in active]
    bounds = np.asarray(PCS_LIMITS)[index]
    model = np.stack([initial[key] for key in PCS_KEYS], axis=-1).astype(np.float64)
    damping = np.full(model.shape[:-1], 1e-4)

    def predict(values):
        return pcs_to_dv(dict(zip(PCS_KEYS, np.moveaxis(values, -1, 0))))

    for _ in range(80):
        prediction = predict(model)
        residual = (np.stack(list(prediction.values()), axis=-1) - target) / scale
        columns = []
        for key, coefficient in zip(DV_KEYS, scale):
            derivative = pcs_to_dv(dict(zip(PCS_KEYS, np.moveaxis(model, -1, 0))),
                                   {key_: float(key == key_) for key_ in DV_KEYS})
            columns.append(np.stack([derivative[field] / coefficient for field in active], axis=-1))
        jacobian = np.stack(columns, axis=-2)
        transpose = np.swapaxes(jacobian, -1, -2)
        gradient = (transpose @ residual[..., None])[..., 0]
        projected = model[..., index] - np.clip(model[..., index] - gradient, *bounds.T)
        if np.max(np.abs(projected)) < 1e-8:
            break
        hessian = transpose @ jacobian + damping[..., None, None] * np.eye(len(active))
        step = np.linalg.solve(hessian, -gradient[..., None])[..., 0]

        trial = model.copy()
        trial[..., index] = np.clip(model[..., index] + step, *bounds.T)
        trial_residual = (np.stack(list(predict(trial).values()), axis=-1) - target) / scale
        improved = np.sum(trial_residual**2, axis=-1) < np.sum(residual**2, axis=-1)
        model = np.where(improved[..., None], trial, model)
        damping = np.clip(np.where(improved, damping / 3, damping * 5), 1e-12, 1e6)

    return dict(zip(PCS_KEYS, np.moveaxis(model, -1, 0)))


def paper_figures(run, output, dh, layered=False):
    def read(file, prefix, keys):
        with np.load(run / file) as arrays:
            return {key: arrays[f"{prefix}_{key}"] for key in keys}

    truth = [read("models.npz", f"pcs_{vintage}", PCS_KEYS) for vintage in ("bl", "ml")]
    initial = read("models.npz", "initial_pcs", PCS_KEYS)
    dv = [read("dv_sim/final.npz", vintage, DV_KEYS) for vintage in ("bl", "ml")]
    direct = [read("pcs_sim/final.npz", vintage, PCS_KEYS) for vintage in ("bl", "ml")]
    baseline = read("pcs_baseline/final.npz", "bl", PCS_KEYS)

    indirect_bl = fit_pcs(dv[0], initial, PCS_KEYS)
    indirect_ml = fit_pcs(dv[1], indirect_bl, ("sw",))
    changes = {"Truth": difference(*truth),
               "DV indirect (Sw only)": difference(indirect_bl, indirect_ml),
               "PCS direct": difference(*direct)}
    indirect_models = {"bl": indirect_bl, "ml": indirect_ml}
    if layered:
        unconstrained_ml = fit_pcs(dv[1], initial, PCS_KEYS)
        changes = {"Truth": changes["Truth"],
                   "DV indirect (all PCS)": difference(indirect_bl, unconstrained_ml),
                   **{key: value for key, value in changes.items() if key != "Truth"}}
        indirect_models["ml_all_pcs"] = unconstrained_ml

    output.mkdir(parents=True, exist_ok=True)
    names = ("ΔPorosity", "ΔClay", "ΔSw")
    elastic_keys = ("ΔVp", "ΔVs", "ΔDensity")
    elastic_units = ("m/s", "m/s", "kg/m³")
    panel_grid(output / "rock_physics_changes.png",
               [(name, [model[key] for key in PCS_KEYS]) for name, model in changes.items()],
               names, ("fraction",) * 3, dh=dh, symmetric=True, shared=True)
    panel_grid(output / "true_rock_physics.png", [
        (name, [model[key] for key in PCS_KEYS])
        for name, model in zip(("Baseline", "Monitor", "Initial"), (*truth, initial))],
        ("Porosity", "Clay", "Sw"), ("fraction",) * 3, dh=dh)
    baseline_rows = [("Truth", truth[0]), ("Initial", initial), ("PCS FWI", baseline)]
    panel_grid(output / "pcs_baseline.png", [
        (name, [*model.values(), *pcs_to_dv(model).values()]) for name, model in baseline_rows],
        ("Porosity", "Clay", "Sw", "Vp", "Vs", "Density"),
        (*(("fraction",) * 3), *elastic_units), dh=dh, shared=True)

    true_sw = changes["Truth"]["sw"]
    true_elastic = difference(*(pcs_to_dv(model) for model in truth))
    elastic = {"Truth": true_elastic, "DV": difference(*dv),
               "PCS": difference(*(pcs_to_dv(model) for model in direct))}
    regions = {"whole": np.ones_like(true_sw, dtype=bool)}
    columns = {}
    if layered:
        regions["reservoir"] = true_sw != 0
        columns["center"] = true_sw.shape[1] // 2
    else:
        for label, mask in (("left", true_sw > 0), ("right", true_sw < 0)):
            regions[label] = mask
            columns[label] = int(np.median(np.where(mask)[1]))
    panel_grid(output / "elastic_changes.png",
               [(name, [model[key] for key in DV_KEYS]) for name, model in elastic.items()],
               elastic_keys, elastic_units, dh=dh, symmetric=True, shared=True)

    off_reservoir = true_sw == 0
    with (output / "saturation_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("method", "region", "relative_l2", "rmse", "estimate_min",
                         "estimate_max", "off_reservoir_rms"))
        for method, model in changes.items():
            if method == "Truth":
                continue
            estimate = model["sw"]
            for region, mask in regions.items():
                error = estimate[mask] - true_sw[mask]
                writer.writerow((method, region, np.linalg.norm(error) / np.linalg.norm(true_sw[mask]),
                                 np.sqrt(np.mean(error**2)), estimate[mask].min(),
                                 estimate[mask].max(), np.sqrt(np.mean(estimate[off_reservoir]**2))))

    parameters = ("phi", "cc", "sw", *DV_KEYS) if layered else ("sw", *DV_KEYS)
    depth = dh * np.arange(true_sw.shape[0])
    fig, axes = plt.subplots(len(columns), len(parameters), squeeze=False,
                             figsize=(3 * len(parameters), 3.5 * len(columns)), layout="constrained")
    with (output / "profiles.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("line", "x_m", "z_m", "parameter", "method", "value"))
        for i, (line, column) in enumerate(columns.items()):
            for j, key in enumerate(parameters):
                ax = axes[i, j]
                series = changes if key in PCS_KEYS else elastic
                for method, model in series.items():
                    profile = model[key][:, column]
                    ax.plot(profile, depth, label=method)
                    writer.writerows((line, column * dh, z, key, method, value)
                                      for z, value in zip(depth, profile))
                ax.set(title=f"Δ{key} at x={column * dh:g} m", ylabel="Depth (m)")
                ax.invert_yaxis()
                if j in (0, len(parameters) - 1):
                    ax.legend(fontsize=7)
    fig.savefig(output / "profiles.png", dpi=155)
    plt.close(fig)
    save_arrays(output / "indirect_models.npz",
                **{f"{tag}_{key}": value for tag, model in indirect_models.items()
                   for key, value in model.items()})
    if layered:
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
        column = columns["center"]
        fields = (*PCS_KEYS, *DV_KEYS)
        for ax, key in zip(axes.flat, fields):
            for name, model in baseline_rows:
                values = model if key in PCS_KEYS else pcs_to_dv(model)
                ax.plot(values[key][:, column], depth, label=name)
            ax.set(title=key, ylabel="Depth (m)")
            ax.invert_yaxis()
        axes[0, 0].legend()
        fig.savefig(output / "baseline_center_profile.png", dpi=155)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    paper_figures(args.run, args.out, dh=5.)
