"""论文样式图、储层剖面及数值误差。"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from case import DH, DT, DV_KEYS, pcs_to_dv, save_arrays


def difference(baseline, monitor):
    return {key: monitor[key] - values for key, values in baseline.items()}


def panel_grid(path, rows, names, units, *, symmetric=False, shared=False, dh=DH):
    fig, axes = plt.subplots(len(rows), len(names), figsize=(4.3 * len(names), 2.7 * len(rows)),
                             squeeze=False, layout="constrained")
    for j, (name, unit) in enumerate(zip(names, units)):
        truth = rows[0][1][j]
        for i, (label, images) in enumerate(rows):
            values = images[j]
            nz, nx = values.shape
            reference = truth if shared else values
            if symmetric and shared and not np.any(truth):
                reference = np.stack([images[j] for _, images in rows])
            bounds = (reference.min(), reference.max())
            if symmetric:
                amplitude = max(abs(bounds[0]), abs(bounds[1]), 1e-12)
                bounds = (-amplitude, amplitude)
            ax = axes[i, j]
            image = ax.imshow(values, extent=(0, dh * nx / 1000, dh * nz / 1000, 0),
                              interpolation="nearest", aspect="equal",
                              cmap="RdBu_r" if symmetric else "viridis",
                              vmin=bounds[0], vmax=bounds[1])
            ax.set(title=f"{label}: {name}", xlabel="x (km)", ylabel="z (km)")
            fig.colorbar(image, ax=ax, shrink=.77, label=unit)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_gathers(path, observed, nr, ns):
    k = ns // 2
    baseline, monitor = (data[:, k * nr:(k + 1) * nr] for data in observed)
    maximum = max(np.max(np.abs(baseline)), np.max(np.abs(monitor)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, data, title in zip(axes, (baseline, monitor, 5 * (monitor - baseline)),
                               ("Baseline", "Monitor", "5 × (Monitor − Baseline)")):
        ax.imshow(data, cmap="gray", vmin=-maximum, vmax=maximum, aspect="auto",
                  extent=(0, nr, (data.shape[0] - 1) * DT, 0))
        ax.set(title=title, xlabel="Trace", ylabel="Time (s)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_true_models(output, elastic, dh=DH):
    names = ("Vp", "Vs", "Density")
    units = ("m/s", "m/s", "kg/m³")
    panel_grid(output / "true_elastic.png",
               [(label, list(model.values())) for label, model in zip(("Baseline", "Monitor"), elastic)],
               names, units, shared=True, dh=dh)
    panel_grid(output / "true_elastic_difference.png",
               [("Truth", list(difference(*elastic).values()))],
               ("ΔVp", "ΔVs", "Δρ"), units, symmetric=True, dh=dh)


def evaluate(output, pcs, elastic, dv_result, pcs_result=None):
    elastic_truth = difference(*elastic)
    dv_change = difference(*dv_result)
    sat_truth = difference(*pcs)["sw"]
    elastic_rows = [("Truth", list(elastic_truth.values())), ("DV", list(dv_change.values()))]
    values = {key: {"DV": dv_change[key]} for key in DV_KEYS}
    truths = {**elastic_truth}
    units = ("m/s", "m/s", "kg/m³")
    arrays = {f"truth_delta_{key}": truth for key, truth in elastic_truth.items()}
    arrays.update({f"dv_delta_{key}": value for key, value in dv_change.items()})
    if pcs_result is not None:
        pcs_change = difference(*pcs_result)
        from_pcs = difference(*(pcs_to_dv(model) for model in pcs_result))
        elastic_rows.append(("PCS", list(from_pcs.values())))
        for key in DV_KEYS:
            values[key]["PCS"] = from_pcs[key]
            arrays[f"pcs_delta_{key}"] = from_pcs[key]
        truths["sw"] = sat_truth
        values["sw"] = {"PCS": pcs_change["sw"]}
        arrays.update({"truth_delta_sw": sat_truth,
                       **{f"pcs_delta_{key}": value for key, value in pcs_change.items()}})
        panel_grid(output / "saturation_change.png", [
            ("Truth", [sat_truth]), ("PCS", [pcs_change["sw"]]),
            ("Error", [pcs_change["sw"] - sat_truth]),
        ], ("ΔSw",), ("fraction",), symmetric=True, shared=True)

    panel_grid(output / "elastic_change_comparison.png", elastic_rows,
               ("ΔVp", "ΔVs", "Δρ"), units, symmetric=True, shared=True)
    masks = {"left_Sw_increase": sat_truth > 0, "right_Sw_decrease": sat_truth < 0}
    off_reservoir = sat_truth == 0
    profile_columns = [int(np.median(np.where(mask)[1])) for mask in masks.values()]
    depth = DH * np.arange(sat_truth.shape[0])
    fig, axes = plt.subplots(2, len(truths), figsize=(3.1 * len(truths), 7),
                             squeeze=False, layout="constrained")
    profile_rows = []
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("parameter", "method", "region", "relative_l2_error", "truth_min",
                         "truth_max", "estimate_min", "estimate_max", "off_reservoir_rms"))
        for j, (key, truth) in enumerate(truths.items()):
            for method, estimate in values[key].items():
                for name, mask in [("whole", np.ones_like(truth, bool)), *masks.items()]:
                    relative = np.linalg.norm((estimate - truth)[mask]) / np.linalg.norm(truth[mask])
                    writer.writerow((key, method, name, relative, truth[mask].min(), truth[mask].max(),
                                     estimate[mask].min(), estimate[mask].max(),
                                     np.sqrt(np.mean(estimate[off_reservoir]**2))))
            for i, (name, column) in enumerate(zip(masks, profile_columns)):
                ax = axes[i, j]
                for method, estimate in {"Truth": truth, **values[key]}.items():
                    ax.plot(estimate[:, column], depth, label=method)
                    profile_rows.extend(zip([name] * len(depth), [DH * column] * len(depth),
                                            depth, [key] * len(depth), [method] * len(depth),
                                            estimate[:, column]))
                ax.invert_yaxis()
                ax.set(title=f"{key}, x={column * DH:g} m", xlabel="Change", ylabel="Depth (m)")
                ax.legend()
    with (output / "reservoir_profiles.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("reservoir", "x_m", "z_m", "parameter", "series", "value"))
        writer.writerows(profile_rows)
    fig.savefig(output / "reservoir_profiles.png", dpi=170)
    plt.close(fig)
    save_arrays(output / "time_lapse_changes.npz", **arrays)
