#!/usr/bin/env python3
"""在同一观测炮集上复现 Marmousi 的 DV 和 PCS 同时式时移反演。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy as np
import pyopencl as cl
import scipy

from PyFWI import acquisition
from PyFWI.wave_propagation import WavePropagator

from case import (
    DH, DT, FDOM, DV_LIMITS, DV_SCALE, PCS_LIMITS, PCS_SCALE, DV_KEYS, PCS_KEYS,
    check_pcs_derivative, load_models, pcs_to_dv, save_arrays, save_json, smooth_model,
)
from check_data import main as check_data
from indirect import paper_figures
from invert import pressure, run_stage
from parallel import ParallelWave
from results import evaluate, plot_gathers, plot_true_models


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="结果目录")
    parser.add_argument("--case", choices=("marmousi", "layered"), default="marmousi")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default=os.environ.get("PYOPENCL_CTX"),
                        help="OpenCL 平台:设备索引，默认使用分配到的第一张 A10")
    parser.add_argument("--allow-non-a10", action="store_true", help="允许其他 OpenCL 设备用于验证")
    parser.add_argument("--forward-only", action="store_true", help="只运行数据、正演和炮集输出")
    parser.add_argument("--truth", choices=("pcs", "author-dv"), default="pcs",
                        help="共享 PCS 转换炮集，或用作者弹性文件单独跑 DV")
    parser.add_argument("--iterations", type=int, help="覆盖各频段的迭代上限")
    parser.add_argument("--sigma", type=float, default=5., help="高斯平滑标准差，单位为网格点")
    parser.add_argument("--coupling", type=float, default=1e-3, help="无量纲差异惩罚权重")
    parser.add_argument("--tikhonov-x", type=float, default=0.,
                        help="归一化模型的一阶横向差分惩罚权重")
    parser.add_argument("--resume", action="store_true", help="复用本目录已完成的频段")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发炮的工作进程数；0 表示每炮一个，默认单进程")
    args = parser.parse_args()
    if ((args.iterations is not None and args.iterations < 1) or not np.isfinite(args.sigma)
            or args.sigma <= 0 or not np.isfinite(args.coupling) or args.coupling < 0
            or not np.isfinite(args.tikhonov_x) or args.tikhonov_x < 0
            or args.workers < 0):
        parser.error("迭代次数和平滑标准差须为正，差异惩罚权重须非负，工作进程数须非负")
    if args.case == "layered" and (args.truth != "pcs" or args.workers != 1):
        parser.error("层状算例使用 PCS 真值和单进程")
    return args


def select_device(requested, allow_other):
    devices = {
        f"{i}:{j}": device for i, platform in enumerate(cl.get_platforms())
        for j, device in enumerate(platform.get_devices())
    }
    if requested is None:
        candidates = [index for index, device in devices.items() if "A10" in device.name.upper()]
        requested = next(iter(candidates or (devices if allow_other else ())), None)
    if requested not in devices or (not allow_other and "A10" not in devices[requested].name.upper()):
        available = {index: device.name for index, device in devices.items()}
        raise RuntimeError(f"未选中 A10。指定 --device 平台:设备；可用设备 {available}")
    os.environ["PYOPENCL_CTX"] = requested
    return requested, devices[requested]


def layered_models():
    # hu_laminar 的层值、层界和注入形状，见 model_dataset.py 及论文 Fig. 2。
    shape = (100, 100)
    initial = {k: np.full(shape, value) for k, value in zip(PCS_KEYS, (.30, .05, .20))}
    baseline = {key: field.copy() for key, field in initial.items()}
    for key, upper, lower in zip(PCS_KEYS, (.20, .10, .50), (.10, .15, .80)):
        baseline[key][37:65] = upper
        baseline[key][65:] = lower
    monitor = {key: field.copy() for key, field in baseline.items()}
    z, x = np.ogrid[:shape[0], :shape[1]]
    reservoir = (z >= 37) & ((z - 37)**2 + (x - 50)**2 < 8**2)
    monitor["sw"][reservoir] *= 1.25
    pcs = [baseline, monitor]
    return pcs, [pcs_to_dv(model) for model in pcs], initial


def main():
    args = arguments()
    index, device = select_device(args.device, args.allow_non_a10)
    layered = args.case == "layered"
    dh = 10. if layered else DH
    if layered:
        pcs, elastic, initial_pcs = layered_models()
        initial_dv = pcs_to_dv(initial_pcs)
        model_differences = None
    else:
        check_data()
        pcs, elastic, model_differences = load_models(args.truth)
        initial_dv = smooth_model(elastic[0], args.sigma, DV_LIMITS)
        initial_pcs = smooth_model(pcs[0], args.sigma, PCS_LIMITS)
    derivative_check = check_pcs_derivative(initial_pcs)
    print(f"PCS 链式梯度方向导数相对误差: {derivative_check}", flush=True)
    shape = pcs[0]["phi"].shape
    ns = 1 if args.profile == "smoke" else 7
    frequencies = [10] if args.profile == "smoke" else [10, 20, 25, 35, 40, 55]
    iterations = args.iterations or (2 if args.profile == "smoke" else 10)
    # Snap x coordinates so saved geometry is exactly the geometry in the kernels.
    parameters = {"dh": dh, "dt": DT, "t": .8 if layered else .55, "fdom": FDOM, "sdo": 4,
                  "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh,
                  "acq_type": 2 if layered else 1, "seimogram_shape": "2d", "energy_balancing": False}
    sources, receivers, nr_surface, nr_well = acquisition.acq_parameters(
        ns, dh, dh * shape[1], dh * shape[0], dh, parameters["sdo"],
        acq_type=parameters["acq_type"])
    sources[:, 0] = np.rint(sources[:, 0] / dh) * dh
    receivers[nr_well:nr_well + nr_surface, 1] -= 2 * dh
    source = acquisition.Source(sources, dh, DT)
    source.Ricker(FDOM)

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            capture_output=True, text=True).stdout.strip()
    config = {"profile": args.profile, "truth": args.truth, "frequencies_Hz": frequencies,
              "iterations_per_band": iterations, "coupling": args.coupling, "sigma_grid": args.sigma,
              "forward": parameters, "checkpoint_percentage": 10, "shape_zx": list(shape),
              "sources_xz_m": sources.tolist(), "receivers_xz_m": receivers.tolist(),
              "dv_scale": list(DV_SCALE), "pcs_scale": list(PCS_SCALE),
              "dv_bounds": [list(bound) for bound in DV_LIMITS],
              "pcs_bounds": [list(bound) for bound in PCS_LIMITS]}
    if layered:
        config["case"] = args.case
        config["n_well_receivers_per_side"] = nr_well
    if args.tikhonov_x:
        config["tikhonov_x"] = args.tikhonov_x
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not args.resume or previous["config"] != config or previous["commit"] != commit:
            raise ValueError("输出目录已有运行记录。请换目录，或在同一提交和配置下使用 --resume")
    elif any(output.iterdir()):
        raise ValueError("请选择空的输出目录")
    workers = min(args.workers or ns, ns)
    if workers > 1:
        wave = ParallelWave(parameters, sources, receivers, shape, workers)
    else:
        wave = WavePropagator(parameters, source, receivers, shape, components=0, chpr=10,
                              n_well_rec=nr_well)
    actual = wave.queue.device
    if actual.name != device.name:
        raise RuntimeError(f"OpenCL 实际设备与所选设备不一致: {actual.name}, {device.name}")
    print(f"运行设备 {index}: {actual.name}；网格 {shape}，{ns} 炮，{wave.nr} 道/炮，"
          f"{workers} 个工作进程", flush=True)
    save_json(manifest, {"commit": commit, "config": config, "workers": workers,
                         "device": {"index": index, "name": actual.name,
                                    "platform": actual.platform.name, "driver": actual.driver_version,
                                    "memory_bytes": actual.global_mem_size},
                         "versions": {"python": sys.version, "numpy": np.__version__,
                                      "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                                      "pyopencl": cl.VERSION_TEXT},
                         "samples": {"nt": wave.nt_ext, "internal_nt": wave.nt,
                                     "internal_dt_s": float(wave.dt)},
                         "pcs_derivative_relative_error": derivative_check,
                         "pcs_to_author_dv_relative_l2": model_differences})
    observations_path = output / "observations.npz"
    if args.resume and observations_path.exists():
        with np.load(observations_path) as saved:
            observations = [saved["pressure_bl"], saved["pressure_ml"]]
    else:
        observations = [pressure(wave, model) for model in elastic]
        for label, data in zip(("BL", "ML", "ML−BL"),
                               (*observations, observations[1] - observations[0])):
            if not np.any(data):
                raise ValueError(f"{label} 的炮集全为零")
        save_arrays(observations_path, pressure_bl=observations[0], pressure_ml=observations[1],
                    pressure_delta=observations[1]-observations[0], source_xz_m=sources,
                    receiver_xz_m=receivers, time_s=DT * np.arange(wave.nt_ext))
    save_arrays(output / "models.npz", **{
        f"{prefix}_{key}": values
        for prefix, model in zip(("pcs_bl", "pcs_ml", "dv_bl", "dv_ml", "initial_dv", "initial_pcs"),
                                 (*pcs, *elastic, initial_dv, initial_pcs))
        for key, values in model.items()
    })
    plot_gathers(output / "shot_gathers.png", observations, wave.nr, ns)
    plot_true_models(output, elastic, dh=dh)
    if args.forward_only:
        if workers > 1:
            wave.close()
        return

    def solve(label, initial, keys, active, scale, limits):
        baseline, = run_stage(wave, observations[:1], initial, keys, scale, limits,
                              frequencies, iterations, 0., output / f"{label}_baseline", args.resume,
                              tikhonov_x=args.tikhonov_x)
        return run_stage(wave, observations, baseline, active, scale, limits, frequencies,
                         iterations, args.coupling, output / f"{label}_sim", args.resume,
                         tikhonov_x=args.tikhonov_x)

    dv_result = solve("dv", initial_dv, DV_KEYS, DV_KEYS, DV_SCALE, DV_LIMITS)
    pcs_result = None
    if args.truth == "pcs":
        pcs_result = solve("pcs", initial_pcs, PCS_KEYS, ("sw",), PCS_SCALE, PCS_LIMITS)
    if layered:
        paper_figures(output, output / "paper_figures", dh=dh, layered=True)
    else:
        evaluate(output, pcs, elastic, dv_result, pcs_result)
    if workers > 1:
        wave.close()
    print(f"复现完成，结果目录 {output}", flush=True)


if __name__ == "__main__":
    main()
