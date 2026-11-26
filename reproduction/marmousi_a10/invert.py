"""用 PyFWI 的 OpenCL 正演和伴随实现多尺度 L-BFGS-B。"""

import csv
from time import perf_counter

import numpy as np
from scipy import fft
from scipy.optimize import minimize
from scipy.signal import butter, sosfreqz

from case import DT, DV_KEYS, PCS_KEYS, pcs_to_dv, save_arrays


class Lowpass:
    """以零填充、零相位三阶 Butterworth 对残差和伴随源进行同一滤波。"""

    def __init__(self, nt, cutoff):
        self.nt = nt
        self.nfft = fft.next_fast_len(nt + 512)
        sos = butter(3, cutoff, fs=1 / DT, output="sos")
        angular = 2 * np.pi * fft.rfftfreq(self.nfft)
        _, response = sosfreqz(sos, worN=angular)
        self.gain = (abs(response)**2).astype(np.float32)[:, None]

    def apply(self, data):
        spectrum = fft.rfft(np.asarray(data, dtype=np.float32), n=self.nfft, axis=0)
        return fft.irfft(spectrum * self.gain, n=self.nfft, axis=0)[:self.nt]


def pressure(wave, elastic):
    typed = {key: np.ascontiguousarray(elastic[key], dtype=np.float32) for key in DV_KEYS}
    data = wave.forward_modeling(typed)["taux"]
    if not np.isfinite(data).all():
        raise FloatingPointError("正演炮集出现 NaN/Inf，请检查模型、步长及 OpenCL 设备")
    return data


def data_term(wave, model, observed, lowpass, weight):
    is_pcs = "sw" in model
    calculated = pressure(wave, pcs_to_dv(model) if is_pcs else model)
    residual = lowpass.apply(calculated - observed)
    loss = weight * 0.5 * np.dot(residual.ravel().astype(np.float64), residual.ravel())
    # Pressure = (taux + tauz) / 2 in PyFWI acquisition.seismic_section.
    adjoint = np.ascontiguousarray(lowpass.apply(residual) * (0.5 * weight))
    grad_dv = wave.gradient({"taux": adjoint, "tauz": adjoint})
    return loss, pcs_to_dv(model, grad_dv) if is_pcs else grad_dv


def run_stage(wave, observations, initial, selected, scale, limits, frequencies,
              iterations, coupling, output, resume):
    """Optimize the baseline alone or the two vintages simultaneously."""
    output.mkdir(parents=True, exist_ok=True)
    keys = PCS_KEYS if "sw" in initial else DV_KEYS
    coefficients = dict(zip(keys, scale))
    selected_limits = dict(zip(keys, limits))
    shape = initial[keys[0]].shape
    size = int(np.prod(shape))
    vintages = ("bl",) if len(observations) == 1 else ("bl", "ml")
    block_size = len(selected) * size

    def unpack(x):
        blocks = x.reshape(len(vintages), len(selected), *shape)
        return [
            {**initial, **{key: block * coefficients[key] for key, block in zip(selected, group)}}
            for group in blocks
        ]

    x = np.tile(np.concatenate([(initial[key] / coefficients[key]).ravel() for key in selected]),
                len(vintages)).astype(np.float64)
    bounds = []
    for key in selected:
        lower, upper = selected_limits[key]
        scaled = (lower / coefficients[key], upper / coefficients[key])
        bounds.extend([scaled] * size)
    bounds *= len(vintages)

    for band, cutoff in enumerate(frequencies, 1):
        checkpoint = output / f"band_{band:02d}.npz"
        if resume and checkpoint.exists():
            with np.load(checkpoint) as saved:
                x = saved["x"]
            print(f"{output.name} 读取检查点 {checkpoint.name}", flush=True)
            continue

        lowpass = Lowpass(observations[0].shape[0], cutoff)
        energy = sum(np.linalg.norm(lowpass.apply(data).astype(np.float64))**2
                     for data in observations)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError(f"{cutoff} Hz 的观测炮集没有有效能量")
        weight = 1. / energy
        started = perf_counter()
        evaluations = []

        def objective(values):
            losses, gradients = zip(*[
                data_term(wave, model, obs, lowpass, weight)
                for model, obs in zip(unpack(values), observations)
            ])
            gradient = np.concatenate([
                (derivative[key].astype(np.float64) * coefficients[key]).ravel()
                for derivative in gradients for key in selected
            ])
            penalty = 0.
            if len(vintages) == 2:
                diff = values[block_size:] - values[:block_size]
                factor = coupling / block_size
                penalty = 0.5 * factor * float(diff @ diff)
                gradient[:block_size] -= factor * diff
                gradient[block_size:] += factor * diff
            loss = float(sum(losses) + penalty)
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise FloatingPointError(f"{output.name} {cutoff:g} Hz 的目标函数或梯度非有限")
            norm = float(np.linalg.norm(gradient))
            evaluations.append((loss, float(sum(losses)), penalty, norm))
            history_writer.writerow((len(evaluations), *evaluations[-1], perf_counter() - started))
            history_file.flush()
            print(f"{output.name} {cutoff:g} Hz 评估 {len(evaluations):03d} "
                  f"J={loss:.6e}, |g|={norm:.3e}", flush=True)
            return loss, gradient

        with (output / f"band_{band:02d}.csv").open("w", newline="", encoding="utf-8") as history_file:
            history_writer = csv.writer(history_file)
            history_writer.writerow(("evaluation", "objective", "data", "penalty", "gradient_l2", "seconds"))
            result = minimize(
                objective, x, method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": iterations, "maxls": 20, "ftol": 1e-10, "gtol": 1e-9},
            )

        update = float(np.linalg.norm(result.x - x))
        print(f"{output.name} 完成 {cutoff:g} Hz：迭代 {result.nit}，"
              f"更新范数 {update:.3e}，{result.message}", flush=True)
        if result.nit == 0 or update == 0.:
            raise RuntimeError(f"{output.name} {cutoff:g} Hz 未更新模型，见逐次评估 CSV")
        if result.status not in (0, 1):
            raise RuntimeError(f"{output.name} {cutoff:g} Hz 优化异常：{result.message}")
        x = result.x
        models = {
            f"{vintage}_{key}": values
            for vintage, model in zip(vintages, unpack(x)) for key, values in model.items()
        }
        save_arrays(checkpoint, x=x, frequency=cutoff, iterations=result.nit,
                    objective_initial=evaluations[0][0], objective_final=result.fun,
                    evaluations=result.nfev, status=result.status,
                    message=str(result.message), update_l2=update, **models)

    models = unpack(x)
    save_arrays(output / "final.npz", **{
        f"{vintage}_{key}": values
        for vintage, model in zip(vintages, models) for key, values in model.items()
    })
    return models
