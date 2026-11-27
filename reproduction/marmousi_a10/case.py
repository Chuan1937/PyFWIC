"""Marmousi 输入、获取系统与论文式 (9)–(15) 的 PCS 转换。"""

import json
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import gaussian_filter

from PyFWI import rock_physics as rp

from check_data import ARCHIVE, DEPTH_OFFSET, DV_KEYS, PCS_KEYS, read_model


# SI units, Geophysical Prospecting (2023), Table 1.
K_QUARTZ, K_CLAY, K_WATER, K_GAS = 37e9, 21e9, 2.25e9, 0.04e9
G_QUARTZ, G_CLAY = 44e9, 10e9
RHO_QUARTZ, RHO_CLAY, RHO_WATER, RHO_GAS = 2650., 2550., 1000., 100.
CS = 20.
DH, DT, FDOM = 5., 0.001, 30.
DV_SCALE = (2500., 1250., 2000.)
PCS_SCALE = (1., 1., 1.)
DV_LIMITS = ((1400., 3600.), (500., 1950.), (1400., 2800.))
PCS_LIMITS = ((0.06, 0.65), (0., 0.5), (0., 1.))


def load_models(truth):
    with ZipFile(ARCHIVE) as archive:
        pcs = [
            read_model(archive, f"marmousi2_pcs_{vintage}", PCS_KEYS, (96, 227))
            for vintage in ("bl", "ml")
        ]
        author = [
            read_model(archive, f"marmousi_{vintage}", DV_KEYS, (116, 227))
            for vintage in ("bl", "ml")
        ]
    author = [
        {key: np.ascontiguousarray(array[DEPTH_OFFSET:] * (1000 if key == "rho" else 1))
         for key, array in model.items()}
        for model in author
    ]
    converted = [pcs_to_dv(model) for model in pcs]
    differences = {
        vintage: {
            key: float(np.linalg.norm(model[key] - original[key]) / np.linalg.norm(original[key]))
            for key in DV_KEYS
        }
        for vintage, model, original in zip(("bl", "ml"), converted, author)
    }
    return pcs, converted if truth == "pcs" else author, differences


def smooth_model(model, sigma, limits):
    return {
        key: np.clip(gaussian_filter(values, sigma), *bound)
        for (key, values), bound in zip(model.items(), limits)
    }


def pcs_to_dv(model, gradient=None):
    """Convert PCS in SI, or apply the transpose of its pointwise Jacobian."""
    phi, clay, sw = (np.asarray(model[key], dtype=np.float64) for key in PCS_KEYS)
    solid_bulk = rp.vrh(K_CLAY, K_QUARTZ, clay, "VRH")
    solid_shear = rp.vrh(G_CLAY, G_QUARTZ, clay, "VRH")
    fluid_bulk = rp.weighted_average(K_WATER, K_GAS, sw)
    solid_density = rp.weighted_average(RHO_CLAY, RHO_QUARTZ, clay)
    fluid_density = rp.weighted_average(RHO_WATER, RHO_GAS, sw)
    dry_bulk, shear = rp.drained_moduli(phi, solid_bulk, solid_shear, CS)
    bulk, _, _ = rp.biot_gassmann(phi, fluid_bulk, solid_bulk, dry_bulk)
    rho = rp.weighted_average(fluid_density, solid_density, phi)
    vp = np.sqrt((bulk + 4. / 3. * shear) / rho)
    vs = np.sqrt(shear / rho)
    if gradient is None:
        return dict(zip(DV_KEYS, (vp, vs, rho)))

    # K = K_D + A² / D, algebraically equivalent to equations (13)–(14).
    dry_ratio = (1 - phi) / (1 + CS * phi)
    a = 1 - dry_ratio
    denom = phi / fluid_bulk + (a - phi) / solid_bulk
    a_phi = (1 + CS) / (1 + CS * phi) ** 2
    shear_phi = -solid_shear * (1 + 1.5 * CS) / (1 + 1.5 * CS * phi) ** 2
    denom_phi = 1 / fluid_bulk + (a_phi - 1) / solid_bulk
    bulk_phi = -solid_bulk * a_phi + 2 * a * a_phi / denom - a**2 * denom_phi / denom**2

    def hill_derivative(clay_modulus, quartz_modulus):
        reciprocal_difference = 1 / clay_modulus - 1 / quartz_modulus
        inverse_reuss = 1 / quartz_modulus + clay * reciprocal_difference
        return 0.5 * (clay_modulus - quartz_modulus - reciprocal_difference / inverse_reuss**2)

    solid_bulk_clay = hill_derivative(K_CLAY, K_QUARTZ)
    bulk_clay = solid_bulk_clay * (dry_ratio + a**2 * (a - phi) / (denom * solid_bulk)**2)
    shear_clay = shear / solid_shear * hill_derivative(G_CLAY, G_QUARTZ)
    bulk_sw = a**2 * phi * (K_WATER - K_GAS) / (denom * fluid_bulk)**2

    grad_vp, grad_vs, grad_rho = (gradient[key] for key in DV_KEYS)
    grad_bulk = grad_vp / (2 * rho * vp)
    grad_shear = 4. / 3. * grad_bulk + grad_vs / (2 * rho * vs)
    grad_density = grad_rho - (grad_vp * vp + grad_vs * vs) / (2 * rho)
    return {
        "phi": grad_bulk * bulk_phi + grad_shear * shear_phi
        + grad_density * (fluid_density - solid_density),
        "cc": grad_bulk * bulk_clay + grad_shear * shear_clay
        + grad_density * (1 - phi) * (RHO_CLAY - RHO_QUARTZ),
        "sw": grad_bulk * bulk_sw + grad_density * phi * (RHO_WATER - RHO_GAS),
    }


def check_pcs_derivative(model):
    rng = np.random.default_rng(1729)
    weights = {key: rng.standard_normal(model["phi"].shape) for key in DV_KEYS}
    gradient = pcs_to_dv(model, weights)
    errors = {}
    for key in PCS_KEYS:
        step = 1e-5
        direction = rng.standard_normal(model[key].shape)
        plus = {**model, key: model[key].astype(np.float64) + step * direction}
        minus = {**model, key: model[key].astype(np.float64) - step * direction}
        upper, lower = pcs_to_dv(plus), pcs_to_dv(minus)
        fd = sum(np.sum(weights[k] * (upper[k] - lower[k])) for k in DV_KEYS) / (2 * step)
        predicted = np.sum(gradient[key] * direction)
        errors[key] = float(abs(fd - predicted) / max(abs(fd), abs(predicted), 1e-12))
    if max(errors.values()) > 1e-5:
        raise ValueError(f"PCS → DV 链式梯度方向导数检查失败: {errors}")
    return errors


def save_arrays(path, **arrays):
    """Replace a checkpoint only after its full contents have been written."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
