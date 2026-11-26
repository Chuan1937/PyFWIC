"""检查原始 Marmousi 压缩包和两期模型的异常位置。"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat


ARCHIVE = Path(__file__).resolve().parent / "data" / "marmousi_pcs.zip"
EXPECTED_SHA256 = "9b997faca19bfdd49471f483a9240ddfe41a5153f9759277d3a9718f7ad1aa67"
PCS_KEYS = ("phi", "cc", "sw")
DV_KEYS = ("vp", "vs", "rho")
DEPTH_OFFSET = 20


def read_model(archive, stem, keys, shape):
    with BytesIO(archive.read(f"marmousi_pcs/{stem}.mat")) as file:
        fields = loadmat(file)
    model = {}
    for key in keys:
        values = fields[key]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{stem}.mat/{key} 尺寸错误或包含非有限值")
        model[key] = values
    print(f"{stem}.mat: {shape}, {', '.join(keys)}")
    return model


def main():
    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"压缩包 SHA-256 不匹配: {digest}")
    print(f"SHA-256: {digest}")

    with ZipFile(ARCHIVE) as archive:
        pcs_bl = read_model(archive, "marmousi2_pcs_bl", PCS_KEYS, (96, 227))
        pcs_ml = read_model(archive, "marmousi2_pcs_ml", PCS_KEYS, (96, 227))
        dv_bl = read_model(archive, "marmousi_bl", DV_KEYS, (116, 227))
        dv_ml = read_model(archive, "marmousi_ml", DV_KEYS, (116, 227))

    for key in ("phi", "cc"):
        if not np.array_equal(pcs_bl[key], pcs_ml[key]):
            raise ValueError(f"{key} 的两期真值不一致")

    delta_sw = pcs_ml["sw"] - pcs_bl["sw"]
    mask = delta_sw != 0
    decreasing = np.isclose(delta_sw, -0.2)
    increasing = np.isclose(delta_sw, 0.2)
    if mask.sum() != 57 or decreasing.sum() != 49 or increasing.sum() != 8:
        raise ValueError("含水饱和度变化的点数或幅值不符")
    print("ΔSw: 57 点变化，其中 49 点约为 -0.20，8 点约为 +0.20")

    for key in DV_KEYS:
        dv_mask = dv_ml[key] != dv_bl[key]
        if dv_mask[:DEPTH_OFFSET].any() or not np.array_equal(
            dv_mask[DEPTH_OFFSET:], mask
        ):
            raise ValueError(f"{key} 的变化掩膜未与 PCS 对齐")
    print(f"弹性变化掩膜与 PCS 一致，深度索引偏移 {DEPTH_OFFSET} 行")


if __name__ == "__main__":
    main()
