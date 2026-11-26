#!/usr/bin/env python3
"""合并双卡拆分的 DV/PCS 分支结果，生成与单进程完全相同的结果目录。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dv", type=Path, required=True, help="DV 分支结果目录")
    parser.add_argument("--pcs", type=Path, required=True, help="PCS 分支结果目录")
    parser.add_argument("--out", type=Path, required=True, help="合并后的结果目录")
    return parser.parse_args()


def read_config(directory):
    with (directory / "run.json").open(encoding="utf-8") as file:
        return json.load(file)["config"]


def load_final(path, keys):
    with np.load(path) as arrays:
        return [{key: arrays[f"{vintage}_{key}"] for key in keys}
                for vintage in ("bl", "ml")]


def copy_tree(source, target):
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def main():
    args = arguments()
    dv_dir, pcs_dir = args.dv.resolve(), args.pcs.resolve()
    output = args.out.resolve()
    if output in (dv_dir, pcs_dir):
        raise ValueError("合并目录必须与两个分支目录不同")

    dv_config, pcs_config = read_config(dv_dir), read_config(pcs_dir)
    strip = lambda config: {key: value for key, value in config.items() if key != "branch"}
    if strip(dv_config) != strip(pcs_config):
        raise ValueError("两个分支的计算配置不一致，无法合并")
    case = dv_config.get("case", "marmousi")

    from case import DV_KEYS, PCS_KEYS, load_models
    from results import evaluate

    dv_models = load_final(dv_dir / "dv_sim" / "final.npz", DV_KEYS)
    pcs_models = load_final(pcs_dir / "pcs_sim" / "final.npz", PCS_KEYS)

    output.mkdir(parents=True, exist_ok=True)
    copy_tree(dv_dir, output)
    for name in ("pcs_baseline", "pcs_sim"):
        shutil.copytree(pcs_dir / name, output / name, dirs_exist_ok=True)

    if case == "layered":
        from indirect import paper_figures
        paper_figures(output, output / "paper_figures", dh=10., layered=True)
    else:
        pcs_truth, elastic_truth, _ = load_models("pcs")
        evaluate(output, pcs_truth, elastic_truth, dv_models, pcs_models)

    manifest = {"dv": str(dv_dir), "pcs": str(pcs_dir), "case": case,
                "config": strip(dv_config)}
    (output / "combine.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"合并完成，结果目录 {output}", flush=True)


if __name__ == "__main__":
    main()
