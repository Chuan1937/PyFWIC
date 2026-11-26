# 在 NVIDIA A10 上复现 PyFWI Marmousi 时移反演

从仓库根目录执行。请用当前仓库的 PyFWI/OpenCL 实现，在服务器上依次完成数据核验、两期正演、同时式 DV 时移反演、同时式 PCS 时移反演和结果对比。先用 **一张 A10** 跑通。两张卡可用于独立作业，分别记录每个进程实际选中的设备。

## 数据与论文

| 文件 | 参数键 | 原始数组尺寸 |
| --- | --- | --- |
| `marmousi2_pcs_bl.mat`、`marmousi2_pcs_ml.mat` | `phi`、`cc`、`sw` | 96 × 227 |
| `marmousi_bl.mat`、`marmousi_ml.mat` | `vp`、`vs`、`rho` | 116 × 227 |

`data/marmousi_pcs.zip` 是作者提供的原始压缩包。BL 为基线期，ML 为监测期，数组按 `[z, x]` 读入。包内是合成模型真值，观测炮集由两期模型分别正演生成。原始速度数值量级为 m/s，密度为 g/cm³，PCS 参数为小数比例。全程约定时移量为 **ML − BL**。

- [PyFWI 软件论文（SoftwareX，2023）](https://doi.org/10.1016/j.softx.2023.101384) 第 3 节及 Fig. 3–5 给出真值、炮集和 DV／PCS 两种参数化下的弹性变化。
- [含水饱和度监测论文（Geophysical Prospecting，2023）](https://doi.org/10.1111/1365-2478.13363) 的表 1、Fig. 9–14 给出岩石物理常数、Marmousi 模型、基线 FWI 结果、时移反演和穿过两个储层的剖面。

## 1. 检查运行环境与数据

在分配到 A10 的节点执行以下命令。Slurm 分区名、GPU 资源参数和环境模块按服务器实际配置填写。

```bash
nvidia-smi -L
clinfo -l
python --version
python -m pip show numpy scipy pyopencl
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from pathlib import Path
import pyopencl as cl
import PyFWI

print("PyFWI", Path(PyFWI.__file__).resolve())
for p_i, platform in enumerate(cl.get_platforms()):
    for d_i, device in enumerate(platform.get_devices()):
        print(f"{p_i}:{d_i}", platform.name, device.name, device.global_mem_size)
PY
```

根据输出设置 `PYOPENCL_CTX` 为目标 A10 的 `平台索引:设备索引`。实际构建 `WavePropagator` 后记录 `ctx.devices` 和队列设备名。显卡驱动可见、OpenCL 可枚举，以及三个 OpenCL 内核成功编译是不同的检查。当前 `wave_propagation.py` 传入 `inpa['device']` 时会改写 `PYOPENCL_CTX`，用环境变量选卡时可以省略该键。保留环境和依赖版本、GPU 设备、作业号及运行时间。仓库 `requirements.txt` 含旧版本约束，先确认 Python 与依赖组合并做导入测试。若遇到 SciPy 或 NumPy 导入兼容性问题，记录环境并做最小兼容修正。

```bash
python reproduction/marmousi_a10/check_data.py
unzip -n reproduction/marmousi_a10/data/marmousi_pcs.zip -d /path/to/workdir/input
```

以核验结果作为输入基准。PCS 的 `phi` 和 `cc` 两期一致，`sw` 恰有 57 点变化（49 点约为 −0.20，8 点约为 +0.20）。三种原始弹性参数的变化掩膜与 PCS 的变化掩膜相差 20 个深度采样点，因此可从 `dv[20:, :]` 开始做对齐核验。**掩膜对齐只证明异常位置一致。** 还需用两期模型及 PCS→DV 转换逐点核对背景、异常幅值和单位。原始包与前期汇报的 94 × 226 算例有尺寸差异。若为比较论文图另做裁剪，应记录具体切片、坐标原点及每块异常是否保留，并保留原始网格的结果。

## 2. 正演与小规模验证

- 参考软件论文设 7 个各向同性爆炸震源、30 Hz Ricker 子波、PML 和地表观测，使用压力类记录并记录 PyFWI 实际输出分量及符号约定。复现初始配置可用 `dh=5 m`、外部采样间隔 `dt=1 ms`、记录时长约 `0.55 s`、`npml=20`。后四项是组会算例的起始配置。用实际数组尺寸生成炮检坐标，记录采样点数、内部步长和边界设置。
- 先跑一个炮的两期正演，检查数据尺寸、有限值、非零能量及 ML−BL 的差值。用相同绘图归一化对照基线和监测炮集，并单独标注差值放大倍数。通过后生成完整的 7 炮数据，保存炮检坐标、波形数组、参数和中间炮三联图。
- 从同一个平滑初始模型开始，先用 10 Hz 低通及 2 次迭代验证 DV 和 PCS 的正演、伴随和目标函数路径。记录初始及最终目标函数、梯度范数和实际参数更新；若迭代失败或未更新，保留日志并修复后再提交正式作业。

## 3. 完整 DV 与 PCS 反演

**DV 对照组。** 用 `TimeLapse` 的同时式方法 `tl_method='sim'`，保存基线 FWI 的输出及后续时移结果。基线和监测期采用同一平滑初值或同一基线 FWI 结果作为时移初值，记录平滑参数和实际优化的参数块。完整多尺度按 **10、20、25、35、40、55 Hz** 低通截止频率运行，每段最多 10 次迭代，记录优化器实际迭代数和停止原因。若复现全部三个 DV 分量，核对仓库按 1 起始且上界排除的块索引，验证全部三个分量实际更新，科学对比时使用同一观测炮集。

**PCS 组。** 使用基线期 `phi`、`cc`、`sw` 的平滑初始模型，先做基线 PCS FWI。时移阶段固定两期共用的 `phi`、`cc`，只更新 `sw`，检查迭代值位于 [0,1]。复用 `fwi_tools.py` 的 PCS 字典／向量转换，以及 `rock_physics.py` 和 `grad_switcher.py` 的 Gassmann 转换与链式梯度，经 `TimeLapse(..., param_functions=...)` 接入。仓库里的块顺序为 `phi, cc, sw`，仅更新第三块时核对 `n_params=1, k_0=3, k_end=4` 及实际更新。当前同时式实现的 L-BFGS-B 传入 `bounds=None`，如需要物理上下界，显式设置相应参数块的边界并记录。基线 FWI 和时移阶段的参数范围分别检查，因为 `TimeLapse.__call__` 的同时式入口将一组参数块范围传给两个阶段；需要区分时可单独调用基线 FWI，再用基线结果启动同时式反演。

- 按方法论文表 1 设置岩石物理常数，统一密度、模量、速度的数值单位与换算。核对标准各向同性速度关系 `Vp²=(Ksat+4μ/3)/ρ` 与当前 `pcs2dv_gassmann` 中 `0.75 * mu_d` 系数的差异，并检查转换和梯度实现是否一致。先做小模型中心差分方向导数检查，保留各步长的数值结果。必要时把有依据的修正单独提交，并给出修正前后对照。
- 逐期将 PCS 真值转换到弹性域，报告与对齐后作者弹性数组的差异。DV 和 PCS 的正式对比应使用同一套观测、网格和几何。若两套真值在弹性域并不一致，先查岩石物理参数、单位和裁剪。若仍有差异，可用同一套 PCS→DV 生成的真值和炮集作为两种参数化的共同对比基准，并标明与作者弹性包的偏差。作者弹性模型生成的 DV 结果可单独留作对照。
- PCS 弹性差值用 `R(PCS_monitor) - R(PCS_baseline)` 计算，其中 `R` 是岩石物理转换。保存实际估计的 `ΔSw` 和两期模型。报告同时式模型差惩罚、正则化和参数缩放的实际设置。当前 `tl_fwi.py` 中的 `gamma` 是硬编码值，论文、组会与代码配置有差异时注明采用哪一套。

## 4. 交付与验收

- 提交可命令行运行的复现脚本、配置、服务器作业示例和中文说明，标注运行命令、代码提交号及依赖版本。数据和计算输出路径可配置。
- 保存两期真值、初始模型、两期反演结果、各频段检查点、目标函数或数据残差历史和对应运行日志。大体积输出放在服务器作业目录，记录其路径。
- 交付 Fig. 3 样式的真值图、Fig. 4 样式的正演炮集图，以及 Fig. 5 样式的 DV／PCS 弹性变化对比。另给出 PCS 的 `ΔSw` 真值／估计／误差图及穿过两处储层的剖面。同一参数的对比图统一坐标、单位和色标。
- 对 `ΔVp`、`ΔVs`、`Δρ` 和 `ΔSw` 报告全域及两个储层内的相对 L2 误差，给出储层外异常幅值或能量。检查两处异常的位置、符号与幅值。论文的 11.7% 改进是其算例报告值，报告本次实测结果和差异。
