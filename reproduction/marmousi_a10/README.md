# NVIDIA A10 上的 PyFWI Marmousi 时移反演

服务器从仓库根目录运行现成脚本。脚本读取随分支提交的原始数据，完成两期正演、基线 FWI、同时式 DV 和 PCS 反演，并输出图、模型和误差表。一张 A10 即可启动一套作业。两张卡可分别调度独立的作业。

## 运行

在服务器准备 Python 3.10–3.12 的环境并安装依赖。已具备兼容依赖的环境可直接运行。运行节点须有 NVIDIA 驱动和可访问的 NVIDIA OpenCL ICD。

```bash
python3 -m venv "$PWD/.venv-marmousi"
export PYFWI_PYTHON="$PWD/.venv-marmousi/bin/python"
"$PYFWI_PYTHON" -m pip install -r reproduction/marmousi_a10/requirements.txt
```

在已分配 GPU 的计算节点先跑单炮短测，再跑 7 炮全频段。将结果放在服务器的作业盘。默认选择 OpenCL 可见的第一张 A10，也可指定 `--device 平台号:设备号`。

```bash
RUN_ROOT="${SCRATCH:-$PWD/runs}"
reproduction/marmousi_a10/run_a10.sh --profile smoke --out "$RUN_ROOT/marmousi-smoke"
reproduction/marmousi_a10/run_a10.sh --profile full --out "$RUN_ROOT/marmousi-full"
```

Slurm 环境可以从仓库根目录直接提交。分区名、A10 资源名与时限按集群配置覆盖。提交前设好 `PYFWI_PYTHON`，以及可写的输出目录。

```bash
sbatch reproduction/marmousi_a10/submit_a10.sbatch \
  --profile full --out "$RUN_ROOT/marmousi-full"
```

- `--forward-only` 只做模型检查和正演。以后可以用同一输出目录加 `--resume` 继续反演。
- `--resume` 复用已有炮集和已经完整保存的频段，要求提交号和计算配置一致。运行中退出的频段会重新计算。每个输出目录同时只提交一个作业。
- `--truth author-dv` 使用原始弹性文件单独运行 DV 分支，用不同的输出目录。默认 `--truth pcs` 运行 DV 和 PCS 的同炮集对照。
- `--allow-non-a10` 用于其他 OpenCL 设备上的小规模测试。正常服务器作业会检查实际选中的设备名。
- 如需先检查数据，可执行 `"$PYFWI_PYTHON" reproduction/marmousi_a10/check_data.py`。主程序会自动执行该检查，无需解压。

## 耗时估算

单张 A10 的预算先按短测约 0.5–2 分钟、完整 DV + PCS 约 **1–6 小时**，首次可申请 8 小时。排队与首次环境安装另计。这个范围是根据求解次数和本地 CPU OpenCL 短测推算，A10 的实际速度以后续短测为准。

完整运行有 7 炮、6 个频段、两个单期基线阶段和两个双期时移阶段。若每个阶段每频段约 11–20 次目标函数评估，计算量相当于约 2,800–5,000 次单炮单期的正演＋伴随。服务器短测后，取 `dv_baseline/band_01.csv` 最后一行的 `seconds / evaluation`，乘以 2,800–5,000 即可估计正式计算的秒数。实际线搜索次数会改变总耗时。

## 模型来源与观测配置

| 压缩包内文件 | 两期参数 | 原始数组尺寸 `[z, x]` |
| --- | --- | --- |
| `marmousi2_pcs_bl.mat`、`marmousi2_pcs_ml.mat` | `phi`、`cc`、`sw` | 96 × 227 |
| `marmousi_bl.mat`、`marmousi_ml.mat` | `vp`、`vs`、`rho` | 116 × 227 |

`data/marmousi_pcs.zip` 保留作者提供的文件，BL 为基线期，ML 为监测期，时移差统一为 **ML − BL**。两期 PCS 模型只有 57 个 `sw` 点改变。49 个下降 0.20，8 个上升 0.20。弹性变化掩膜下移 20 个深度采样点，读取弹性文件时用 `[20:, :]` 与 PCS 对齐，密度由 g/cm³ 转为 kg/m³。变换后与作者弹性真值的逐期相对 L2 差异写入 `run.json`。按论文方法和表 1 参数转换的基线模型与作者弹性文件在 Vp、Vs、密度上的差异分别约为 21%、31%、13%。因此默认用 **同一套 PCS→DV 真值**合成 DV 和 PCS 共用的观测，从而保证对照使用相同的地震数据。使用原始弹性文件的 DV 结果单独保存。

以软件论文第 3 节和组会正演设置为依据，全算例采用 7 个地表附近爆炸源、30 Hz Ricker 子波、5 m 网格、约 0.55 s 记录时长、1 ms 外部采样、四阶空间差分和四周 20 点 PML。压力类记录使用 PyFWI 的 `(τxx + τzz)/2`。实际炮检点、时步、采样数和设备信息保存在输出中。全算例的 Butterworth 低通截止频率为 **10、20、25、35、40、55 Hz**，每段最多 10 次 L-BFGS-B 迭代。短测采用网格中央的单炮、10 Hz、最多 2 次迭代。

## 反演实现与验收

- 正演与伴随调用仓库的 PyFWI OpenCL 传播器。入口脚本用 SciPy L-BFGS-B 实现基线及同时式目标函数，每个频段用上一段结果初始化。数据项按该频段观测总能量归一化。两期模型差惩罚基于无量纲的参数差，`--coupling` 可调。实际设置、上下界和每次目标函数评估都可追溯。
- DV 在基线和时移阶段更新三种弹性参数。PCS 在基线阶段更新三种岩石物理参数，时移阶段固定共同的 `phi`、`cc`，只更新两期 `sw`，保留 [0, 1] 上下界。PCS 模型用方法论文 Eq. (9)–(15) 的 VRH、Pride 和 Gassmann 公式。纵波速度采用 Eq. (15) 的 `4/3` 系数。运行时先检查 PCS 转换的链式梯度，再跑 OpenCL 正演与伴随。
- 上游四阶 OpenCL 核心仍写有零权重的宽模板项。本分支让这些项读取网格内的地址，并修正模板核的上界。此修正在本地 POCL 上解决了正演出现 NaN 的问题。此算例的几何、参数约束及归一化均已记录，论文报告的 11.7% 改进用于结果讨论。
- 短测验收看 BL、ML 及差值记录有限且非零，并检查 DV、PCS 的基线和时移阶段都有实际模型更新。正式计算用两个储层的时移异常位置、符号和幅值验收，同时报告储层外伪影。

| 输出 | 内容 |
| --- | --- |
| `run.json`、`models.npz`、`observations.npz` | 数据来源、参数、版本、模型、实测设备、炮检点和波形 |
| `dv_baseline/`、`dv_sim/`、`pcs_baseline/`、`pcs_sim/` | 每频段检查点、评估历史、目标函数和最终模型 |
| `true_elastic*.png`、`shot_gathers.png` | 真值、两期差值、中间炮两期炮集和放大 5 倍的差值炮集 |
| `elastic_change_comparison.png`、`saturation_change.png` | 真值和估计的弹性差值、PCS 含水饱和度差值及误差 |
| `metrics.csv`、`reservoir_profiles.*`、`time_lapse_changes.npz` | 相对 L2、储层外 RMS、穿过两处储层的剖面及差值数组 |

参考 [PyFWI 软件论文 Fig. 3–5](https://doi.org/10.1016/j.softx.2023.101384) 和 [含水饱和度监测论文表 1、Eq. (9)–(15)、Fig. 9–14](https://doi.org/10.1111/1365-2478.13363)。
