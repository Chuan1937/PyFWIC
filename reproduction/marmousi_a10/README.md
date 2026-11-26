# NVIDIA A10 上的 PyFWI Marmousi 时移反演

服务器从仓库根目录运行现成脚本。脚本读取随分支提交的原始数据，完成两期正演、基线 FWI、同时式 DV 和 PCS 反演，并输出图、模型和误差表。一张 A10 即可启动一套作业。两张卡可分别调度独立的作业。

## 运行

在服务器准备 Python 3.10–3.12 的环境并安装依赖。已具备兼容依赖的环境可直接运行。运行节点须有 NVIDIA 驱动和可访问的 NVIDIA OpenCL ICD。

```bash
python3 -m venv "$PWD/.venv-marmousi"
export PYFWI_PYTHON="$PWD/.venv-marmousi/bin/python"
"$PYFWI_PYTHON" -m pip install -r reproduction/marmousi_a10/requirements.txt
```

在已分配 GPU 的计算节点先跑单炮短测，再跑 7 炮全频段。将结果放在服务器的作业盘。默认选择 OpenCL 可见的第一张 A10，可用 `CUDA_VISIBLE_DEVICES` 限定可见的卡，也可指定 `--device 平台号:设备号`。经内核与数据传输优化后，默认单进程（`--workers 1`）已能让 A10 满载，完整运行不再需要按炮开进程；用 `taskset` 绑定少量 CPU 核并保持线程数为 1 即可。

```bash
export PYFWI_PYTHON="${PYFWI_PYTHON:-$PWD/.venv-marmousi/bin/python}"
export CUDA_VISIBLE_DEVICES=0          # 选中空闲的 A10；另一套作业改用 1
export NUMEXPR_NUM_THREADS=1
RUN_ROOT="${SCRATCH:-$PWD/runs}"

taskset -c 0-1 reproduction/marmousi_a10/run_a10.sh \
  --profile smoke --out "$RUN_ROOT/marmousi-smoke"
taskset -c 0-3 reproduction/marmousi_a10/run_a10.sh \
  --profile full --out "$RUN_ROOT/marmousi-full"
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

### 并发与 CPU 绑定

经内核与数据传输优化后，默认的单进程路径已能把 A10 压到约 100% 利用率和约 145 W 功耗，因此完整运行使用默认的 `--workers 1`，不必再按炮开进程。`taskset` 仍建议保留：短测绑 `0-1`，完整运行绑 `0-3`，`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 保持为 1。前三个 `run_a10.sh` 已默认设为 1，`NUMEXPR_NUM_THREADS` 由上面的命令导出。

- `--workers 1`（默认）走单进程路径，结果与旧实现逐位一致：正演与伴随的最大绝对差为 0.0，物理、滤波、上下界、频段、迭代次数和归一化均未改动。
- `--workers 0` 仍表示每炮一个工作进程：短测 1 个，完整 7 炮 7 个（上限即炮数）。该模式保留给其他设备或负载；在这张 A10 上并未更快。实测 7 炮每次评估的正演/伴随耗时为 `W=1` 0.695 s / 1.16 s，`W=4` 0.558 s / 1.26 s，`W=7` 0.564 s / 1.28 s，净评估时间基本持平（`W=1` 约 1.85 s，`W=4` 约 1.82 s）。
- 多进程版本按炮在进程间累加梯度，用 float64 求和，与单进程只差最后几位（相对约 2e-6）。这只属于调度差异，不改变反演设置。
- 父进程与 worker 共用一张 A10，用 `CUDA_VISIBLE_DEVICES` 选定空闲卡；两张卡上可各跑一套作业，互不抢占。
- `run.json` 会记录 `workers` 字段。它位于 `config` 之外，因此不参与 `--resume` 的配置一致性判定。

## 耗时估算

单张 A10 的实测总时长为 **约 12 分钟**（7 炮、6 频段、DV/PCS 四个阶段，共 275 次目标函数评估、约 2,900 次单炮单期正演＋伴随；当前代码 11 分 47 秒，其中优化器内约 636 秒）。首次排队、环境安装和图形输出另计。早期文档里的 1–6 小时（首次申请 8 小时）是按本地 CPU OpenCL 短测外推的保守上界，不是 A10 实测；7 炮共用一次评估的主机开销，因此实际远低于按单炮逐次累加的估计。

完整运行有 7 炮、6 个频段、两个单期基线阶段和两个双期时移阶段。若每个阶段每频段约 11–20 次目标函数评估，计算量相当于约 2,800–5,000 次单炮单期的正演＋伴随。服务器短测后，取某个 `band_XX.csv` 中 `(最后一行 seconds − 第一行 seconds) / (评估次数 − 1)` 得到单次评估秒数，再乘以预计的评估次数即可。7 炮在一次评估内完成，不要再乘炮数。实际线搜索次数会改变总耗时。

本分支加入了内核与数据传输优化：整炮残差一次上传、在 `Adj_injSrc` 内按时间索引；速度/应力更新成对融合（`update_vel`、`update_tau`）；地表 `injSrc`/`Adj_injSrc` 改用接收点的一维网格；波场检查点存入设备缓冲，用异步设备间拷贝代替阻塞的主机拷贝；不随步变化的内核参数只绑定一次，时间循环内直接入队。7 炮单次评估的墙钟时间由约 3.5 s 降到约 1.64 s（约 2.1 倍），单进程即可让 A10 保持约 100% 利用率和约 145 W 功耗（峰值约 156 W，标称上限 150 W）。这些改动只涉及调度和数据搬运，正演与伴随与原实现逐位一致（最大绝对差 0.0），物理、滤波、上下界、频段、迭代次数和归一化均未改动。完整算例的实测墙钟时间为 11 分 47 秒，其中优化器内约 636 秒，其余为启动、观测正演和出图。

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

本分支在 `results/full_a10/` 附带一次完整运行的摘要、最终模型、观测、指标和图片，运行条件与主要误差见其中的 `RESULTS.md`。

参考 [PyFWI 软件论文 Fig. 3–5](https://doi.org/10.1016/j.softx.2023.101384) 和 [含水饱和度监测论文表 1、Eq. (9)–(15)、Fig. 9–14](https://doi.org/10.1111/1365-2478.13363)。

## 方法论文的第二套实验

### Marmousi 补充图

已用原有 A10 两期模型生成 [PCS 基线、DV 间接饱和度、PCS 直接饱和度及剖面](results/full_a10/paper2/README.md)。重新生成只需 CPU，输入目录应当含 `models.npz`、`dv_sim/final.npz`、`pcs_sim/final.npz` 及 `pcs_baseline/final.npz`。

```bash
PYTHONPATH="$PWD/src" "${PYFWI_PYTHON:-python3}" reproduction/marmousi_a10/indirect.py \
  --run reproduction/marmousi_a10/results/full_a10 \
  --out "$RUN_ROOT/marmousi-paper2"
```

### 简单层状模型（Fig. 2–8）

从仓库根目录执行，使用与 Marmousi 相同的依赖环境。先运行单炮、10 Hz 短测，再运行七炮、六频段的正式实验。

```bash
RUN_ROOT="${SCRATCH:-$PWD/runs}"
reproduction/marmousi_a10/run_a10.sh --case layered --profile smoke --out "$RUN_ROOT/layer-smoke"
sbatch reproduction/marmousi_a10/submit_a10.sbatch \
  --case layered --profile full --out "$RUN_ROOT/layer-full"
```

三层参数、层界、半圆掩膜和 25% 相对饱和度增量沿用仓库 `hu_laminar`。采用 100 × 100 网格、10 m 网格间距并在距左端 0.5 km 处取中心剖面。七个 30 Hz 爆炸震源位于地表附近，同时记录地表和两侧井中的无噪声压力类炮集。计算约定使用 0.8 s 记录、1 ms 外部采样和原脚本的 PML 设置。模式使用单进程，偏移方向按保存的炮检坐标解释。

输出 `paper_figures/` 包含两期和初始 PCS 模型、PCS 基线影像及中心剖面、时移弹性变化、三种饱和度对照和误差表。三种对照为 DV 两期自由拟合 PCS、DV 监测期仅拟合饱和度，以及 PCS 直接反演。

可用 `--tikhonov-x` 做横向平滑权重试验。该选项惩罚归一化活跃模型的横向一阶差分，默认权重为零。要核对论文报告的百分比，还需校准该项权重与论文式 (6) 的梯度缩放，并对齐误差定义。

层状算例已在 A10 上完成单炮短测和七炮六频段正式实验，总墙钟 12 分 37 秒（280 次评估，优化器内约 678 秒）。结果保存在 `results/layer_full/`，包括 `paper_figures/` 和 `RESULTS.md`。储层内饱和度相对 L2：PCS 直接反演 **0.247**，DV 间接（仅 Sw）0.965，DV 间接（全部 PCS 参数）0.942；储层外 RMS 分别为 0.0081、0.0303、0.0102。混合观测的左右井记录按坐标顺序输出，并限制右井内核只访问本井实际道数。
