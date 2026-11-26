# A10 完整运行结果

本目录保存一次完整 `--profile full` 运行的输出，用于对照论文与检查复现。

## 运行条件

| 项 | 值 |
| --- | --- |
| 设备 | NVIDIA A10（单卡，运行时空闲卡） |
| 进程 | 单进程，`--workers 1`，`taskset -c 0-3` |
| 网格 | 96 × 227，5 m 网格，四周 20 点 PML，四阶差分 |
| 观测 | 7 炮，223 道/炮，30 Hz Ricker，记录约 0.55 s |
| 采样 | 外部 551 点（1 ms），内部 1101 步（0.5 ms） |
| 频段 | 10、20、25、35、40、55 Hz，每段最多 10 次 L-BFGS-B 迭代 |
| 目标函数评估 | 275 次（71 + 70 + 68 + 66） |
| 优化器内耗时 | 712 s |
| 总墙钟时间 | 13 分 20 秒（含启动、观测正演和出图） |

`run.json` 中的 `commit` 记录的是启动时的 HEAD（`2efc656`）。本目录的结果由该提交之上的性能优化工作区产生；之后的提交只改变内核调度与数据搬运，正演和伴随已验证逐位一致（最大绝对差 0.0），因此用当前分支代码可得到相同的数值结果。

用当前分支（含内核参数预绑定）按同一配置重跑：275 次评估，优化器内 636 秒，总墙钟 **11 分 47 秒**；`metrics.csv` 与本目录逐行相同，首个目标函数值也完全一致。

用 `run_split.sh` 把 DV/PCS 拆到两张 A10 并行：从分支启动到合并在 **6 分 03 秒** 完成，`metrics.csv` 同样与本目录逐行相同。

## 主要指标

储层内相对 L2 误差（`metrics.csv` 的 `relative_l2_error`）：

| 参数 | DV 左 / 右 | PCS 左 / 右 |
| --- | --- | --- |
| ΔVp | 0.268 / 0.357 | 0.700 / 0.266 |
| ΔVs | 0.825 / 0.827 | 0.677 / 0.294 |
| Δρ | 0.685 / 0.568 | 0.584 / 0.277 |
| ΔSw | — | 0.612 / 0.247 |

储层外 ΔSw RMS 为 0.0047。左储层（Sw 上升）与右储层（Sw 下降）的位置与符号与真值一致，左储层幅值偏低。DV 间接饱和度的对照见 [方法论文补充结果](paper2/README.md)。

## 文件

- `run.json`：数据来源、配置、设备、采样与版本
- `metrics.csv`、`reservoir_profiles.csv`：误差表与储层剖面
- `models.npz`、`observations.npz`、`time_lapse_changes.npz`：模型、观测炮集、时移差值数组
- `*/final.npz`：各阶段最终模型；`*/band_*.csv`：每次目标函数评估的历史
- `*.png`：真值、差值、炮集、含水饱和度与储层剖面图

各频段的逐次检查点 `*/band_*.npz` 体积较大（约 30 MB），未纳入仓库；运行后会生成。

## 复现命令

```bash
export PYFWI_PYTHON=/path/to/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
taskset -c 0-3 reproduction/marmousi_a10/run_a10.sh --profile full --out "$PWD/runs/full"
```
