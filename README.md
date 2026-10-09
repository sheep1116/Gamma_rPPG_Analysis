# Gamma_rPPG_Analysis

研究逐通道 Gamma 变换对经典 rPPG 心率估计的影响，包含正式实验、固定 ROI 控制实验、结果统计和 ROI 响应机制分析。原 `ROI_gamma_mechanism_analysis` 已整合进本仓库，不再需要在两个项目之间切换。

保留原来的 Gamma 参数、响应分类、CSV 列定义和结果目录。现有 `GammaBrightness_rPPG_v5` 结果可以继续使用；仓库版本升级不要求重新提取信号。

## 项目结构

```text
gamma_rppg/
  common/                 配置、结果路径、原子 CSV 存储、共享统计工具
  datasets/               PURE 六类运动/AVI/帧目录适配
  signal/                 Gamma LUT、ROI RGB/luma 特征、信号提取
  experiments/            pre-ROI、post-ROI、算法注册、快速测试
  analysis/
    summary.py            汇总、完整配对检验、质量控制
    sensitivity.py        响应分类、基线 ROI 特征与敏感性分析
    roi_mechanism/         ROI 机制统计、绘图、报告与 CLI
  diagnostics/            PURE AVI 完整性与数据结构检查
  figures/                典型案例直方图、RGB 时序和 BVP 频谱
configs/                  实验配置、公共 datasets.yaml、ROI 机制配置
scripts/                  Linux 运行入口；figures/ 保存 Node 论文图脚本
tests/                    原有回归测试及整合验证
docs/                     方法、配置、输出说明；archive/ 保存历史记录
src/                      旧导入和 python -m src.* 的兼容入口
run_analysis.py           原 ROI 项目的 CLI 兼容入口
```

模块和数据流见 [architecture.md](docs/architecture.md)，迁移与保留策略见 [integration.md](docs/integration.md)。

## 环境配置

推荐保留服务器的 Python 3.9 GPU 环境；独立 CPU 分析支持 Python 3.9–3.12。仅分析已有结果 CSV 时无需 pyVHR、MediaPipe、CuPy、CUDA 或 GPU：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

`requirements.txt` 是 CPU 分析依赖的唯一来源；`requirements-dev.txt` 增加 pytest。原 `requirements_extra.txt` 保留为兼容入口。

视频实验需要已配置的 pyVHR、OpenCV、MediaPipe、CuPy 与匹配的 CUDA 运行库。服务器已有环境可按完整路径激活，避免把同名环境和 `/opt/anaconda3/envs/pyvhr` 混淆：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/conda_envs/pyvhr_GPU"
cd "$HOME/Project/Gamma_rPPG_Analysis"
```

已有 GPU 环境不必为了目录整合升级依赖。先核对 [环境记录](docs/environment.md)，缺少的分析/测试依赖可单独安装。不要直接升级 NumPy、MediaPipe、CuPy 或 CUDA；它们与现有 pyVHR 栈存在版本约束。

## 数据与配置

数据集、模型和研究结果保持在仓库之外。配置默认使用现有服务器的数据布局：

```bash
export GAMMA_DATA_ROOT=/tank/数据集
export GAMMA_RESULT_ROOT="$HOME/Result"
```

环境变量可以省略，默认值分别是 `/tank/数据集` 和 `~/Result`。`configs/datasets.yaml` 集中维护 UBFC2、LGI_PPGI 和 PURE 六类运动路径。其它 YAML 通过 `extends` 继承；相对路径以当前配置文件所在目录解析，字典递归合并，列表整体替换。

GPU 编号默认修正为本服务器实际存在的 `0`。如换用多卡服务器，应修改 `experiment.cuda_device`，或通过自己的 `*.local.yaml` 配置覆盖。正式 PURE 实验仍优先 AVI，禁止自动回退 PNG。

完整配置说明见 [CONFIGURATION.md](docs/CONFIGURATION.md)。

## 运行方法

以下命令在仓库根目录执行。视频实验使用 GPU 环境；只读 CSV 的统计/机制分析可使用 CPU 环境。

```bash
# 数据诊断和小样本检查
bash scripts/check_pure_avi_integrity.sh
bash scripts/inspect_pure_structure.sh
bash scripts/quick_test_all.sh

# 正式 pre-ROI 实验：默认断点续跑
bash scripts/run_pre_roi_all_datasets.sh
bash scripts/summarize_pre_roi.sh
bash scripts/analyse_pre_roi_sensitivity.sh

# 新整合的 ROI 机制分析：读取已有结果，不重跑视频
bash scripts/run_roi_mechanism_analysis.sh "$GAMMA_RESULT_ROOT/GammaBrightness_rPPG_v5"

# 典型案例信号诊断和固定 ROI 控制实验
bash scripts/generate_case_mechanism_figures.sh
bash scripts/run_post_roi_analysis.sh
```

ROI 机制分析可以显式指定独立输出目录，保留已有分析结果：

```bash
python -m gamma_rppg.analysis.roi_mechanism.cli \
  --config configs/roi_mechanism.yaml \
  --result-root "$HOME/Result/GammaBrightness_rPPG_v5" \
  --output "$HOME/Result/ROI_Gamma_Mechanism_new"
```

默认输出为 `<结果根目录>/roi_gamma_mechanism_analysis`，再次执行会更新其中的分析文件。`bash scripts/run_analysis.sh` 和 `python run_analysis.py` 仍接受原 ROI 项目的参数；原 `python -m src.*` 命令也继续可用。

主实验重试失败任务：`bash scripts/retry_failed_all.sh`。`--overwrite` 会删除所选数据集的旧实验输出，只有明确需要重跑时才使用。

## 科研口径与输出

- UBFC2、LGI_PPGI、PURE 分别统计，不跨数据集池化；ROI 分析总表虽包含多个 dataset 标签，每个检验族仍在数据集和方法内部计算。
- pre-ROI 和 post-ROI 结果物理隔离，模式比较使用共同完整样本及各自基线。
- `MAE benefit = MAE(gamma=1.0) - MAE(gamma)`，正值表示改善。
- ROI 基线关联只使用 Gamma=1.0 特征；视频内机制关联先按视频计算，再汇总视频，保留原随机种子、bootstrap 和多重检验范围。
- ROI 机制模块读取已有汇总 CSV，输出七张统计表、报告和八类图表；不训练模型，也不从 CSV 重建逐像素直方图。

详见 [输出说明](docs/OUTPUT_REFERENCE.md)、[ROI 机制分析](docs/roi_mechanism.md)、[统计口径](docs/STATISTICAL_ANALYSIS.md)。

## 验证与 Git

```bash
python -m compileall -q gamma_rppg src tests run_analysis.py
find scripts -name '*.sh' -exec bash -n {} \;
pytest
```

CPU 环境会跳过需要完整 pyVHR 的测试。视频/GPU 全量实验不能由单元测试替代，正式运行前仍需检查小样本。整合验证记录见 [VALIDATION.md](docs/VALIDATION.md)。

`.gitignore` 排除数据集、权重、结果表、日志、缓存、虚拟环境、本地配置和凭据。重要研究结果继续保存在外部结果目录；如需发布经过审核的样例表，应放在 `tests/fixtures/` 并显式检查内容。原项目和历史日志保存在服务器备份目录中。
