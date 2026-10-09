# 整合验证记录

2026-10-09，在原 Linux 服务器的独立暂存目录完成验证。现有研究结果只作为输入，验证输出位于 `~/Project_backups/gamma_integration_20261009/validation/`。

## 实际检查结果

- 原 Gamma_rPPG_Analysis：14 项测试通过；原 ROI 项目：2 项测试通过。
- 整合项目：22 项测试通过，包含原全部测试与 6 项配置、兼容、共享统计和 CPU 入口检查。
- 测试环境为 `~/conda_envs/pyvhr_local`，Python 3.9.19；本次未跳过 pyVHR 相关测试。
- Python compileall、全部 Shell 的 bash -n、Node 论文图脚本的 node --check 通过。
- 8 份原实验 YAML 与 ROI 配置的有效内容完全一致，唯一有意变化是 GPU 编号由 1 修正为 0。
- Wheel 构建和独立目录安装通过，离开源码目录后 CPU CLI 与打包默认配置仍可用。
- 激活原 pyvhr_GPU 后，CuPy 查询到 1 张 GPU，CUDA runtime 11030（11.3）。

## 真实结果表对照

使用现有 GammaBrightness_rPPG_v5 的 LGI_PPGI / CHROM 结果，对照整合前后的 ROI 程序：3 个基线特征、2 个机制特征、30 次 bootstrap，保留原随机种子和所有绘图类别。

- 七张输出 CSV 的列、行、数值完全一致。
- ANALYSIS_SUMMARY.md 文本完全一致。
- 20 张输出图像的相对路径和文件字节完全一致。
- 详细一致性记录保存为备份验证目录的 equivalence.json。

这验证了所选真实结果切片的机制分析兼容性，不代表重新执行了所有数据集、方法和视频。未覆盖正式研究结果，也未运行全量视频 GPU 实验。

## 后续验证方法

```bash
python -m compileall -q gamma_rppg src tests run_analysis.py
find scripts -name '*.sh' -exec bash -n {} \;
pytest -q
```

关键验收条件：

- 正式 PURE 配置在 AVI 和 PNG 同时存在时选择 AVI；
- 禁止回退时，只有 PNG 的样本不会进入正式实验；
- AVI 检查脚本独立输出，不写入主实验 CSV；
- pre/post 结果物理隔离；
- post-ROI 同一视频所有 Gamma 共用相同关键点与 ROI；
- Gamma=1.0 的 post-ROI 像素保持不变；
- pre/post 比较只使用共同完整样本，并分别使用自身基线；
- 敏感性分析只读取 pre-ROI 结果；
- 基线规律只使用 Gamma=1.0 ROI 特征；
- PURE 六类运动仍统一进入 PURE 主结果；
- 不生成跨数据集合并结果。

以上单元测试和结果表对照不能代替新环境的视频端到端测试。正式全量运行前仍须先执行 quick test。
