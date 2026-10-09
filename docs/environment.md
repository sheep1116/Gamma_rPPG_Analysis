# 服务器环境记录

记录时间：2026-10-09。以下是已有环境的观测值，不是对任意新环境的安装保证。

- Ubuntu 22.04.4 LTS，Linux 6.8.0-1008-nvidia。
- Intel Xeon Platinum 8336C，64 个逻辑 CPU，约 125 GiB 内存。
- GPU 0：NVIDIA RTX A5000，24564 MiB 显存；驱动 580.178.04。
- ffmpeg、ffprobe、tmux、Node.js 和 Git 已存在。
- `/home` 约 99% 已使用，整合不复制数据集；新实验输出应先核对空间。

| 依赖 | ~/conda_envs/pyvhr_GPU | ~/conda_envs/pyvhr_local |
|---|---|---|
| Python | 3.9.19 | 3.9.19 |
| pyVHR | 2.0 | 2.0 |
| NumPy | 1.22.4 | 1.23.5 |
| pandas | 2.2.2 | 2.2.2 |
| SciPy | 1.10.1 | 1.10.1 |
| matplotlib | 3.8.4 | 3.8.4 |
| PyYAML | 6.0.1 | 6.0.1 |
| OpenCV | 4.9.0.80 | 4.9.0.80 |
| MediaPipe | 0.9.1.0 | 0.9.1.0 |
| CuPy | 9.6.0 | 9.6.0 |
| pytest | 未安装 | 8.4.2 |

环境列表中的这些目录未必能通过 `conda activate pyvhr_GPU` 这个短名称找到，推荐使用完整路径。`/opt/anaconda3/envs/pyvhr` 是另一个环境；base 使用 Python 3.11.7，不能与视频实验环境混用。

共享 CPU 依赖下限允许现有 NumPy 1.22.4；ROI 模块使用的 NumPy API 已包含在该版本中。依赖范围限定 NumPy <2，减少对旧 pyVHR 栈的冲击。原实验代码和所有配置默认继续读取既有 v5 结果路径。

激活既有 GPU 环境后，CuPy 实际查询返回 GPU 数量 1、CUDA runtime 11030（11.3）。这证明 CUDA 运行库可见，不能代替完整视频处理链的验证。查看 VALIDATION.md 中的检查范围；新服务器请先配置匹配的 GPU 依赖，再执行小样本验证。
