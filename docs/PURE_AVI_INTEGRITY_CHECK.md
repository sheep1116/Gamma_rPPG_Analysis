# PURE AVI 完整性检查

该检查是完全独立的诊断脚本，不被主实验导入或调用，因此不会增加正式实验运行开销。

## 运行

```bash
bash scripts/check_pure_avi_integrity.sh
```

## 检查内容

- AVI 是否存在、是否为非空文件；
- OpenCV 是否能够打开；
- 声明 FPS 和声明帧数；
- 实际逐帧解码数量；
- 声明帧数与实际解码帧数差；
- 实际时长；
- 与预期 30 FPS、30 秒的偏差；
- 与 PURE JSON 生理信号长度估算时长的偏差；
- 同目录 PNG 数量，仅作诊断比较。

## 输出

```text
diagnostics/PURE_AVI_CHECK/
├── pure_avi_integrity.csv
├── pure_avi_integrity_summary.json
├── missing_avi_samples.csv
├── unreadable_avi_samples.csv
├── duration_mismatch_samples.csv
└── logs/
```

脚本不修复 AVI、不补帧，也不会自动改变正式实验样本集合。应人工审核报告，再决定是否修复、排除或重新制作 AVI。
