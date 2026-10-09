# Validation record

The project was validated against the uploaded complete `GammaBrightness_rPPG_v5` result archive.

## Static checks

```text
python -m compileall .     passed
pytest                     2 passed
```

## End-to-end run

Input datasets discovered:

```text
LGI_PPGI
PURE
UBFC2
```

The end-to-end run completed with exit code 0 and generated:

- seven analysis CSV tables;
- an automatically generated Chinese Markdown summary;
- correlation forests;
- method-specific scatter plots;
- positive-versus-stable distributions;
- Gamma mechanism curves;
- paired sample-by-Gamma heatmaps;
- pixel-composition curves;
- failure heatmaps and success-rate plots;
- cross-method consensus plots.

The test environment used existing v5 result CSV files only. It did not require raw videos, MediaPipe, pyVHR, CUDA, or GPU access.
