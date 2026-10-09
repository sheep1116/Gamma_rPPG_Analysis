# Case Mechanism Figures

`Project_v6` adds a post-processing module for paper-ready mechanism figures. It does not rerun the full Gamma-rPPG experiment. It reads existing `pre_roi_gamma` results, selects representative positive, stable, and failure cases, and regenerates a small number of diagnostic signals from the original videos.

All figure text is written in English for direct use in a conference paper.

## Outputs

For each selected case, the module writes:

| File | Content |
|---|---|
| `roi_luma_histogram.png` | Encoded ROI luma distribution at Gamma=1.0 and the target Gamma |
| `rgb_timeseries.png` | ROI RGB mean time series at Gamma=1.0 and the target Gamma |
| `bvp_spectrum.png` | Normalized BVP spectrum for the selected rPPG method |
| `case_metadata.csv/json` | Dataset, sample, method, Gamma values, MAE values, case category, and selection reason |
| `signal_extraction_status.csv` | Whether signal extraction succeeded for each Gamma |
| `bvp_spectrum_status.csv` | Whether BVP spectral analysis succeeded for each Gamma |

The top-level output folder also contains `case_index.csv` and `README.md`.

## Case Selection

The script reads:

- `raw_results.csv`
- `pre_roi_sensitivity_labels.csv`

Case categories:

- `positive`: `response_type == positive_sensitive`, sorted by largest MAE improvement.
- `stable`: `response_type == stable`, sorted by smallest absolute relative improvement.
- `failure`: non-baseline Gamma tasks whose `status != ok`, sorted from higher Gamma values first.

The default number of cases is configured in `configs/case_mechanism_figures.yaml`:

```yaml
case_mechanism:
  cases_per_dataset:
    positive: 2
    stable: 2
    failure: 2
```

## Run On The Original Server Layout

If your v6 results use the default server layout in the YAML file:

```bash
cd Project_v6
conda activate pyvhr_GPU
bash scripts/generate_case_mechanism_figures.sh
```

The output will be written to:

```text
/home/yangjicheng/Result/GammaBrightness_rPPG_v6/case_mechanism_figures
```

## Run From Existing v5 Results

If you want to reuse an existing v5 result folder and write only the new figures to a v6 folder:

```bash
cd Project_v6
conda activate pyvhr_GPU
bash scripts/generate_case_mechanism_figures.sh \
  configs/case_mechanism_figures.yaml \
  /home/yangjicheng/Result/GammaBrightness_rPPG_v5/pre_roi_gamma \
  /home/yangjicheng/Result/GammaBrightness_rPPG_v6/case_mechanism_figures \
  all
```

On this Codex workspace, the equivalent local paths would be:

```powershell
cd G:\Project_Data\Gamma_rPPG_Codex\Project_v6
python -m src.generate_case_mechanism_figures `
  --config configs/case_mechanism_figures.yaml `
  --pre-roi-root ..\Result_v5\pre_roi_gamma `
  --output-root ..\Result_v6\case_mechanism_figures `
  --datasets all
```

The local command still needs access to the original video paths stored in `raw_results.csv`. If those paths point to the experiment server, run this module on that server.

## Dataset Subset

To generate figures only for one dataset:

```bash
bash scripts/generate_case_mechanism_figures.sh \
  configs/case_mechanism_figures.yaml \
  /home/yangjicheng/Result/GammaBrightness_rPPG_v5/pre_roi_gamma \
  /home/yangjicheng/Result/GammaBrightness_rPPG_v6/case_mechanism_figures \
  UBFC2
```

For multiple datasets:

```bash
... UBFC2,PURE
```

## Notes

- `roi_luma_histogram.png` uses the same pre-ROI semantics as the main experiment: Gamma is applied to the full frame before FaceMesh and ROI extraction.
- `bvp_spectrum.png` uses the selected method from pyVHR and the same band-pass filter settings as the main experiment.
- Failure cases may intentionally fail at the target Gamma. In that case, the script keeps the available baseline plots and writes the failure message to the status files instead of stopping the whole run.
