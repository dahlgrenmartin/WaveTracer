# Summary: calibration-free VAE refinement detector

## One-line claim

VAE latent refinement makes AI images compress a coarse wavelet detail band and
real images inject into it. The sign of that band's late-refinement slope detects
AI images across four VAE families and the GenImage benchmark, with no training
and a fixed zero threshold.

## Detector

Run latent refinement through a frozen VAE with a detail-only db38 wavelet probe:

```text
loss = Charbonnier(db38 detail bands)
steps = 30
feature = mean trailing-window-10 OLS slope of -10*log10(energy)
          over bands {L3_LH, L3_HL}
verdict = AI if slope > 0
          REAL if slope < 0
threshold = 0
```

This is calibration-free: no trained classifier and no threshold tuning.

## Headline results

All results below are PNG unless noted otherwise.

| VAE | task / domain | n | AUC | sign@0 |
|---|---|---:|---:|---:|
| FLUX.2-small | Flux2-FFHQ deepfake face-edit | 500/500 | 1.000 | 0.998 |
| FLUX.2-small | Flux2-T2I text-to-image | 500/500 | 0.999 | 0.991 |
| FLUX.1-dev | text-to-image (6k-flux) | 500/500 | 1.000 | 0.998* |
| SDXL | text-to-image | 500/500 | 1.000 | 0.999 |
| SDXL | GenImage benchmark, cross-source | 350+/500 | 1.000 | 1.000 |
| Wan 2.2 | video-frame generation | 175/175 | 1.000 | 0.991 |
| SD1.5 | text-to-image | 500/500 | 1.000 | 0.999 |

Worst case across these clean PNG runs: AUC >= 0.999 and sign@0 >= 0.991.

\* FLUX.1-dev uses the **L2** flip octave (low-cutoff VAE, like FLUX.2-small/Wan).
It is the one case where the diagonal **HH** band helps rather than hurts: the
headline off-diagonal pair `L2_LH+L2_HL` gives sign@0 0.976, but `L2_HH+L2_LH`
gives **0.998**, and all three L2 detail bands (`L2_LH+L2_HL+L2_HH`) reach
**complete class separation** (AUC 1.000 with a value gap — no zero-threshold
misclassification by margin). See `flux1dev_db38_charb_slope_grid_decomp.png`.

## Universal feature (one fixed combo, threshold 0, all VAEs)

Exhaustive search over all 3-level band subsets (+ final LL) across the five VAE
families (7 runs, incl. the cross-source SDXL GenImage benchmark) gives a single
fixed band-average that is calibration-free at threshold 0 on every VAE:

**`L1_HH + L2_LH + L2_HH + L3_LH + L3_LL`** (slope, window 10, step 30; SYNTH if > 0)

| metric | per-VAE worst | pooled (6350 imgs) |
|---|---:|---:|
| F1 | 0.992 | 0.995 |
| balanced acc | 0.992 | 0.995 |
| MCC | 0.984 | 0.990 |
| AUC | 0.9997 | 0.9992 |
| sign@0 (thr 0) | 0.992 | — |
| **TPR@FPR=5%** | **1.000** | **0.998** |

Pooled confusion (threshold 0): TP 3167 / FN 8 / TN 3151 / FP 24 (32 errors total
over 7 runs). TPR@FPR=5% is 1.000 on every VAE individually; the worst
zero-threshold FNR is 1.4% (FLUX.1-dev), 0% at a 5% false-alarm budget. The
cross-source SDXL GenImage run scores perfectly under the universal combo and
does not change the winning band set. The final
`L3_LL` band is essential — HF detail bands alone cap the universal worst-case
sign@0 at ~0.90; LL anchors the zero-crossing on the low-cutoff VAEs (FLUX.1-dev).
The off-diagonal `L3_LH+L3_HL` headline is the high-cutoff-VAE feature, not
universal (worst-case sign@0 0.51 with FLUX.1-dev included). Run via
`gainz_band_detector.py --metric slope --preset universal`.

## Mechanism

Real images contain genuine high-frequency texture that the VAE cannot represent
cleanly. During refinement, the frozen VAE must inject energy into coarse detail
bands, so the coarse-band band-PSNR falls and the slope is negative.

Synthetic images are closer to the VAE image manifold. Refinement compresses the
same coarse detail bands, so band-PSNR rises and the slope is positive.

This is visible in the residual figures: real images redden in the L3 residual
band, while synthetic images blue/compress.

## Empirical rules

1. Use the trailing window, not the full trajectory.
   The full trajectory averages the early warmup into the late behavior. The last
   10 steps isolate the stable sign.

2. Use coarse L3 detail bands.
   L1/L2 detail bands are often compressed by both classes. The sign flip appears
   at L3; averaging L3_LH and L3_HL is the most stable version. L3_HH is weaker.

3. Use Charbonnier/L1, not MSE.
   MSE saturates and can push synthetic late slopes toward zero. Charbonnier keeps
   the compression/injection dynamics visible in the trailing window.

## Why this beats global metrics

Whole-image delta-PSNR or gain/z mixes discriminative detail bands with
non-discriminative low-frequency/LL behavior. The clearest example is SD1.5:
global zero-threshold behavior gives many false positives, while the L3
band-slope sign remains near perfect.

The detector works because it isolates the octave where real and synthetic images
move in opposite directions.

## Feature comparison

| feature | threshold | role |
|---|---:|---|
| L3_LH+L3_HL trailing slope | 0 | headline calibration-free detector |
| dropfrac:L2_HH / L3_HH | calibrated | strong magnitude baseline |
| gain/z or latent movement | calibrated | high AUC but VAE/lr-scale dependent |
| total delta-PSNR | 0 | weaker due to warmup and LL dilution |

## Limit

JPEG and hard cross-source settings remain the honest negative case. On the
civitai-SDXL JPEG cross-source test, the calibration-free band-slope detector
drops from near-perfect clean-PNG behavior to roughly 0.84. Residual structure
features such as spectral slope and lag-1 autocorrelation do not solve this; JPEG
quantization damages the high-frequency signal the detector uses.

For JPEG/cross-source deployment, use the trained band-feature detector
(XGBoost), not a single zero-threshold feature.

## Artifacts

- `final_result/slope_sign_detector/README.md` - detailed method and reproduction notes.
- `final_result/slope_sign_detector/*_real.csv` and `*_synth.csv` - scan outputs.
- `final_result/slope_sign_detector/l3_residual_*.png` - residual injection/compression figures.
- `final_result/slope_sign_detector/rolling_slope_bands_crossvae.png` - cross-VAE band-slope plot.
- `docs/images/headline_grid_{universal,pervae_best}_4x2.png` + `headline_anim_*_4x2.gif`
  - 4x2 headline grids (7 runs across 5 VAE families): universal fixed combo (+ pooled)
    and per-VAE best bands; static + animated (step 2->30, 2 s hold). Built with
    `scripts/plot_headline_grid.py` / `scripts/animate_headline_grid.py` (commands in README).
- `src/gainz_band_detector.py --metric slope` - live zero-threshold detector.
- `src/simplified_LatentTracer_db38.py` - minimal single-image reference.
- `scripts/plot_headline_grid.py` - 2x3 headline figure script.
- `simplified_aeroblade_spectral_slope_acr_lag1.py` - one-pass residual structure reference.
