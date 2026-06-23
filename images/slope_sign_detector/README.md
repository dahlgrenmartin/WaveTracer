# Coarse-band slope-sign detector (white-box, matched-VAE)

## ★ FINAL recommended cross-VAE detector

**Scope (same as the baselines).** This is a *white-box, matched-VAE,
training-free* detector — like AEROBLADE (encode→decode reconstruction distance),
LatentTracer (latent optimization vs the decoder), and AEDR (double-reconstruction
ratio), it assumes access to the candidate generator's autoencoder and is evaluated
on real-vs-known-generator sets. Cross-/unknown-generator detection is out of scope
here (and unsolved by those baselines too).

**Primary metric = AUC** (directly comparable to AEROBLADE/AEDR, and offset-invariant
so calibration can't inflate it): ≥0.998 on every panel including the
out-of-selection Qwen-Image-Edit; pooled 0.9993.

**Feature:** `L2_LH+L2_HH+L3_LL` trailing-window-10 slope at step 30.
**Operating points:**
- *Calibration-free* (universal threshold 0, no offset): pooled sign@0 **0.945**
  — a genuine zero-threshold contribution vs baselines that need a tuned threshold.
- *With per-VAE calibration* (one scalar KDE offset per VAE; subtract then
  `synth if slope > 0`): pooled sign@0 **0.997**. CAVEAT: the offset below is fit
  **in-sample** (label-leaky calibration mode); treat it as an upper bound. It is a
  1-param per-VAE bias shown to transfer across that VAE's task domains by
  cross-domain 2-fold CV (see the `kde_offset_centering` note) — less calibration
  than most baselines need, but report it as calibrated, not calibration-free.
Figure: `docs/images/workshop_headline_kde_8vae.png` (+ `.gif`).

Machine-readable copy: `detector_universal_table.csv`. Current 8-panel set (Wan
video-frame replaced by Qwen-Image-Edit; Wan i2v on its targeted offset):

`sign@0 w/o KDE` = calibration-free universal threshold-0 accuracy (no offset);
`sign@0 w/ KDE` = after subtracting the per-VAE KDE offset. AUC is offset-invariant.

| VAE | task | n | offset@30 | sign@0 w/o KDE | sign@0 w/ KDE | AUC | TPR@5% |
|---|---|---|--:|--:|--:|--:|--:|
| FLUX.2-dev | FFHQ face-edit (i2i) | 500/500 | +0.022 | 0.996 | 1.000 | 1.0000 | 1.000 |
| FLUX.2-dev | Text-to-Image | 500/500 | +0.022 | 0.996 | 0.998 | 0.9980 | 1.000 |
| SDXL | Text-to-Image | 500/500 | −0.001 | 0.999 | 0.999 | 1.0000 | 1.000 |
| SDXL | GenImage (cross-source) | 500/500 | −0.001 | 0.998 | 0.999 | 1.0000 | 1.000 |
| Qwen-Image-Edit | Face edit (i2i) | 500/500 | −0.029 | 0.881 | 0.998 | 1.0000 | 1.000 |
| Wan2.2 | i2v smile-edit | 500/500 | −0.058 | **0.600** | 0.986 | 0.9994 | 0.996 |
| FLUX.1-dev | Text-to-Image | 1800/1800 | −0.018 | 0.987 | 0.998 | 0.9989 | 0.998 |
| SD1.5 | Text-to-Image | 500/500 | +0.013 | 0.996 | 0.998 | 1.0000 | 1.000 |
| **POOLED** | 8 panels | 5300/5300 | — | **0.945** | **0.997** | **0.9993** | **0.999** |

Pooled w/o KDE = 0.945 (dragged down almost entirely by Wan i2v's 0.600); per-VAE
KDE centering lifts it to 0.997. AUC is offset-invariant per panel, so the per-row
AUC is identical w/ or w/o KDE; only the *pooled* AUC depends on alignment —
0.9757 raw vs **0.9993** after centering. The Wan i2v row (0.600 → 0.986) is the
clearest case for the offset: separable (AUC 0.9994) but badly off-centre at 0.
Qwen-Image-Edit is the strongest
per-band-beats-global case (global gain/z inverts, ref AUC 0.11; slope AUC 1.000).

**Two findings that fixed the combo + centering:**
1. **Multi-task VAEs must share one offset** (Wan = video+i2v, like FLUX.2 FFHQ+T2I,
   SDXL T2I+GenImage). Under that constraint **`L2_LH+L2_HH+L3_LL` wins** (worst
   sign@0 0.989) — `L3_LL`'s offset is *task-invariant*; higher-AUC `L3_HH` combos
   lose because their offset drifts between a VAE's tasks (worst 0.91–0.98).
2. **The single best per-VAE offset = pooled-KDE** (concat all tasks, fit one
   crossover) — beats averaging the per-task offsets (worst 0.989 vs 0.971); it
   equals the brute-force optimal threshold over the VAE's images.

**Canonical figures** (8-panel 4×2, both Wan tasks stacked, no pooled panel):
recommended = `headline_grid_universal_centered_4x2.png`/`.gif`; no-KDE limit =
`headline_grid_universal_4x2.png`; alternatives = `headline_grid_{L2_LH_L3_HH,
L2_LH_L2_HH_L3_HH,L2_LH}_centered_4x2.png`; per-VAE-best =
`headline_grid_pervae_best_4x2.png`. Offset tables: `center_correction/*_pervae_offsets.json`.

### Per-DEPLOYMENT calibration (generation vs deepfake split)

The "one offset per VAE" rule above pools *all* of a VAE's tasks. But a VAE can be
deployed on **two physically different detection problems**: **generation**
(`t2i`/`t2v`, a prompted image) vs **deepfake** (`i2i`/`i2v`, a reference image
edited locally). These are different off-manifold physics (full vs local), so the
KDE offset is calibrated **per task-TYPE, not per VAE**: pool only same-type tasks,
keep cross-type separate. Concretely:

- **SDXL** T2I + GenImage are both `t2i` → **pooled** into one offset.
- **Wan2.2** video (`t2v`, generation) and i2v smile-edit (`i2v`, deepfake) →
  **separate** offsets (−0.044 vs −0.058). Sharing one mis-centres one task.
- **FLUX.2-dev** FFHQ face-edit (`i2i`) and T2I (`t2i`) → **separate** (+0.020 vs +0.026).

Same combo `L2_LH+L2_HH+L3_LL`, step 30, offset = pooled-KDE over same-type data:

| deployment | type | n | offset | KDE-acc | AUC |
|---|---|--:|--:|--:|--:|
| FLUX.2-dev FFHQ face-edit | deepfake (i2i) | 500×2 | +0.020 | 1.000 | 1.0000 |
| FLUX.2-dev Text-to-Image | generation (t2i) | 500×2 | +0.026 | 0.998 | 0.9980 |
| SDXL Text-to-Image | generation (t2i) | 500×2 | −0.001 | 0.999 | 1.0000 |
| SDXL GenImage (cross-source) | generation (t2i) | 500×2 | −0.001 | 0.999 | 1.0000 |
| Wan2.2 video frame | generation (t2v) | 175×2 | −0.044 | 0.997 | 1.0000 |
| **Wan2.2 i2v smile-edit** | **deepfake (i2v)** | **300×2** | **−0.058** | **0.988** | 0.9996 |
| FLUX.1-dev Text-to-Image | generation (t2i) | 1800×2 | −0.018 | 0.998 | 0.9989 |
| SD1.5 Text-to-Image | generation (t2i) | 500×2 | +0.013 | 0.998 | 1.0000 |

Mean KDE-acc **0.997**. Table: `center_correction/L2_LH_L2_HH_L3_LL_bydeployment.json`
(`method: pooled_kde_per_deployment`; keys carry the type, e.g. `Wan2.2 (i2v)`).
Figure: `docs/images/headline_grid_bydeployment_4x2.png` (panels labelled with type
+ image count). Wan i2v uses the full **300/300** smile-edit scan.

Regenerate the synced per-deployment KDE table from the committed scan CSVs:

```bash
python scripts/build_bydeployment_offsets.py \
  --band L2_LH+L2_HH+L3_LL \
  --window 10 \
  --steps 30 \
  --out final_result/slope_sign_detector/center_correction/L2_LH_L2_HH_L3_LL_bydeployment.json
```

The output JSON is the table used by `gainz_band_detector.py`,
`plot_wavelet_curve.py`, `plot_headline_grid.py`, and
`animate_headline_grid.py`. When using the table, the label/key before `·` must
match the JSON key exactly (`FLUX.2-dev (i2i)`, `Wan2.2 (i2v)`, etc.).

### One fixed combo vs per-task-best combo (keep ONE combo)

We use the **single universal combo** `L2_LH+L2_HH+L3_LL` for every deployment, not
a hand-picked best combo per task. Searching each deployment's best combo (with its
own pooled-KDE offset) only lifts mean acc **0.9969 → 0.9984** and worst-case
**0.987 → 0.992** — and at a real cost:

- The per-task accuracies are **in-sample optima** (combo *and* offset selected on
  the same data) → optimistic; we have no held-out task to prove the combo choice
  transfers. By contrast the *offset* is cross-domain-CV-validated (it transfers);
  the *combo* choice is not (same class of move as learned band weights, which
  collapse LOVO to 0.745).
- The per-task winners don't even agree (`L2_LH`, `L2_LH+L2_HL`, `L3_LH`…) — noise-
  fitting on already-saturated AUCs, no physical rationale.
- It destroys the one-sentence claim: *one combo, threshold 0 (+ optional 1-param
  offset), every VAE and task.* The universal combo lands within ~0.5% of the
  per-task ceiling everywhere.

So **per-task-best is a ceiling footnote, not the detector.** Only Wan i2v gains
materially (0.987→0.992 with `L2_LH+L2_HL+L2_HH`).

---

The strongest, most thesis-aligned AI-image detector found: the **trailing-window
linear-regression slope of a coarse detail band's band-PSNR**, at the final
refinement step, with **threshold 0** (the sign). It is the slope/mechanism the
thesis describes — *real degrades, synth improves* — measured at the band level.

```
feature(image) = mean of [ OLS slope of -10*log10(E_b(k)) over k in [N-9..N] ]   (window 10)
                 for b in {L3_LH, L3_HL}     (averaged off-diagonal L3 detail bands)
verdict = SYNTHETIC if slope > 0  (band compresses / on-manifold)
          REAL      if slope < 0  (band degrades / off-manifold)
```

No trained model, no calibration: the threshold is literally **0**. Run it with
`src/gainz_band_detector.py --metric slope` (auto-sets band `L3_LH+L3_HL`, threshold 0,
steps 30, window 10). The `+`-joined band averages the two off-diagonal L3 detail bands
— the most robust cross-VAE form (avoids the per-band image noise that makes a single
band occasionally dip). `L3_HL` alone is the best single band; `L3_HH` is the worst (avoid).

## Headline result (PNG, db38 / detail-only / Charbonnier)

Averaged off-diagonal L3 slope (`L3_LH+L3_HL`), window 10, step 30, threshold 0:

| VAE | task / domain | n | AUC | sign@0 acc |
|---|---|---|---|---|
| FLUX.2-small | **Flux2-FFHQ** — deepfake face-edit (add/remove glasses) | 500/500 | 1.000 | 0.998 |
| FLUX.2-small | **Flux2-T2I** — text-to-image (flux2small) | 500/500 | 0.999 | 0.991 |
| FLUX.1-dev | text-to-image (laion vs 6k-flux) | 500/500 | 1.000 | 0.998† |
| SDXL | text-to-image (laion vs sdxl) | 500/500 | 1.000 | 0.999 |
| Wan2.2 | video-frame gen (laion vs wan) | 175/175 | 1.000 | 0.991 |
| SD1.5 | text-to-image (laion vs sd15) | 500/500 | 1.000 | 0.999 |
| **worst-case** | | | **0.999** | **0.991** |

† **FLUX.1-dev (added 2026-06-16, 5th VAE family)** flips at the **L2** octave
(low-cutoff, grouping with FLUX.2-small/Wan). It is the lone exception to rule 2's
"avoid HH": the headline off-diagonal `L2_LH+L2_HL` gives sign@0 **0.976** (the
club floor), but adding the diagonal band — `L2_HH+L2_LH` — lifts it to **0.998**,
and all three L2 detail bands (`L2_LH+L2_HL+L2_HH`) achieve **complete class
separation** (AUC 1.000 with a positive value gap; the two classes' slope ranges
do not overlap, so a small calibrated threshold separates with zero error — the
zero-threshold sign@0 0.993 only loses to the offset of the boundary, not overlap).
Recipe identical to the others (db38 / detail-only / Charbonnier / lr 0.01, FLUX.1-dev
VAE subfolder `vae`). Plot: `flux1dev_db38_charb_slope_grid_decomp.png`.

**Across FOUR VAE families (FLUX.2-small-decoder, SDXL, Wan2.2, SD1.5), all PNG,
matched (Charbonnier) loss: AUC ≥ 0.999, sign@0 ≥ 0.991, calibration-free.**
FLUX.2-small-decoder is tested on **two task domains** — a deepfake face-edit
(Flux2-FFHQ) and normal text-to-image (Flux2-T2I) — both clean, showing the
VAE-physics detector handles manipulation *and* full generation on the same VAE.
At full 500/500 the worst-case sign@0 (0.991) is tied between Flux2-T2I and Wan —
~4–5 borderline images per 1000; Flux2-T2I is the only sub-1.0 AUC (0.999).

**SD1.5 is the strongest per-band-beats-global case:** its *global* gain/z zero-threshold
fails (FPR 0.41 — real laion *improves* globally, partly the 1024-real-vs-512-synth
resolution mismatch), yet the averaged-off-diagonal L3 band slope is **0.999** (500/500). The whole-image ΔPSNR is
ambiguous there; the coarse-band slope still cleanly separates at zero. SD1.5 images are
AEROBLADE's released set (`dataset/sd15`, CLIP-interrogated prompts); a matched-512 real
set would widen its margin further. Per-VAE flip octave: **L2** for FLUX.2-small/Wan, **L3**
for SDXL/SD1.5 (the cutoff tracks the flip band); L3 is the universal pick.

## Universal feature (one fixed combo, calibration-free threshold 0)

Exhaustively scoring band subsets across all five VAE families — **excluding L1**
(it is the refinement-loss objective + JPEG-fragile) but **keeping L3_LL** (the
JPEG-robust coarse band) — yields a **single fixed 3-band average** whose
late-refinement slope sign separates real from synthetic at **threshold 0 on every
VAE**, with no per-VAE octave choice:

**`L2_LH + L2_HH + L3_LL`**  (slope, window 10, step 30; verdict = SYNTH if slope > 0)

| VAE | TP | FN | TN | FP | Prec | Rec | Spec | F1 | BalAcc | MCC | AUC | TPR@FPR5% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| FLUX.2-sm FFHQ | 500 | 0 | 496 | 4 | 0.992 | 1.000 | 0.992 | 0.996 | 0.996 | 0.992 | 1.0000 | 1.000 |
| FLUX.2-sm T2I | 500 | 0 | 496 | 4 | 0.992 | 1.000 | 0.992 | 0.996 | 0.996 | 0.992 | 0.9980 | 1.000 |
| FLUX.1-dev | 485 | 15 | 500 | 0 | 1.000 | 0.970 | 1.000 | 0.985 | 0.985 | 0.970 | 0.9999 | 1.000 |
| SDXL | 500 | 0 | 499 | 1 | 0.998 | 1.000 | 0.998 | 0.999 | 0.999 | 0.998 | 1.0000 | 1.000 |
| SDXL GenImage | 499 | 1 | 499 | 1 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.996 | 1.0000 | 1.000 |
| Wan2.2 | 174 | 1 | 175 | 0 | 1.000 | 0.994 | 1.000 | 0.997 | 0.997 | 0.994 | 1.0000 | 1.000 |
| SD1.5 | 500 | 0 | 496 | 4 | 0.992 | 1.000 | 0.992 | 0.996 | 0.996 | 0.992 | 1.0000 | 1.000 |
| **Pooled** | **3158** | **17** | **3161** | **14** | **0.996** | **0.995** | **0.996** | **0.995** | **0.995** | **0.990** | **0.9991** | **0.999** |

(F1 = harmonic mean of precision/recall; BalAcc = mean of recall/specificity.
TP/FN/TN/FP and the rates left of AUC are at the **fixed zero threshold**;
TPR@FPR5% is the calibrated operating point. Pooled over 6350 images (7 runs incl.
the cross-source SDXL GenImage benchmark): 31 errors at threshold 0 — 14 FP +
17 FN — and TPR@FPR=5% is **1.000 on every VAE** individually, 0.999 pooled, so the
worst zero-threshold FNR is 3.0% (FLUX.1-dev, 15/500) and 0% at a 5% false-alarm
budget. GenImage (cross-source SDXL) scores essentially perfectly.)

**`L3_LL` is what makes threshold 0 universal — and it is JPEG-robust.** L2/L3
*detail* bands alone cap the universal worst-case sign@0 at ~0.89 (FLUX.1-dev and
SD1.5 fail the zero threshold); adding the coarse `L3_LL` lifts worst-case sign@0
to **0.985**. `L3_LL` alone is useless (it only *anchors the zero-crossing* in
combination), but because JPEG preserves low frequencies, keeping it costs no
JPEG-robustness — unlike the L1 detail bands, which JPEG destroys and which are
anyway the loss objective. No single band is universal at threshold 0
(best single = `L3_HH`, worst-case 0.75). The off-diagonal `L3_LH+L3_HL` headline
is the **high-cutoff-VAE** feature — its universal worst-case sign@0 is only 0.51
once FLUX.1-dev is included.

Run it (preset wires band + threshold 0 + steps 30):

```
python src/gainz_band_detector.py --metric slope --preset universal \
  --real-dir R --synth-dir S --model-id ... --vae-subfolder ... --lr <lr>
# presets: universal (all VAEs), offdiag-l3 (high-cutoff), offdiag-l2 (FLUX.1-dev)
```

## The three rules that make it work

1. **Measure the slope over a TRAILING WINDOW (last ~10 steps), not the full
   trajectory.** Real peaks at step 1–4 then degrades; a full-trajectory fit
   averages the early rise into the later fall, so real looks flat/positive. The
   trailing window clears the warmup transient. Total-PSNR slope sign@0: full
   0.73–0.91 → window-10 0.91–1.00; band-level → 0.98–1.00.
2. **Use a COARSE (level-3) detail band.** Fine bands (L1_HH, L2_HH) are compressed
   by *both* classes (both slopes > 0) → sign@0 ≈ 0.5 (not sign-usable, only
   magnitude separates). Only L3 bands flip sign: real injects HF energy
   (slope < 0), synth compresses (slope > 0).
3. **Charbonnier loss is required (on high-cutoff VAEs).** L2 loss *saturates* —
   synth finishes compressing early, so its late-window slope flattens to ~0 and
   sign@0 collapses (FLUX2-detail L2: 0.49). Charbonnier keeps synth compressing
   through the late window (FFHQ-charb: 1.00). Low-cutoff VAEs (Wan/SDXL) survive
   L2 because real degrades hard regardless, but match the loss to be safe.

## Why slope-sign beats the alternatives (feature head-to-head)

| feature | calibration | AUC (worst) | acc (worst) |
|---|---|---|---|
| **L3_LH slope @ thr 0** | **none (sign)** | **1.000** | **0.99** (sign@0) |
| dropfrac:L2_HH | thr ≈ 0.3 | 0.996 | ~0.99 (calibrated) |
| gain/z (ΔPSNR/zd) | VAE-scale thr | 0.99 | needs thr |
| zd (latent move) | VAE-scale thr | ~1.0 | needs thr |
| total ΔPSNR @ 0 | sign | 0.998 | 0.82 (warmup/high-cutoff) |

`dropfrac:L2_HH ≈ 0.3` is the best *magnitude* feature (stable threshold across
VAE/lr/domain — FFHQ 0.29 / SDXL 0.31), but the slope-sign matches/beats its AUC
*and* needs no threshold. gain/z and zd separate well (AUC ~1.0) but their
thresholds are lr/VAE-scale-dependent.

## Limits (where the calibration-free sign fails)

- **JPEG / cross-source** (`sdxl_civitai_*` — real=laion-PNG, synth=civitai-JPEG,
  levels-2 detail probe): the zero-threshold sign **breaks** (L3_HH sign@0 0.83,
  others 0.50–0.73; global @0 FPR 0.60). But the *separability survives* — coarse-band
  **dropfrac:L3_HH AUC 0.98, calibrated-acc 0.97**. So JPEG/cross-source needs a
  *calibrated* threshold (or the trained XGBoost band detector ~94% on civitai),
  not the calibration-free zero. L1 is destroyed by JPEG (AUC 0.59) — levels-2
  detail correctly relies on L2/L3.
- **Qwen** (n=12, not saved here): inconclusive; favors L2_HH over L3, needs a
  proper-sized dataset.

## Headline figures (7 runs across 5 VAE families)

Static grids + animated (step 2→30, x fixed to the final step centred on 0,
2 s hold) versions in `docs/images/` (PNG) and here (collection copy):

| figure | what | files |
|---|---|---|
| Universal | one fixed combo `L2_LH+L2_HH+L3_LL` (no L1, keep L3_LL) + a pooled panel | `headline_grid_universal_4x2.png` / `headline_anim_universal_4x2.gif` |
| Per-VAE best | one combo **per VAE** (shared across that VAE's task domains) | `headline_grid_pervae_best_4x2.png` / `headline_anim_pervae_best_4x2.gif` |

**Band pool: exclude L1, keep L3_LL.** L1 is excluded because (a) it *is* the
refinement-loss objective (the db38 detail probe optimises L1 detail), so measuring
it is circular, and (b) JPEG destroys it. **L3_LL is kept** — it is the coarse
low-frequency band that JPEG *preserves*, and it anchors the zero-crossing on the
low/high-cutoff VAEs (FLUX.1-dev, SD1.5). So the pool is the L2/L3 detail bands +
L3_LL. Per-VAE combos chosen by worst-case sign@0 across that VAE's domains
(tie→AUC→fewer bands); same VAE shares one combo across its task domains:

| VAE | shared combo | per-domain sign@0 |
|---|---|---|
| FLUX.2-small | `L2_HH+L3_LH+L3_HL` | FFHQ 1.000 · T2I 0.998 |
| FLUX.1-dev | `L2_LH+L2_HH` | T2I 0.998 |
| SDXL | `L2_LH+L2_HL+L3_HH+L3_LL` | T2I 1.000 · GenImage 1.000 |
| Wan 2.2 | `L2_HL+L2_HH+L3_LL` | video 0.997 |
| SD1.5 | `L3_LH+L3_LL` | T2I 1.000 |

> **Recommended deployable detector.** For an unknown VAE, use the **3-band
> universal `L2_LH+L2_HH+L3_LL`**: one fixed feature, threshold 0, no
> training/calibration, **worst-case sign@0 0.987 / pooled AUC 0.9987 / TPR@FPR5%
> 0.999** across all 5 VAE families. With a known/suspected VAE and a saved KDE
> offset table, `L2_LH+L2_HH+L3_LL` matches the best adjusted accuracy observed
> here while remaining much stronger as a pooled/universal detector.
> `L2_LH+L3_HH` remains the cleaner small-offset-gap calibrated baseline.
> L3_LL is the JPEG-robust low-freq anchor that makes the universal zero-threshold
> detector work without the JPEG-fragile L1 band.

### Both variants — full metrics (threshold 0)

acc = sign@0; AUC = rank-based (synth positive); F1 = harmonic mean of
precision/recall; HM = harmonic mean of TPR/TNR. (Balanced classes → acc≈F1≈HM.)

**Universal combo** `L2_LH+L2_HH+L3_LL` (no L1, keep L3_LL):

| VAE | acc | AUC | F1 | HM | TPR@FPR5% |
|---|--:|--:|--:|--:|--:|
| FLUX.2-sm FFHQ | 0.996 | 1.0000 | 0.996 | 0.996 | 1.000 |
| FLUX.2-sm T2I | 0.996 | 0.9980 | 0.996 | 0.996 | 1.000 |
| FLUX.1-dev | 0.987 | 0.9989 | 0.987 | 0.987 | 0.998 |
| SDXL | 0.999 | 1.0000 | 0.999 | 0.999 | 1.000 |
| SDXL GenImage | 0.998 | 1.0000 | 0.998 | 0.998 | 1.000 |
| Wan 2.2 | 0.997 | 1.0000 | 0.997 | 0.997 | 1.000 |
| SD1.5 | 0.996 | 1.0000 | 0.996 | 0.996 | 1.000 |
| **Pooled** | **0.993** | **0.9987** | **0.993** | **0.993** | **0.999** |

**Per-VAE best bands** — one combo per VAE (shared across its domains), no L1 /
keep L3_LL (optimistic ceiling — bands selected on the same data):

| VAE | domain | shared combo | acc | AUC | F1 | HM |
|---|---|---|--:|--:|--:|--:|
| FLUX.2-small | FFHQ | `L2_HH+L3_LH+L3_HL` | 1.000 | 1.0000 | 1.000 | 1.000 |
| FLUX.2-small | T2I | `L2_HH+L3_LH+L3_HL` | 0.998 | 0.9999 | 0.998 | 0.998 |
| FLUX.1-dev | T2I | `L2_LH+L2_HH` | 0.998 | 1.0000 | 0.998 | 0.998 |
| SDXL | T2I | `L2_LH+L2_HL+L3_HH+L3_LL` | 1.000 | 1.0000 | 1.000 | 1.000 |
| SDXL | GenImage | `L2_LH+L2_HL+L3_HH+L3_LL` | 1.000 | 1.0000 | 1.000 | 1.000 |
| Wan 2.2 | video | `L2_HL+L2_HH+L3_LL` | 0.997 | 1.0000 | 0.997 | 0.997 |
| SD1.5 | T2I | `L3_LH+L3_LL` | 1.000 | 1.0000 | 1.000 | 1.000 |

**Publication caveat:** both band sets were chosen by exhaustively searching the
same runs they are scored on, so these are *upper bounds*, not leakage-free
estimates. (The per-VAE combo is now constrained to be shared across a VAE's task
domains and to L2/L3 detail bands, which removes the per-domain/L1/LL leakage but
not the in-sample band-selection leakage.) A publishable claim must select the band combo on held-out
VAEs/generators (or nested CV) and report on untouched ones, add baseline
comparisons (AEROBLADE/LatentTracer/AEDR) on identical data, and a JPEG/resize
robustness section (the calibration-free sign is known to drop to ~0.84 on JPEG
cross-source).

## Centering add-on — per-step KDE offset (optional, cross-domain validated)

The equal-average feature is near-zero-centred but carries a small *systematic
per-VAE bias*. An optional add-on records, per step, the Bayes-optimal threshold
(the KDE density crossover between the real/synth modes) into a tiny correction
table; applying `corrected = slope − offset[k]` makes the zero-decision the optimal
boundary. **It does not change the bands or the detector** — it is one calibrated
number per step. (`scripts/center_correction.py` builds/saves the table;
`scripts/plot_center_correction.py` plots it; `animate_headline_grid.py --center`
shows it.)

Universal combo `L2_LH+L2_HH+L3_LL`, step 30:

| VAE | raw sign@0 | KDE offset | adjusted |
|---|--:|--:|--:|
| FLUX.2-sm FFHQ | 0.996 | +0.020 | 1.000 |
| FLUX.2-sm T2I | 0.996 | +0.026 | 0.998 |
| FLUX.1-dev | 0.987 | -0.018 | 0.998 |
| SDXL | 0.999 | −0.001 | 0.999 |
| SDXL GenImage | 0.998 | −0.001 | 0.998 |
| Wan 2.2 | 0.997 | −0.044 | 0.997 |
| SD1.5 | 0.996 | +0.013 | 0.998 |

FLUX.1-dev improves **0.987 -> 0.998**; worst centered panel remains **0.997**.
Centering also makes the detector usable several steps earlier (the bottom panel
of `docs/images/kde_offset_over_steps.png`).

**Cross-domain 2-fold CV (the offset is a real VAE property, not overfitting).**
FLUX.2-small (FFHQ, T2I) and SDXL (T2I, GenImage) each have two same-VAE domains,
so the offset can be calibrated on one and tested on the other. The held-out
adjusted accuracy **equals the in-sample one** to 3 decimals:

| VAE | 2-fold CV held-out | in-sample |
|---|--:|--:|
| FLUX.2-small (FFHQ ↔ T2I) | 0.999 | 0.999 |
| SDXL (T2I ↔ GenImage, cross-source) | 0.998 | 0.998 |

These are hard shifts (face-edit ↔ general T2I; in-distribution ↔ cross-source
GenImage) yet the offset transfers perfectly — it is a systematic property of the
VAE's reconstruction geometry, independent of dataset content. **Offset
trajectory:** undefined at steps 0–1 (the slope needs ≥2 points); large at step 2
(±0.1–0.39, warmup transient) decaying to a small, stable plateau by ~step 10–20
(NOT linear — curvature ratio 0.4–1.1; an exponential-to-plateau fits the shape).
Deployment: calibrate the late-step (20–30) offset once on a reference set from the
suspected VAE (you have its decoder), reuse on unseen images. Tables:
`final_result/slope_sign_detector/center_correction/*_offsets.json`.

### KDE combo comparison: known-VAE vs universal

Step-30 comparison on the current multi-run headline set (7 runs across 5 VAE
families; FLUX.1-dev uses the completed `results/flux1_1800...` scan,
1800/1800 at measurement time). Two readings matter:

- **per-VAE KDE**: use one saved offset per suspected VAE; best when the VAE is
  known or can be guessed.
- **pooled/global KDE**: use one global threshold for all runs; best proxy for
  "universal" behavior.

| combo | raw @0 acc | per-VAE KDE acc | worst per-VAE panel | pooled KDE offset | pooled KDE acc | worst pooled panel | pooled AUC | max domain offset gap |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `L2_LH+L3_HH` | 0.9463 | 0.9978 | 0.9953 | +0.0035 | 0.9465 | 0.8956 | 0.9867 | 0.0096 |
| `L2_LH+L2_HH+L3_HH` | 0.9381 | **0.9984** | **0.9971** | +0.0085 | 0.9697 | 0.9160 | 0.9946 | 0.0276 |
| `L2_LH+L2_HH+L3_LL` | **0.9931** | **0.9984** | **0.9971** | -0.0066 | **0.9931** | **0.9800** | **0.9987** | **0.0062** |

**Finding.** `L2_LH+L3_HH` is clean and has a small cross-domain offset gap, but
it is not the best *universal* pooled feature: FLUX.1-dev and SD1.5 still want
opposite nontrivial offsets, so a single pooled offset leaves a weak FLUX.1-dev
panel. Adding `L2_HH` improves known-VAE KDE accuracy, but the `L3_HH` version
is still weaker as a pooled/global detector. Adding `L3_LL` is the universal
anchor: it keeps the pooled KDE offset near zero, matches the best per-VAE KDE
accuracy on the completed FLUX.1-dev scan, gives the best pooled accuracy/AUC,
and has the smallest cross-domain offset gap. So:

- Unknown VAE / one fixed detector: **`L2_LH+L2_HH+L3_LL`**.
- Known/suspected VAE with saved KDE offset: **`L2_LH+L2_HH+L3_LL`** is now tied
  for best adjusted accuracy among the tested combos and remains the universal
  choice.
- Clean calibrated baseline / small offset-gap story: **`L2_LH+L3_HH`**.

The saved tables used by the centered plots are:

- `center_correction/L2_LH_L3_HH_pervae_offsets.json`
- `center_correction/L2_LH_L2_HH_L3_HH_pervae_offsets.json`
- `center_correction/L2_LH_pervae_offsets.json`
- `center_correction/L2_LH_L2_HH_L3_LL_pervae_offsets.json`

### `L2_LH+L3_HH` + offset = clean calibrated baseline

The offset **decouples separation from centering**, so the feature is chosen for
separation and the offset handles the threshold. **Recommended combo:
`L2_LH+L3_HH`** — it keeps `L2_LH`'s top separation (AUC ≥0.9999, Cohen's d
4.7–7.3) while the coarse, *content-invariant* `L3_HH` **halves the cross-domain
offset gap** (FLUX.2 0.021→0.0096), so the per-VAE offset centres both domains to
within ±0.005. Worst adj acc 0.993 (FLUX.1-dev):

adj acc is reported both in-sample and cross-validated. `adj (CV)` = within-domain
**5-fold** CV of the offset (calibrate on 4 folds, test on the held-out fold);
2-domain VAEs are *additionally* validated by cross-domain **2-fold** CV
(calibrate on one domain, test on the other — col `xCV`):

| VAE | offset | raw sign@0 | adj (in-samp) | adj (5-fold CV) | xCV (cross-domain) | AUC |
|---|--:|--:|--:|--:|--:|--:|
| FLUX.2-sm FFHQ | +0.003 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0000 |
| FLUX.2-sm T2I | +0.013 | 0.998 | 0.998 | 0.998 | 0.998 | 0.9999 |
| FLUX.1-dev | −0.031 | 0.911 | 0.993 | 0.993 | — | 0.9999 |
| SDXL | +0.012 | 0.961 | 1.000 | 0.999 | 1.000 | 1.0000 |
| SDXL GenImage | +0.012 | 0.968 | 1.000 | 1.000 | 1.000 | 1.0000 |
| Wan 2.2 | −0.013 | 0.997 | 0.997 | 0.994 | — | 0.9999 |
| SD1.5 | +0.049 | 0.860 | 1.000 | 0.997 | — | 1.0000 |

Cohen's d 4.7–7.3 (classes 4.7–7.3 std apart). **CV status:** FLUX.2-small and
SDXL pass *both* within-domain 5-fold and cross-domain 2-fold CV (CV ≈ in-sample,
the offset transfers). FLUX.1-dev / Wan / SD1.5 have only one dataset each, so the
cross-domain fold (`xCV`) is N/A — but the within-domain 5-fold CV holds them
(0.993 / 0.994 / 0.997), confirming the offset generalizes to held-out images of
the same VAE. Figures: `headline_grid_L2_LH_L3_HH_centered_4x2.png` /
`headline_anim_L2_LH_L3_HH_centered_4x2.gif`; offsets
`center_correction/L2_LH_L3_HH_offsets_all.csv`.

**Offset-invariance trade-off (why this combo):** the offset's cross-domain gap
comes from *content-sensitive* fine bands; the *coarse* L3 diagonal/vertical bands
have a domain-invariant offset. So `L3_HL+L3_HH` alone drives the gap to ~0.0002
(one offset centres every domain exactly) but weakens FLUX.1-dev to 0.989 (L3
doesn't flip for the low-cutoff FLUX.1-dev); pure `L2_LH` maximises separation
(AUC 1.0000, FLUX.1-dev 0.996) but has the largest gap (0.021). `L2_LH+L3_HH` is
the balance: keep L2_LH's AUC, borrow L3_HH's invariance.

\#\#\#\# Clean single-band alternative: `L2_LH`

For the simplest possible feature, **single `L2_LH`** has the highest worst-case
AUC (**1.0000** every VAE, Cohen's d 4.6–7.8). Its raw sign@0 is *deliberately*
off-centre on some VAEs (SDXL 0.899, SD1.5 0.740) — perfectly separated but not at
0 — which the per-VAE KDE offset (CV-mean) fixes, at the cost of a larger offset
gap (FLUX.2 0.021, ±0.01 residual):

| VAE | raw sign@0 | offset | adj acc | AUC | Cohen's d |
|---|--:|--:|--:|--:|--:|
| FLUX.2-sm FFHQ | 1.000 | +0.015 | 1.000 | 1.0000 | 7.48 |
| FLUX.2-sm T2I | 0.996 | +0.036 | 1.000 | 1.0000 | 7.76 |
| FLUX.1-dev | 0.982 | −0.015 | 0.996 | 0.9999 | 5.37 |
| SDXL | 0.899 | +0.016 | 0.997 | 1.0000 | 5.92 |
| SDXL GenImage | 0.892 | +0.017 | 0.998 | 1.0000 | 5.77 |
| Wan 2.2 | 0.997 | −0.002 | 0.997 | 0.9999 | 7.18 |
| SD1.5 | 0.740 | +0.040 | 0.998 | 1.0000 | 4.61 |

The `L2_LH` offset also transfers cross-domain (2-fold CV: FLUX.2 0.998, SDXL↔GenImage
0.998). Figures: `headline_grid_L2_LH_centered_4x2.png` / `headline_anim_L2_LH_centered_4x2.gif`
(AUC 1.0, clean split at 0 every panel), `kde_offset_over_steps_L2_LH.png`. All
per-step offsets: `center_correction/L2_LH_offsets_all.csv` (7 VAEs × steps 2–30)
+ per-VAE `L2_LH_*_offsets.json`.

**Operating points:** (a) *zero-calibration / unknown VAE* — 3-band
`L2_LH+L2_HH+L3_LL` at threshold 0 (LL anchors centering), worst sign@0 0.987;
(b) *known/suspected VAE + saved KDE offset* — `L2_LH+L2_HH+L3_LL` ties the
highest adjusted accuracy among the tested calibrated combos while staying
universal; (c) *clean
calibrated baseline* — `L2_LH+L3_HH` + per-VAE KDE offset has a smaller
cross-domain offset gap and AUC ≥0.9999; (d) *max-separation / simplest* —
single `L2_LH` + offset, AUC 1.0 / adj acc ≥0.996 but a larger offset gap.

### Using the per-VAE offset in the live detector (early-step classification)

`scripts/build_pervae_offsets.py` writes a per-VAE, **per-step** offset table (one
offset per VAE = CV mean over its domains, every step 2–30):
`center_correction/*_pervae_offsets.json` (consolidated
`{per_vae:{vae:{step:off}}}`). The detector/plot scripts subtract `offset[step]`
before the sign test, so the **zero threshold is correct from the first few
steps**. Current headline tables:

- `L2_LH_L3_HH_pervae_offsets.json`
- `L2_LH_L2_HH_L3_HH_pervae_offsets.json`
- `L2_LH_pervae_offsets.json`
- `L2_LH_L2_HH_L3_LL_pervae_offsets.json`

Live detector example:

```
python src/gainz_band_detector.py --metric slope --band L2_LH+L3_HH --step 5 \
  --correction-table final_result/slope_sign_detector/center_correction/L2_LH_L3_HH_pervae_offsets.json \
  --correction-vae FLUX.1-dev \
  --real-dir R --synth-dir S --model-id black-forest-labs/FLUX.1-dev --vae-subfolder vae --lr 0.01
```

Worst-case accuracy across the 7 runs, raw vs offset-corrected, by step — the
correction makes the detector usable several steps earlier and it keeps improving:

| step | raw sign@0 | offset-corrected |
|---|--:|--:|
| 3 | 0.569 | 0.852 |
| 5 | 0.565 | **0.941** |
| 10 | 0.615 | 0.958 |
| 20 | 0.858 | 0.974 |
| 30 | 0.860 | **0.993** |

(Offsets are large early — ±0.2–0.6 at step 2, the warmup transient — and decay to
±0.05 by step 30; the table stores all of them so any `--step` is centred.)

Add **`--trace`** (optionally `--trace-every N`) to stream the offset-corrected
feature + verdict at *every* refinement step **live** (flushed as each step
completes, via a refine `step_callback`), so you can watch the sign flip and
tighten in real time during the image's refinement:

```
... --trace --trace-every 5
    step  5: slope=-0.21  off=-0.207  corr=-0.003  REAL
    step 10: slope=-0.15  off=-0.139  corr=-0.011  REAL
    step 30: slope=-0.06  off=-0.031  corr=-0.029  REAL
```

**Early stop (`--early-stop`).** The refine `step_callback` returns a stop signal
that breaks the loop, so this is a *true* early stop (saves GPU steps, not just an
early read). The rule uses the known future KDE offsets plus a geometric-tail
bound on the recent corrected-score drift. For slope metrics the early-stop guard
must wait until the trailing slope window has matured; the default
`--early-stop-kmin` is now `max(4, --window + 3)` (13 for the standard
`--window 10`). This avoids the universal `L2_LH+L2_HH+L3_LL` failure mode where
some FLUX.1-dev synthetics are still negative at steps 4-12 but cross positive by
step 13. `--early-stop-safety` (default 1.5; higher = later/safer) and
`--early-stop-kmin` tune it; needs `--correction-table`.

Window-size check on the completed FLUX.1-dev 1800/1800 scan:

| slope window | full-30 KDE acc | AUC | p1 margin | default early-stop mean | full-30 agreement | disagreements |
|---:|--:|--:|--:|--:|--:|--:|
| 10 | **0.9983** | **0.99894** | **0.0103** | 13.7 steps | **0.9986** | **5 / 3600** |
| 5 | 0.9972 | 0.99884 | 0.0076 | **9.3 steps** | 0.9967 | 12 / 3600 |

So `--window 5` is a speed/accuracy tradeoff, not an equal replacement. It stops
earlier but has smaller near-zero margins and more early-stop/full-30
disagreements. Use `--window 10` for result-quality runs.

## Frequency-domain crossover (FFT + chirp)

A radial-FFT decomposition of the per-step residual (`src/fft_band_probe.py
--transform fft`) + the chirp probe (`scripts/chirp_cutoff.py`) locate the per-VAE
spectral cutoff that the wavelet octaves only bracket. For each VAE there is a
single radial band where the trailing-slope is sign-opposite (real < 0 / synth > 0):

| VAE | detection crossover (cyc/px) | single-FFT-band sign@0 | chirp residual onset | wavelet flip octave |
|---|--:|--:|--:|---|
| SDXL | 0.09–0.13 | 1.000 | ~0.10 | L3 |
| FLUX.2-small | 0.13–0.18 | 1.000 | ~0.20 | L3 |
| FLUX.1-dev | 0.22–0.27 | 0.981 | ~0.22 | L2 |

Findings (`scripts/analyze_fft_crossover.py`):
- **Detection lives at the *onset* of reconstruction failure**, not the half-power
  cutoff. The chirp's relative residual crosses 0.5 much higher (SDXL 0.38,
  FLUX 0.55); above that **both** classes fail (FFT bands F7+ have sign@0 ≈ 0.5).
  The signal is a thin band at the foot of the rolloff.
- **FFT ≈ wavelet for AUC** (dropfrac AUC ≈ 1.0 across a wide band on every VAE);
  a *single* radial band at the crossover ≈ the 5-band wavelet universal. FFT's
  value is interpretability (it names the cutoff) + parsimony, not higher AUC.
- **The cutoff moves with the VAE** (0.11 → 0.15 → 0.24), so no fixed band/combo is
  universal at threshold 0 — the wavelet universal needs 5 bands + LL precisely to
  bracket a moving target. The principled universal detector is: chirp → measure
  `f_c` → band-pass refine + measure at `f_c` → slope-sign at 0.

**Cutoff-targeted band-pass loss — A/B done, it does NOT help (negative result).**
`src/losses/bandpass.py` (`detail_loss="bandpass"`, or `fft_band_probe.py --loss
bandpass --center f_c --bandwidth bw`) penalises residual energy in a soft radial
annulus at the cutoff `f_c` instead of the broad wavelet detail. Hypothesis:
concentrating optimisation pressure at the cutoff → bigger margin. **Result: the
opposite** — vs the broad detail-only Charbonnier baseline the band-pass margin is
*smaller* everywhere (Cohen's d FLUX.1-dev 6.51→4.51, SDXL 6.51→2.20, FLUX.2-small
9.18→7.51) and the crossover shifts higher. **Mechanism:** the detection signal is
a *collateral* effect of **broad** detail-band optimisation pushing real
off-manifold across the whole detail range (the collateral leak); a narrow
band-pass applies far less off-manifold pressure, so real diverges less. The broad
detail-only wavelet Charbonnier loss stays the best refinement objective; the
band-pass prototype is kept for reference but not recommended. Scans:
`results/fft_probe/*_fft16*.csv`, `*_bandpass.csv`.

## Files

- `*_real.csv` / `*_synth.csv` — db38 / detail-only / Charbonnier bands scans
  (steps 30, band-levels 3, with zd_k). lr in the filename now matches the run:
  **FLUX2 = lr0.03** (`ffhq_..._lr03`, `f2s_..._lr03`), **everything else = lr0.01**
  (`sdxl_..._lr01`, `wan_..._lr01`, `sd15_..._lr01`), and the JPEG `sdxl_civitai_...lr01_lvl2detail`.
- `rolling_slope_bands_crossvae.png` — per-band rolling slope, Wan/Qwen/SDXL/FLUX2.
- `rolling_slope.png` — total-PSNR rolling slope (FFHQ/f2s/SDXL).
- `dropfrac_compare.png` — dropfrac threshold stability across datasets.
- `sdxl_civitai_lvl2detail_decomp.png` — the JPEG/cross-source decomposition.

## Reproduce

```
# scan (per VAE; lr: FLUX2 0.03, SDXL/Wan 0.01)
python src/scan_dir.py --real-dir R --synth-dir S --out X_combined.csv \
  --model-id ... --vae-subfolder ... --dtype fp16 --steps 30 --lr <lr> \
  --loss wavelet --wavelet db38 --levels 1 --wavelet-band-weights 0 1 1 1 \
  --wavelet-loss-type charbonnier --wavelet-reduction mean \
  --bands --band-levels 3 --interleaved --checkpoint-every 50

# live detector (no model)
python src/gainz_band_detector.py --metric slope \
  --real-dir R --synth-dir S --model-id ... --vae-subfolder ... --lr <lr>
# universal preset = the cross-VAE 5-band combo at threshold 0:
python src/gainz_band_detector.py --metric slope --preset universal \
  --real-dir R --synth-dir S --model-id ... --vae-subfolder ... --lr <lr>
```

### Wavelet-curve decomposition plot

`src/plot_wavelet_curve.py` can apply the same saved KDE offset table to matching
`slope:<band>` AUC and histogram panels. The universal/global winner is
`L2_LH+L2_HH+L3_LL`, so make that the default example; use `L2_LH+L3_HH` only
when you specifically want the clean small-offset-gap baseline.

The argparse **defaults** (changed 2026-06-18) now give the preferred layout, so the
command is minimal: top-left raw PSNR, top-right ΔPSNR, Detection-AUC panel =
`psnr` + `slope:L2_LH+L2_HH+L3_LL`, per-band grid = `psnr`, distribution = the
universal slope, and the dotted per-band AUC overlay (`--panel-auc`) stays OFF.
Pass `--metric`/`--traj`/`--objective`/`--auc-curves`/`--hist`/`--panel-auc` only to override.

```bash
python src/plot_wavelet_curve.py \
  final_result/slope_sign_detector/qwenedit_db38_charbonnier_s30_lr01_real.csv \
  final_result/slope_sign_detector/qwenedit_db38_charbonnier_s30_lr01_synth.csv \
  --tags real synthetic \
  --out results/qwenedit_db38_charbonnier_s30_lr01_wavelet_decomp_universal_kde.png \
  --title "Qwen-Image-Edit (i2i), db38 Charbonnier, KDE-corrected L2_LH+L2_HH+L3_LL" \
  --kde-offset-table final_result/slope_sign_detector/center_correction/L2_LH_L2_HH_L3_LL_pervae_offsets.json \
  --kde-offset-vae Qwen-Image-Edit \
  --band-pct 10 90
```

Small-gap calibrated-baseline variant: switch the slope tokens to
`slope:L2_LH+L3_HH` and use
`results/flux1-dev_L2_LH_L3_HH_offsets.json` or
`final_result/slope_sign_detector/center_correction/L2_LH_L3_HH_pervae_offsets.json`.

### Headline-figure commands

`$D = final_result/slope_sign_detector`. Both scripts take repeated
`--dataset "LABEL" REAL.csv SYNTH.csv`; the animated script adds `--fps 6
--hold 2.0`. The label before `·` must match the offset-table VAE key when using
`--kde-offset-table`.

**8-panel layout (current), column-stacked by VAE, both Wan tasks, no pooled panel.**
Row-major fill, so row0 then row1; each VAE's tasks land in one column (FLUX.2-dev
FFHQ/T2I col1, SDXL T2I/GenImage col2, Wan video/i2v col3, FLUX.1-dev/SD1.5 col4).
FLUX.2 panels are labelled `FLUX.2-dev` to match the offset-table key. FLUX.1-dev
uses the 1800-checkpoint CSV.

Use a **bash array** for the 8 datasets (quoted labels survive word-splitting via
`"${DS[@]}"`; a plain string variable does NOT — the quotes would be literal):

```bash
D=final_result/slope_sign_detector
DS=(
  --dataset "FLUX.2-dev · FFHQ face-edit"      $D/ffhq_png_db38_charbonnier_s30_lr03_real.csv $D/ffhq_png_db38_charbonnier_s30_lr03_synth.csv
  --dataset "SDXL · Text-to-Image"             $D/sdxl_png_db38_charbonnier_s30_lr01_real.csv $D/sdxl_png_db38_charbonnier_s30_lr01_synth.csv
  --dataset "Wan2.2 · video frame"             $D/wan_db38_charbonnier_s30_lr01_real.csv $D/wan_db38_charbonnier_s30_lr01_synth.csv
  --dataset "FLUX.1-dev · Text-to-Image"       results/flux1_1800_db38_charbonnier_s30_lr01_real.csv results/flux1_1800_db38_charbonnier_s30_lr01_synth.csv
  --dataset "FLUX.2-dev · Text-to-Image"       $D/f2s_png_db38_charbonnier_s30_lr03_real.csv $D/f2s_png_db38_charbonnier_s30_lr03_synth.csv
  --dataset "SDXL · GenImage (cross-source)"   $D/sdxl-genimage_png_db38_charbonnier_s30_lr03_real.csv $D/sdxl-genimage_png_db38_charbonnier_s30_lr03_synth.csv
  --dataset "Wan2.2 · i2v smile-edit"          $D/wan_i2v_smile500_db38_charbonnier_s30_lr01_real.csv $D/wan_i2v_smile500_db38_charbonnier_s30_lr01_synth.csv
  --dataset "SD1.5 · Text-to-Image"            $D/sd15_db38_charbonnier_s30_lr01_real.csv $D/sd15_db38_charbonnier_s30_lr01_synth.csv)
```

GIF = same args with `scripts/animate_headline_grid.py --fps 6 --hold 2.0` → `headline_anim_*.gif`.

```bash
# ★ RECOMMENDED headline (L2_LH+L2_HH+L3_LL + per-VAE pooled-KDE offset, centered)
python scripts/plot_headline_grid.py --ncols 4 \
  --bands L2_LH L2_HH L3_LL \
  --kde-offset-table $D/center_correction/L2_LH_L2_HH_L3_LL_pervae_offsets.json \
  --suptitle "RECOMMENDED: L2_LH+L2_HH+L3_LL + per-VAE KDE offset (pooled per VAE)" \
  "${DS[@]}" \
  --out docs/images/headline_grid_universal_centered_4x2.png
cp docs/images/headline_grid_universal_centered_4x2.png docs/images/workshop_headline_kde_8vae.png

# Universal zero-threshold (no KDE) — shows the calibration-free limit (Wan i2v ~0.60)
python scripts/plot_headline_grid.py --ncols 4 \
  --bands L2_LH L2_HH L3_LL \
  --suptitle "Universal L2_LH+L2_HH+L3_LL, threshold 0 (no KDE)" \
  "${DS[@]}" \
  --out docs/images/headline_grid_universal_4x2.png

# Per-VAE best zero-threshold, one panel combo per dataset, no pooled panel.
python scripts/plot_headline_grid.py --ncols 4 \
  --suptitle "Per-VAE best detection bands - one combo per VAE (no L1, keep L3_LL), threshold 0" \
  --dataset "FLUX.2-small · FFHQ face-edit" $D/ffhq_png_db38_charbonnier_s30_lr03_real.csv $D/ffhq_png_db38_charbonnier_s30_lr03_synth.csv --panel-bands L2_HH+L3_LH+L3_HL \
  --dataset "SDXL · Text-to-Image" $D/sdxl_png_db38_charbonnier_s30_lr01_real.csv $D/sdxl_png_db38_charbonnier_s30_lr01_synth.csv --panel-bands L2_LH+L2_HL+L3_HH+L3_LL \
  --dataset "Wan2.2 · Video frame" $D/wan_db38_charbonnier_s30_lr01_real.csv $D/wan_db38_charbonnier_s30_lr01_synth.csv --panel-bands L2_HL+L2_HH+L3_LL \
  --dataset "FLUX.1-dev · Text-to-Image" results/flux1_1800_db38_charbonnier_s30_lr01_real.csv results/flux1_1800_db38_charbonnier_s30_lr01_synth.csv --panel-bands L2_LH+L2_HH \
  --dataset "FLUX.2-small · Text-to-Image" $D/f2s_png_db38_charbonnier_s30_lr03_real.csv $D/f2s_png_db38_charbonnier_s30_lr03_synth.csv --panel-bands L2_HH+L3_LH+L3_HL \
  --dataset "SDXL · GenImage (cross-source)" $D/sdxl-genimage_png_db38_charbonnier_s30_lr03_real.csv $D/sdxl-genimage_png_db38_charbonnier_s30_lr03_synth.csv --panel-bands L2_LH+L2_HL+L3_HH+L3_LL \
  --dataset "SD1.5 · Text-to-Image" $D/sd15_db38_charbonnier_s30_lr01_real.csv $D/sd15_db38_charbonnier_s30_lr01_synth.csv --panel-bands L3_LH+L3_LL \
  --out docs/images/headline_grid_pervae_best_4x2.png

# Centered clean calibrated baseline.
python scripts/plot_headline_grid.py --ncols 4 --pooled \
  --bands L2_LH L3_HH \
  --kde-offset-table $D/center_correction/L2_LH_L3_HH_pervae_offsets.json \
  --suptitle "Clean calibrated baseline L2_LH+L3_HH + per-VAE KDE offset (centered)" \
  "${DS[@]}" \
  --out docs/images/headline_grid_L2_LH_L3_HH_centered_4x2.png

# Centered known-VAE best among the tested KDE combos.
python scripts/plot_headline_grid.py --ncols 4 --pooled \
  --bands L2_LH L2_HH L3_HH \
  --kde-offset-table $D/center_correction/L2_LH_L2_HH_L3_HH_pervae_offsets.json \
  --suptitle "Known-VAE KDE combo L2_LH+L2_HH+L3_HH (centered)" \
  "${DS[@]}" \
  --out docs/images/headline_grid_L2_LH_L2_HH_L3_HH_centered_4x2.png

# Centered max-separation single band.
python scripts/plot_headline_grid.py --ncols 4 --pooled \
  --bands L2_LH \
  --kde-offset-table $D/center_correction/L2_LH_pervae_offsets.json \
  --suptitle "Single-band L2_LH + per-VAE KDE offset (centered)" \
  "${DS[@]}" \
  --out docs/images/headline_grid_L2_LH_centered_4x2.png

# Centered universal combo.
python scripts/plot_headline_grid.py --ncols 4 --pooled \
  --bands L2_LH L2_HH L3_LL \
  --kde-offset-table $D/center_correction/L2_LH_L2_HH_L3_LL_pervae_offsets.json \
  --suptitle "Universal L2_LH+L2_HH+L3_LL + per-VAE KDE offset (centered)" \
  "${DS[@]}" \
  --out docs/images/headline_grid_universal_centered_4x2.png
```

**Per-DEPLOYMENT (task-type-split) headline.** Same combo, but the offset table is
keyed per task-TYPE (`L2_LH_L2_HH_L3_LL_bydeployment.json`) and the labels carry the
type — `_vae_key` splits on `·`, so the prefix `"FLUX.2-dev (i2i)"` must match the
JSON key exactly. No `--pooled` (we do not pool across task types). Each panel auto-
annotates type + image count (`n 500×2`, `n 300×2`, …):

```bash
D=final_result/slope_sign_detector
T=$D/center_correction/L2_LH_L2_HH_L3_LL_bydeployment.json
python scripts/plot_headline_grid.py --ncols 4 \
  --bands L2_LH L2_HH L3_LL \
  --kde-offset-table $T \
  --suptitle "L2_LH+L2_HH+L3_LL + per-deployment KDE offset  (t2i/t2v = generation, i2i/i2v = deepfake; same-type pooled, cross-type separate)" \
  --dataset "FLUX.2-dev (i2i) · FFHQ face-edit"  $D/ffhq_png_db38_charbonnier_s30_lr03_real.csv $D/ffhq_png_db38_charbonnier_s30_lr03_synth.csv \
  --dataset "SDXL (t2i) · Text-to-Image"         $D/sdxl_png_db38_charbonnier_s30_lr01_real.csv $D/sdxl_png_db38_charbonnier_s30_lr01_synth.csv \
  --dataset "Wan2.2 (t2v) · video frame"         $D/wan_db38_charbonnier_s30_lr01_real.csv $D/wan_db38_charbonnier_s30_lr01_synth.csv \
  --dataset "FLUX.1-dev (t2i) · Text-to-Image"   results/flux1_1800_db38_charbonnier_s30_lr01_real.csv results/flux1_1800_db38_charbonnier_s30_lr01_synth.csv \
  --dataset "FLUX.2-dev (t2i) · Text-to-Image"   $D/f2s_png_db38_charbonnier_s30_lr03_real.csv $D/f2s_png_db38_charbonnier_s30_lr03_synth.csv \
  --dataset "SDXL (t2i) · GenImage"              $D/sdxl-genimage_png_db38_charbonnier_s30_lr03_real.csv $D/sdxl-genimage_png_db38_charbonnier_s30_lr03_synth.csv \
  --dataset "Wan2.2 (i2v) · smile-edit"          $D/wan_i2v_smile500_db38_charbonnier_s30_lr01_real.csv $D/wan_i2v_smile500_db38_charbonnier_s30_lr01_synth.csv \
  --dataset "SD1.5 (t2i) · Text-to-Image"        $D/sd15_db38_charbonnier_s30_lr01_real.csv $D/sd15_db38_charbonnier_s30_lr01_synth.csv \
  --out docs/images/headline_grid_bydeployment_4x2.png
# (GIF: same args with scripts/animate_headline_grid.py --fps 6 --hold 2.0)
```
