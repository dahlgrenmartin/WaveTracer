import numpy as np
import time
import torch
import ptwt
from sklearn.linear_model import LinearRegression

from utils import decode_raw, encode, load_image, load_vae

#INPUT_PATH = "dataset/ffhq_random_500/people_without_glasses/deepfake_added_glasses/flux2dev-edit-add-glasses_00001_.png"   # FLUX.2 FFHQ synthetic example
#INPUT_PATH = "dataset/ffhq_random_500/people_without_glasses/real_unedited/00053.png"
INPUT_PATH = "dataset/sdxl_png_1024_original_500/03264.png"   # synth example
#INPUT_PATH = "dataset/laion_png_1024_raw/00000/000000000.png"   # real example

STEPS = 30
LR = 0.01  # 0.01 for SDXL/Wan/SD1.5 and 0.03 for FLUX.2
#MODEL_ID = "black-forest-labs/FLUX.2-small-decoder"
MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
VAE_SUBFOLDER = "."

DTYPE = "fp16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WAVELET = "sym4"
WINDOW = 10                   
CHARB_EPS = 2e-3

# Alternative to L1 for wavelet detaial loss
def charbonnier(x):
    return torch.sqrt(x * x + CHARB_EPS * CHARB_EPS).mean()

def ptwt_input(x):
    n, c, h, w = x.shape
    return x.float().reshape(n * c, 1, h, w)


def slope(values, window):
    y = np.asarray(values[-window:], dtype=np.float64)
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y)).reshape(-1, 1)
    return float(LinearRegression().fit(x, y).coef_[0])


def psnr(pred, target):
    # Images are in [-1, 1], so data_range=2 -> divide MSE by 4.
    mse = (pred - target).square().mean().clamp_min(1e-30)
    return float(-10.0 * torch.log10(mse / 4.0))


def main():
    vae = load_vae(MODEL_ID, VAE_SUBFOLDER, DEVICE, DTYPE)
    for p in vae.parameters():
        p.requires_grad_(False)

    target = load_image(INPUT_PATH, DEVICE)
    z = encode(vae, target).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == "fp16" and DEVICE == "cuda"))

    history = []
    s = float("nan")
    with torch.no_grad():
        baseline = psnr(decode_raw(vae, z), target)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        residual = decode_raw(vae, z) - target
        coeffs = ptwt.wavedec2(ptwt_input(residual), WAVELET, level=3, mode="zero")
        # coeffs = [L3_LL, (L3 details), (L2 details), (L1 details)]

        # level-1 detail bands -> refinement loss
        lh1, hl1, hh1 = coeffs[3]
        loss = torch.abs(lh1).mean() + torch.abs(hl1).mean() + torch.abs(hh1).mean() # L1

        # level-3 off-diagonal residual energy -> detector trace (band-PSNR).
        lh3, hl3, _ = coeffs[1]
        e = 0.5 * (lh3.detach().square().mean() + hl3.detach().square().mean())
        history.append(float(-10.0 * torch.log10(e + 1e-30))) # "band-PSNR"

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        s = slope(history, WINDOW)
        print(f"{step}: L3off_band_psnr={history[-1]:+.3f} slope={s:+.4f}", flush=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    verdict = "SYNTHETIC" if s > 0 else "REAL"
    print(f"\nbaseline={baseline:.4f}")
    print(f"Slope={s:+.4f}: {verdict}")
    print(f"wall_clock={time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
