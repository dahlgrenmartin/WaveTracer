import numpy as np
import torch

from utils import decode_raw, encode, load_image, load_vae


INPUT_PATH = "dataset/laion_png_1024_raw/00000/000000000.png"   # real example
#INPUT_PATH = "dataset/sdxl_png_1024_original_500/03264.png"        # synth example
MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
VAE_SUBFOLDER = ""            
DTYPE = "fp16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lag1_autocorr(x):
    x = x - x.mean()
    h = np.mean(x[:, :-1] * x[:, 1:])
    v = np.mean(x[:-1, :] * x[1:, :])
    return float(0.5 * (h + v) / (np.mean(x * x) + 1e-30))


def spectral_slope(x, flo=0.01, fhi=0.5):
    x = x - x.mean()
    P = np.abs(np.fft.fft2(x)) ** 2
    H, W = P.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2).ravel(); p = P.ravel()
    m = (r >= flo) & (r < fhi) & (p > 0)
    return float(np.polyfit(np.log(r[m]), np.log(p[m]), 1)[0]) if m.sum() >= 8 else 0.0


def main():
    vae = load_vae(MODEL_ID, VAE_SUBFOLDER, DEVICE, DTYPE)
    for p in vae.parameters():
        p.requires_grad_(False)

    target = load_image(INPUT_PATH, DEVICE)         

    with torch.no_grad():
        recon = decode_raw(vae, encode(vae, target)) 
        residual = (recon - target).float().cpu().numpy()[0].mean(0)  

    lag1 = lag1_autocorr(residual)
    slope = spectral_slope(residual)
    print(f"lag1_autocorr={lag1:+.4f}   spectral_slope={slope:+.4f}")


if __name__ == "__main__":
    main()
