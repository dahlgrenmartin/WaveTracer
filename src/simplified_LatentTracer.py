import torch
import torch.nn.functional as F

from utils import decode_raw, encode, load_image, load_vae, psnr_from_unsigned, to_unsigned

INPUT_PATH = "dataset/flux2small/Flux2_dev_00004_.png"
#INPUT_PATH = "dataset/laion_png_1024_raw/00000/000000000.png"
STEPS = 30
LR = 0.01
MODEL_ID = "black-forest-labs/FLUX.2-small-decoder"
VAE_SUBFOLDER = "."
DTYPE = "fp16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_row(step, pred, target):
    psnr = psnr_from_unsigned(to_unsigned(pred), to_unsigned(target))
    print(f"{step}: psnr={psnr:.6f}", flush=True)

def detail_loss(pred, target):
    return F.mse_loss(pred, target)

def main():
    vae = load_vae(MODEL_ID, VAE_SUBFOLDER, DEVICE, DTYPE)
    for p in vae.parameters():
        p.requires_grad_(False)

    target = load_image(INPUT_PATH, DEVICE)

    with torch.no_grad():
        z0 = encode(vae, target)
        baseline = decode_raw(vae, z0)
        log_row(0, baseline, target)

    z = z0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=LR)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DTYPE == "fp16" and DEVICE == "cuda"),
    )

    for step in range(1, STEPS + 1):
        pred = decode_raw(vae, z)
        loss = detail_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = decode_raw(vae, z)
            log_row(step, pred, target)


if __name__ == "__main__":
    main()
