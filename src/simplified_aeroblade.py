import torch

from utils import decode_raw, encode, load_image, load_vae, psnr_from_unsigned, to_unsigned


INPUT_PATH = "dataset/flux2small/Flux2_dev_00004_.png"
MODEL_ID = "black-forest-labs/FLUX.2-small-decoder"
VAE_SUBFOLDER = "."
DTYPE = "fp16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    vae = load_vae(MODEL_ID, VAE_SUBFOLDER, DEVICE, DTYPE)
    for p in vae.parameters():
        p.requires_grad_(False)

    target = load_image(INPUT_PATH, DEVICE)

    with torch.no_grad():
        z = encode(vae, target)
        recon = decode_raw(vae, z)

        psnr = psnr_from_unsigned(to_unsigned(recon), to_unsigned(target))

    print(f"psnr={psnr:.6f}")


if __name__ == "__main__":
    main()
