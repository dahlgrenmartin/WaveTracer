import torch
import torch.nn.functional as F
from diffusers import AutoModel
from PIL import Image
import torchvision.transforms as T


def load_image(path, device):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    x = T.ToTensor()(img).unsqueeze(0).to(device)
    return x * 2.0 - 1.0


def load_vae(model_id, vae_subfolder, device, dtype):
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    kwargs = {"torch_dtype": torch_dtype}
    if vae_subfolder:
        kwargs["subfolder"] = vae_subfolder
    vae = AutoModel.from_pretrained(model_id, **kwargs)
    # Video / Qwen VAEs take (B, C, T, H, W); flag so encode/decode wrap a T=1 axis.
    vae._needs_temporal_dim = any(s in type(vae).__name__ for s in
                                  ("QwenImage", "Wan", "Video", "LTX", "Cosmos"))
    return vae.to(device).eval()


def _norm_params(vae, ref):
    # Return (shift, scale) so z_norm = (z_raw - shift) * scale
    cfg = vae.config
    if getattr(cfg, "scaling_factor", None) is not None:
        shift = float(getattr(cfg, "shift_factor", None) or 0.0)
        scale = float(cfg.scaling_factor)
        return (torch.tensor(shift, device=ref.device, dtype=ref.dtype),
                torch.tensor(scale, device=ref.device, dtype=ref.dtype))
    if getattr(cfg, "latents_mean", None) is not None and \
       getattr(cfg, "latents_std", None) is not None:
        mean = torch.tensor(cfg.latents_mean, device=ref.device, dtype=ref.dtype)
        std = torch.tensor(cfg.latents_std, device=ref.device, dtype=ref.dtype)
        view = (1, -1) + (1,) * (ref.ndim - 2)
        return mean.view(view), (1.0 / std).view(view)
    return (torch.tensor(0.0, device=ref.device, dtype=ref.dtype),
            torch.tensor(1.0, device=ref.device, dtype=ref.dtype))


def encode(vae, x):
    needs_t = getattr(vae, "_needs_temporal_dim", False)
    x_in = x.unsqueeze(2) if needs_t else x
    with torch.no_grad():
        z = vae.encode(x_in.to(vae.dtype)).latent_dist.mode()
        if needs_t:
            z = z.squeeze(2)
        shift, scale = _norm_params(vae, z)
        return ((z - shift) * scale).float()


def decode_raw(vae, z):
    needs_t = getattr(vae, "_needs_temporal_dim", False)
    shift, scale = _norm_params(vae, z)
    z_in = (z / scale + shift).to(vae.dtype)
    if needs_t:
        z_in = z_in.unsqueeze(2)
    dec = vae.decode(z_in, return_dict=False)[0]
    if needs_t:
        dec = dec.squeeze(2)
    return dec.float()


def to_unsigned(x):
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def psnr_from_unsigned(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    return (10.0 * torch.log10(1.0 / (mse + 1e-8))).item()
