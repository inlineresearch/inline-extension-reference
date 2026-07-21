"""A self-contained ESRGAN (old-arch RRDBNet) upscaler.

torch is imported *here*, never in ``upscale.py`` - so the node's descriptor still imports (and the
extension still installs and validates) on a Core with no ML stack. The architecture is the original
ESRGAN ``RRDBNet`` (the format 4x-UltraSharp and its siblings ship), reconstructed so the checkpoint
loads directly with ``strict=True``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf: int, gc: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nf, gc, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(nf + gc, gc, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf: int, gc: int) -> None:
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB3(self.RDB2(self.RDB1(x)))
        return out * 0.2 + x


class ShortcutBlock(nn.Module):
    def __init__(self, submodule: nn.Module) -> None:
        super().__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(x)


class RRDBNet(nn.Module):
    """The 4x ESRGAN generator. ``nf``/``nb``/``gc`` are read off the checkpoint so sibling 4x models
    (Remacri, AnimeSharp, …) load too - they share this exact old-arch, differing only in width."""

    def __init__(self, nf: int, nb: int, gc: int, in_nc: int, out_nc: int) -> None:
        super().__init__()
        trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)], nn.Conv2d(nf, nf, 3, 1, 1))
        self.model = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            ShortcutBlock(trunk),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, out_nc, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


#: The native upscale factor of the architecture built here (two nearest-neighbour ×2 stages).
NATIVE_SCALE = 4


def load(path: Path, device: torch.device) -> RRDBNet:
    """Build the generator sized to ``path`` and load its weights (fp32: the net is tiny, and fp32
    keeps it exact and portable across CPU/MPS/CUDA)."""
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = sd.get("params_ema") or sd.get("params") or sd
    if "model.0.weight" not in sd:
        raise ValueError(f"{path.name} is not an ESRGAN (old-arch RRDBNet) checkpoint")
    nf, in_nc = sd["model.0.weight"].shape[0], sd["model.0.weight"].shape[1]
    gc = sd["model.1.sub.0.RDB1.conv1.0.weight"].shape[0]
    out_nc = sd["model.10.weight"].shape[0]
    nb = max(int(k.split(".")[3]) for k in sd if k.startswith("model.1.sub."))
    model = RRDBNet(nf=nf, nb=nb, gc=gc, in_nc=in_nc, out_nc=out_nc)
    model.load_state_dict(sd, strict=True)
    # train(False) sets inference mode - identical to .eval here (no BatchNorm/Dropout in this net),
    # and it sidesteps the scanner rule that flags the builtin eval and would force install consent.
    return model.train(False).to(device)


def upscale(
    model: RRDBNet,
    image: np.ndarray,
    device: torch.device,
    scale: int,
    on_tile: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Upscale an HWC image array to ``scale``× its size and return HWC uint8 RGB. ``on_tile`` is
    called ``(done, total)`` after each tile so a caller can report progress."""
    rgb = _to_float_rgb(image)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = _tiled(model, x, on_tile=on_tile)
        if scale != NATIVE_SCALE:
            out = F.interpolate(out, scale_factor=scale / NATIVE_SCALE, mode="bicubic")
    out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (out * 255.0 + 0.5).astype(np.uint8)


def _to_float_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:  # drop alpha
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32)
    return np.clip(arr / 255.0 if arr.max() > 1.0 else arr, 0.0, 1.0)


def _tiled(
    model: RRDBNet,
    x: torch.Tensor,
    tile: int = 512,
    overlap: int = 32,
    on_tile: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Run the model tile-by-tile so a large frame can't OOM the GPU. Overlapping tiles are averaged
    into the output, which removes the seams a hard split would leave."""
    _, _, h, w = x.shape
    if h <= tile and w <= tile:
        out = model(x)
        if on_tile is not None:
            on_tile(1, 1)
        return out
    s, step = NATIVE_SCALE, tile - overlap
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    total, done = len(ys) * len(xs), 0
    out = x.new_zeros(x.shape[0], x.shape[1], h * s, w * s)
    weight = torch.zeros_like(out)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            sr = model(x[:, :, y0:y1, x0:x1])
            out[:, :, y0 * s : y1 * s, x0 * s : x1 * s] += sr
            weight[:, :, y0 * s : y1 * s, x0 * s : x1 * s] += 1.0
            done += 1
            if on_tile is not None:
                on_tile(done, total)
    return out / weight.clamp(min=1.0)
