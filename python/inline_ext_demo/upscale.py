"""A node that upscales an image with an ESRGAN weight file.

The weights are declared as data in ``inline-extension.json``, not fetched here - Inline Studio's
model popup downloads them into ``models/upscale_models/``. The heavy work (torch + the RRDBNet) lives
in ``esrgan.py`` and is imported lazily inside ``run``, so this module still imports on a Core with no
ML stack - the node then reports its missing runtime instead of failing to load.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from inline_core.config import models_dir
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import ParamField, Port, Widget
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind
from inline_core.takes import AssetRef

WEIGHTS = "4x-UltraSharp.pth"


def _image_array(value: Any) -> np.ndarray:
    """Materialize an image input to a pixel array. An upstream generating node hands over a numpy
    array directly; a wired frame or imported asset arrives as an AssetRef pointing at a file - the
    same shape Core's own img2img input takes."""
    if isinstance(value, AssetRef):
        if value.ref == "path" and value.path:
            from PIL import Image

            return np.asarray(Image.open(value.path).convert("RGB"))
        raise ValueError("Demo Upscale needs a readable image path input.")
    return np.asarray(value)


@inline_node(
    type="demo/upscale",
    title="Demo Upscale",
    category="Enhance",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField(
            "model",
            "Weights",
            Widget.SELECT,
            WEIGHTS,
            # Filled from whatever is installed under models/upscale_models/.
            options_from="upscale_models",
        ),
        ParamField("scale", "Scale", Widget.NUMBER, 4, min=2, max=4, step=2),
    ),
)
class Upscale(NodeRunner):
    produces_takes = True

    def __init__(self) -> None:
        # The registrar instantiates the runner once and reuses it, so a loaded model persists across
        # runs. Keyed by (file, device) so switching weights or device reloads.
        self._cache: dict[tuple[str, str], Any] = {}

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        params = {**Upscale.__inline_descriptor__.defaults(), **node.params}
        sources = inputs.get("image") or []
        if not sources:
            raise ValueError("Connect an image into Demo Upscale before running.")
        image = _image_array(sources[0])

        weights = models_dir() / "upscale_models" / str(params["model"] or WEIGHTS)
        if not weights.is_file():
            raise FileNotFoundError(
                f"{weights.name} is not installed. Open the node's model popup to download it."
            )

        from . import esrgan  # lazy: keeps torch out of import/validation on a torch-less Core

        placement = ctx.policy.placement("vae")
        device = esrgan.torch.device(str(placement.device))
        key = (str(weights), str(device))
        model = self._cache.get(key)
        if model is None:
            model = self._cache[key] = esrgan.load(weights, device)

        result = esrgan.upscale(model, image, device, scale=int(params["scale"]))

        if ctx.takes is None:
            return NodeResult(outputs={"image": result})
        take = ctx.takes.save(ctx.run_id, node.id, result, params)
        return NodeResult(outputs={"image": result}, takes=[take])
