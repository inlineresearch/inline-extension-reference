"""Node behaviour, without a running server. Run with `PYTHONPATH=python pytest`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inline_ext_demo.basic import Brightness, Invert
from inline_ext_demo.gradient import Gradient
from inline_ext_demo.upscale import Upscale


def _node(**params: Any) -> Any:
    return SimpleNamespace(params=params)


def _run(runner: Any, image: np.ndarray, **params: Any) -> np.ndarray:
    result = runner.run(_node(**params), {"image": [image]}, None)
    return np.asarray(result.outputs["image"])


def test_invert_flips_the_image_at_full_amount() -> None:
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    assert _run(Invert(), image, amount=1.0).tolist() == [[255, 127, 0]]


def test_invert_at_zero_is_a_no_op() -> None:
    image = np.array([[10, 200]], dtype=np.uint8)
    assert _run(Invert(), image, amount=0.0).tolist() == image.tolist()


def test_invert_uses_its_declared_default() -> None:
    """Params come from the descriptor's defaults merged under the node's saved values."""
    image = np.array([[0]], dtype=np.uint8)
    assert _run(Invert(), image).tolist() == [[255]]


def test_brightness_scales_and_clips() -> None:
    image = np.array([[10, 200]], dtype=np.uint8)
    assert _run(Brightness(), image, scale=2.0).tolist() == [[20, 255]]


def test_upscale_reports_a_missing_weight_file_clearly() -> None:
    image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="model popup"):
        _run(Upscale(), image, model="definitely-not-installed.pth")


def test_upscale_reads_an_image_from_an_asset_ref(tmp_path: Any) -> None:
    """A wired frame or asset arrives as an AssetRef, not a numpy array - the node must load it. No
    torch or weights needed, so this runs in CI."""
    from PIL import Image
    from inline_core.takes import AssetRef
    from inline_ext_demo.upscale import _image_array

    path = tmp_path / "in.png"
    Image.fromarray(np.full((3, 5, 3), 128, np.uint8)).save(path)
    out = _image_array(AssetRef(ref="path", path=str(path)))

    assert out.shape == (3, 5, 3)
    np.testing.assert_array_equal(_image_array(np.zeros((2, 2, 3), np.uint8)), 0)


def test_upscale_runs_the_real_esrgan_from_an_asset_ref() -> None:
    """End-to-end through the node the way the canvas drives it: an AssetRef image input and the
    actual 4x-UltraSharp weights. Skips where the ML stack or the weight file is absent (e.g. CI)."""
    pytest.importorskip("torch")
    from PIL import Image
    from inline_core.config import models_dir
    from inline_core.takes import AssetRef

    weights = models_dir() / "upscale_models" / "4x-UltraSharp.pth"
    if not weights.is_file():
        pytest.skip(f"{weights} not downloaded")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = f"{tmp}/in.png"
        Image.fromarray((np.random.rand(24, 32, 3) * 255).astype(np.uint8)).save(src)
        ctx = SimpleNamespace(
            policy=SimpleNamespace(placement=lambda role: SimpleNamespace(device="cpu")),
            takes=None,
            run_id="test",
        )
        inputs = {"image": [AssetRef(ref="path", path=src)]}
        out = np.asarray(Upscale().run(_node(scale=4), inputs, ctx).outputs["image"])

    assert out.shape == (24 * 4, 32 * 4, 3)
    assert out.dtype == np.uint8


def test_upscale_errors_clearly_when_no_image_is_connected() -> None:
    with pytest.raises(ValueError, match="Connect an image"):
        Upscale().run(_node(scale=4), {}, SimpleNamespace(takes=None))


def test_the_entry_point_registers_exactly_what_the_manifest_declares() -> None:
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parent.parent / "inline-extension.json").read_text(encoding="utf-8")
    )
    declared = {node["type"] for node in manifest["nodes"]}
    implemented = {
        cls.__inline_descriptor__.type for cls in (Gradient, Invert, Brightness, Upscale)
    }
    assert implemented == declared
