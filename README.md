# Inline Demo Extension

A reference extension for [Inline Studio](https://github.com/inline-studio/inline-studio). Copy this
repo as the starting point for your own.

It ships two independently toggleable sub-extensions:

| Sub-extension | Nodes | Shows |
| --- | --- | --- |
| `basic` | `demo/invert`, `demo/brightness` | The minimum: a decorated runner and a `register()` |
| `upscale` | `demo/upscale` | Declaring weights, and a `options_from` model picker (off by default) |

## Install it

Extensions → **Install from URL** → this repo's URL, tag `v1.0.0`.

## Layout

```
inline-extension.json          the manifest (identity, sub-extensions, dependencies, models)
python/inline_ext_demo/        your code - the package name is mandatory (see below)
  basic.py                     nodes with no extra dependencies
  upscale.py                   a node that needs a weight file
tests/test_nodes.py            plain pytest
```

**The package name is not a convention, it's enforced.** It must be `inline_ext_<your id>` with
dashes as underscores: `id: "acme-tools"` → `python/inline_ext_acme_tools/`. That's what guarantees
two extensions can never shadow each other's modules.

## Writing a node

```python
@inline_node(
    type="demo/invert",              # must be listed in the manifest's `nodes`
    title="Invert",
    category="Image",                # groups it in the add-node menu
    icon="wand",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("amount", "Amount", Widget.NUMBER, 1.0, min=0.0, max=1.0),),
)
class Invert(NodeRunner):
    produces_takes = False

    def run(self, node, inputs, ctx) -> NodeResult:
        ...

def register(reg: ExtensionRegistrar) -> None:
    reg.nodes(Invert)
```

The canvas builds the whole node UI from that descriptor: ports, the settings sidebar, take
history. There is no frontend code to write.

Set `output_kind=MediaKind.IMAGE` and `produces_takes = True` when your node should become a Frame
with its own take history.

## Dependencies

List them in `requirements`:

```json
"requirements": ["einops>=0.7,<0.9"]
```

They install into your extension's own folder. **You cannot install `torch`, `diffusers`,
`transformers`, `numpy`, or anything else in Inline's runtime.** Import those directly, they're
already there. Declaring one is a hard install failure, not a warning: two copies of torch in one
process is a segfault, not a version conflict.

If your dependency needs a version of a shared package that Inline doesn't have, the install fails
with both versions named. Align with the host, or say so in your README.

## Models

Declare weights and Inline handles the rest: the download popup, progress, and the
`options_from` dropdown all work with no code:

```json
"models": [
  { "id": "realesrgan-x4", "label": "Real-ESRGAN x4", "category": "upscale_models",
    "repo": "ai-forever/Real-ESRGAN", "repoFile": "RealESRGAN_x4.pth",
    "filename": "RealESRGAN_x4.pth", "approxBytes": 67040989 }
]
```

`category` must be one of Inline's model folders (`diffusion_models`, `vae`, `text_encoders`,
`loras`, `controlnet`, `checkpoints`, `clip_vision`, `upscale_models`, `embeddings`). Never download
weights yourself at import or run time.

## What the security review flags

Every install is scanned, and the same scan runs in registry CI.

**Blocked outright:** declaring a host package (`torch`, …); `exec`/`eval` over a decoded or
compressed payload; shipping a `setup.py`; bundling `libtorch`/CUDA binaries.

**Needs the user to type your extension's name:** `subprocess`, raw sockets, `ctypes`,
`pickle.loads`, `eval`, and network calls to a host that isn't a known model host, including any
URL built at runtime, since it can't be checked by reading the code.

**Just noted:** downloading from Hugging Face or GitHub, shipping a compiled `.so`, no license.

Keep to the first two lists empty and users install without a warning.

## Publishing

1. Tag a release (`v1.0.0`). Installs pin to the commit behind the tag, so moving it later doesn't
   change what an existing user has.
2. Open a PR against
   [inline-studio/extension-registry](https://github.com/inline-studio/extension-registry) adding
   `registry/<your-id>.json`.

## Testing

```bash
pip install -e /path/to/inline-studio/core
PYTHONPATH=python pytest
```
