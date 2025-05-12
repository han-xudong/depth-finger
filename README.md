# Depth Finger

This work presents Depth Finger, a depth-based deformation perception approach for soft finger in robotic manipulation.

## Usage

### Prepraration

```bash
git clone https://github.com/han-xudong/depth-finger.git
pip install -e.
```

To download pretrained checkpoints, follow the code snippet below:

```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

### Hardware

Please refer to the [Meta Finger](https://github.com/han-xudong/meta-finger) repository for the hardware.

### Use our models

```python
from depthfinger import DepthFinger

depthfinger = DepthFinger(model='vitl', camera_params=camera_params)

result = depthfinger.infer()
```

The `camera_params` is a dictionary containing the camera intrinsic parameters, which is defined via `.yaml` file. The `model` can be one of the following:

| Name | Model | Params |
|:-:|:-:|:-:|
| `vits` | Depth-Anything-V2-Small | 24.8M |
| `vitm` | Depth-Anything-V2-Base | 97.5M |
| `vitl` | Depth-Anything-V2-Large | 335.3M |

## Acknowledgement

We are sincerely grateful to the awesome Hugging Face team ([@Pedro Cuenca](https://huggingface.co/pcuenq), [@Niels Rogge](https://huggingface.co/nielsr), [@Merve Noyan](https://huggingface.co/merve), [@Amy Roberts](https://huggingface.co/amyeroberts), et al.) for their huge efforts in supporting our models in Transformers and Apple Core ML.

We also thank the [DINOv2](https://github.com/facebookresearch/dinov2) team for contributing such impressive models to our community.

## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.
