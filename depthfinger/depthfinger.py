import matplotlib
import cv2
import numpy as np
import torch
from .dpt import DepthAnythingV2
from .camera import WebCamera, UsbCamera


class DepthFinger:
    def __init__(
        self,
        model: str = "vitl",
        camera_params=None,
        detector_params=None,
        pred_only: bool = False,
    ):

        # Check if the camera parameters are provided
        if camera_params is None:
            raise ValueError("Camera parameters must be provided.")

        # Create the camera object based on the mode
        if camera_params["mode"] == "usb":
            self.camera = UsbCamera(
                name="Camera",
                camera_params=camera_params,
            )
        elif camera_params["mode"] == "web":
            self.camera = WebCamera(
                name="Camera",
                camera_params=camera_params,
            )
        else:
            raise ValueError("Invalid camera mode. Use 'usb' or 'web'.")

        print("{:-^80}".format(f" DepthAnythingV2 Initialization "))
        # Set the device to GPU if available, or MPS if available, otherwise CPU
        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # Model configurations for different models
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        
        # Create the model based on the encoder type
        if model not in model_configs:
            raise ValueError(
                f"Invalid encoder type. Available options are: {list(model_configs.keys())}"
            )
        print(f"model: {model}")
        print(f"Features: {model_configs[model]['features']}")
        print(f"Out channels: {model_configs[model]['out_channels']}")
        self.depth_anything = DepthAnythingV2(**model_configs[model])
        self.depth_anything.load_state_dict(
            torch.load(
                f"checkpoints/depth_anything_v2_{model}.pth", map_location=DEVICE
            )
        )
        self.depth_anything = self.depth_anything.to(DEVICE).eval()
        
        # Get the number of parameters in the model
        total_params = sum(param.numel() for param in self.depth_anything.parameters())
        print("Total parameters: {:.2f}M".format(total_params / 1e6))
        print("Device: ", DEVICE)
        print("{:-^80}".format(""))

        # Set the colormap for depth visualization
        self.cmap = matplotlib.colormaps.get_cmap("Spectral_r")
        
        # Set the flag for prediction only
        self.pred_only = pred_only
        
        # Set the input size for the model
        self.input_size = camera_params["height"]

    def infer(self):
        # Get the camera image
        raw_image = self.camera.readImage()
        
        # Infer depth
        depth = self.depth_anything.infer_image(
            cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), self.input_size
        )
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Colorize the depth map
        depth = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Return the result
        # If pred_only is True, return only the depth map
        if self.pred_only:
            return depth
        # Otherwise, return the concatenated image
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth])

        return combined_result

    def __del__(self):
        self.camera.release()
