import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
DEFAULT_LOCAL_REPO = Path("/projects/weilab/liupeng/dinov3")

# Examples of available DINOv3 models.
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

LOCAL_WEIGHTS_DIR = Path("/projects/weilab/liupeng/models/dinov3")
LOCAL_WEIGHTS_FILENAMES = {
    MODEL_DINOV3_VITS: "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
}


def resolve_dinov3_location() -> str:
    env_path = os.getenv("DINOV3_LOCATION")
    if env_path:
        return env_path
    if DEFAULT_LOCAL_REPO.is_dir():
        return str(DEFAULT_LOCAL_REPO)
    return DINOV3_GITHUB_LOCATION


DINOV3_LOCATION = resolve_dinov3_location()
print(f"DINOv3 location set to {DINOV3_LOCATION}")


def make_transform(resize_size: int = 256) -> T.Compose:
    return T.Compose(
        [
            T.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_model(model_name: str) -> tuple[torch.nn.Module, Path]:
    if model_name not in LOCAL_WEIGHTS_FILENAMES:
        available = ", ".join(sorted(LOCAL_WEIGHTS_FILENAMES))
        raise ValueError(f"No local weights configured for model '{model_name}'. Available: {available}")

    weights_path = (LOCAL_WEIGHTS_DIR / LOCAL_WEIGHTS_FILENAMES[model_name]).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Expected weights at {weights_path}")

    source = "local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github"
    model = torch.hub.load(DINOV3_LOCATION, model_name, source=source, weights=str(weights_path))
    model.eval()
    return model, weights_path


def load_volume(volume_path: Path, slice_axis: int) -> tuple[np.ndarray, np.ndarray]:
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1, or 2 (got {slice_axis})")

    nii = nib.load(str(volume_path))
    volume = nii.get_fdata().astype(np.float32)

    # Drop extra channels if present (e.g., shape (H, W, D, 1)).
    if volume.ndim > 3:
        volume = volume[..., 0]

    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.moveaxis(volume, slice_axis, 0)
    return volume, nii.affine


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    vol_min = float(np.min(volume))
    vol_max = float(np.max(volume))
    if vol_max > vol_min:
        normalized = (volume - vol_min) / (vol_max - vol_min)
    else:
        normalized = np.zeros_like(volume, dtype=np.float32)
    return normalized.astype(np.float32, copy=False)


def slice_to_pil(slice_data: np.ndarray) -> Image.Image:
    slice_uint8 = np.clip(slice_data * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(slice_uint8).convert("RGB")


def parse_layers_arg(layers_arg: str) -> int | list[int]:
    spec = layers_arg.strip()
    if not spec:
        raise ValueError("Layer specification cannot be empty")

    if "," in spec:
        parts = [part.strip() for part in spec.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid layer specification '{layers_arg}'")
        try:
            layer_list = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid layer specification '{layers_arg}'. Use comma-separated integers.") from exc
        return layer_list

    try:
        return int(spec)
    except ValueError as exc:
        raise ValueError(
            f"Invalid layer specification '{layers_arg}'. Provide an integer or comma-separated integers."
        ) from exc


def extract_slice_feature_maps(
    model: torch.nn.Module,
    volume: np.ndarray,
    transform: T.Compose,
    device: torch.device,
    layers: int | list[int],
    batch_size: int = 8,
) -> tuple[np.ndarray, tuple[int, int]]:
    layer_feature_blocks: list[list[np.ndarray]] = []
    batch_tensors: list[torch.Tensor] = []
    spatial_shape: tuple[int, int] | None = None
    channel_dim: int | None = None

    for idx in tqdm(range(volume.shape[0]), desc="Extracting slice feature maps"):
        pil_image = slice_to_pil(volume[idx])
        tensor = transform(pil_image)
        batch_tensors.append(tensor)

        if len(batch_tensors) == batch_size:
            batch = torch.stack(batch_tensors, dim=0).to(device)
            with torch.inference_mode():
                outputs = model.get_intermediate_layers(batch, n=layers, reshape=True)
            for layer_idx, feature_map in enumerate(outputs):
                feature_map = feature_map.detach().cpu()
                channel_dim = feature_map.shape[1]
                spatial_shape = feature_map.shape[-2:]
                feature_array = feature_map.numpy().astype(np.float32)
                while len(layer_feature_blocks) <= layer_idx:
                    layer_feature_blocks.append([])
                layer_feature_blocks[layer_idx].append(feature_array)
            batch_tensors.clear()

    if batch_tensors:
        batch = torch.stack(batch_tensors, dim=0).to(device)
        with torch.inference_mode():
            outputs = model.get_intermediate_layers(batch, n=layers, reshape=True)
        for layer_idx, feature_map in enumerate(outputs):
            feature_map = feature_map.detach().cpu()
            channel_dim = feature_map.shape[1]
            spatial_shape = feature_map.shape[-2:]
            feature_array = feature_map.numpy().astype(np.float32)
            while len(layer_feature_blocks) <= layer_idx:
                layer_feature_blocks.append([])
            layer_feature_blocks[layer_idx].append(feature_array)

    if not layer_feature_blocks:
        if channel_dim is None or spatial_shape is None:
            return np.zeros((0, 0, 0, 0, 0), dtype=np.float32), (0, 0)
        zero_map = np.zeros((0, channel_dim, spatial_shape[0], spatial_shape[1]), dtype=np.float32)
        return zero_map[None, ...], spatial_shape

    stacked_layers = []
    for layer_blocks in layer_feature_blocks:
        if not layer_blocks:
            continue
        stacked_layers.append(np.concatenate(layer_blocks, axis=0))

    if not stacked_layers:
        assert spatial_shape is not None
        zero_map = np.zeros((0, channel_dim, spatial_shape[0], spatial_shape[1]), dtype=np.float32)
        return zero_map[None, ...], spatial_shape

    feature_tensor = np.stack(stacked_layers, axis=0)
    assert spatial_shape is not None
    return feature_tensor, spatial_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv3 features for each slice in a 3D NIfTI volume.")
    parser.add_argument("--input", dest="input_path", required=True, help="Path to the input .nii or .nii.gz volume")
    parser.add_argument("--output", dest="output_path", help="Optional path to save the slice features as .npz")
    parser.add_argument(
        "--model-name",
        default=MODEL_DINOV3_VITS,
        choices=sorted(LOCAL_WEIGHTS_FILENAMES.keys()),
        help="DINOv3 backbone to use",
    )
    parser.add_argument("--slice-axis", type=int, default=2, help="Axis along which to slice the volume (0, 1, or 2)")
    parser.add_argument("--resize", type=int, default=256, help="Resize shorter side to this resolution before encoding")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of slices to encode per forward pass")
    parser.add_argument(
        "--layers",
        default="1",
        help=(
            "Which transformer layers to extract. "
            "Use a single integer N to take the last N layers, or a comma-separated list of 0-indexed layer IDs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input volume not found: {input_path}")

    model, weights_path = load_model(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layers_spec = parse_layers_arg(args.layers)

    volume, affine = load_volume(input_path, slice_axis=args.slice_axis)
    volume = normalize_volume(volume)

    transform = make_transform(resize_size=args.resize)
    feature_maps, spatial_shape = extract_slice_feature_maps(
        model=model,
        volume=volume,
        transform=transform,
        device=device,
        layers=layers_spec,
        batch_size=max(1, args.batch_size),
    )

    output_summary = (
        f"Extracted feature maps with shape {feature_maps.shape}"
        if feature_maps.size
        else "No features extracted"
    )
    print(output_summary)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            feature_maps=feature_maps,
            slice_axis=args.slice_axis,
            source_volume=str(input_path),
            affine=affine,
            model_name=args.model_name,
            weights_path=str(weights_path),
            spatial_shape=np.array(spatial_shape),
            layer_spec=args.layers,
        )
        print(f"Saved slice features to {output_path}")


if __name__ == "__main__":
    main()
