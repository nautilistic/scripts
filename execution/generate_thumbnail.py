#!/usr/bin/env python3
"""
Generate YouTube thumbnails from a reference face photo + text prompt.

This is a simplified alternative to recreate_thumbnails.py that generates
NEW thumbnails from scratch rather than face-swapping existing ones.

Uses: Stable Diffusion 1.5 + IP-Adapter FaceID (runs locally, free, 8GB VRAM)

Usage:
    python generate_thumbnail.py --face "reference.jpg" --prompt "excited man explaining AI"
    python generate_thumbnail.py --face "reference.jpg" --prompt "person holding money, shocked expression" -n 3
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / ".tmp" / "thumbnails"

# Thumbnail dimensions (16:9) - SD 1.5 works best at 512-768 range
WIDTH = 768
HEIGHT = 432  # 16:9 at 768 width


class ThumbnailGenerator:
    """Simple thumbnail generator using SD 1.5 + IP-Adapter (8GB VRAM friendly)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None

    def load_models(self):
        """Load SD 1.5 and IP-Adapter models (one-time setup)."""
        if self.pipe is not None:
            return

        print("Loading models (this may take a minute on first run)...")

        # Load SD 1.5 base (much lighter than SDXL)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # Load IP-Adapter for face consistency
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus-face_sd15.safetensors"
        )

        # Set IP-Adapter scale (0.0-1.0, higher = more face influence)
        self.pipe.set_ip_adapter_scale(0.7)

        self.pipe.to(self.device)

        # Memory optimizations for 8GB cards
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers enabled (extra memory savings)")
        except Exception:
            pass  # xformers not installed, that's fine

        print("Models loaded!")

    def generate(
        self,
        face_image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted face, bad anatomy, watermark, text",
        num_images: int = 1,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        seed: int = None,
    ) -> list[Image.Image]:
        """
        Generate thumbnail(s) with the given face and prompt.

        Args:
            face_image: Reference photo of the face to use
            prompt: Description of the thumbnail to generate
            negative_prompt: What to avoid
            num_images: Number of variations to generate
            guidance_scale: How closely to follow the prompt (5-15)
            num_inference_steps: Quality vs speed tradeoff (20-50)
            seed: Random seed for reproducibility

        Returns:
            List of generated PIL Images
        """
        self.load_models()

        # Prepare face image for IP-Adapter
        face_image = face_image.convert("RGB")

        # Add thumbnail-specific prompt enhancements
        enhanced_prompt = f"professional YouTube thumbnail, {prompt}, dramatic lighting, high contrast, eye-catching, 16:9 aspect ratio"

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        results = []
        for i in range(num_images):
            print(f"Generating image {i + 1}/{num_images}...")

            # Generate with IP-Adapter
            result = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image=face_image,
                width=WIDTH,
                height=HEIGHT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

            results.append(result)

            # Increment seed for variation
            if generator is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + i + 1)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate YouTube thumbnails with your face"
    )
    parser.add_argument(
        "--face", "-f",
        type=str,
        required=True,
        help="Path to reference face photo",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Description of the thumbnail to generate",
    )
    parser.add_argument(
        "--negative", "-neg",
        type=str,
        default="blurry, low quality, distorted face, bad anatomy, watermark",
        help="What to avoid in generation",
    )
    parser.add_argument(
        "--variations", "-n",
        type=int,
        default=1,
        help="Number of variations to generate",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Inference steps (more = better quality, slower)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="Guidance scale (higher = follows prompt more closely)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to run on",
    )

    args = parser.parse_args()

    # Load face image
    face_path = Path(args.face)
    if not face_path.exists():
        print(f"Error: Face image not found: {face_path}")
        sys.exit(1)

    print(f"Loading face reference: {face_path}")
    face_image = Image.open(face_path)

    # Create output directory
    date_folder = OUTPUT_DIR / datetime.now().strftime("%Y%m%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%H%M%S")

    # Initialize generator
    generator = ThumbnailGenerator(device=args.device)

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.variations} variation(s)...\n")

    images = generator.generate(
        face_image=face_image,
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_images=args.variations,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    # Save outputs
    output_paths = []
    for i, img in enumerate(images):
        if args.output and len(images) == 1:
            output_path = date_folder / args.output
        else:
            output_path = date_folder / f"{time_stamp}_gen_{i + 1}.png"

        img.save(output_path)
        output_paths.append(output_path)
        print(f"Saved: {output_path}")

    print(f"\n=== Generated {len(output_paths)} thumbnail(s) ===")
    return output_paths


if __name__ == "__main__":
    main()
