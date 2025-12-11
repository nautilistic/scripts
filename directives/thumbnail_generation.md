# Thumbnail Generation: Google Genai → Local Stable Diffusion

This guide explains how thumbnail generation works with Google Genai and how to swap to free local alternatives using Stable Diffusion.

## Current Usage (Google Genai)

### What It Does

The `recreate_thumbnails.py` script:
1. Downloads a YouTube thumbnail
2. Detects face pose (yaw/pitch) using MediaPipe
3. Finds matching reference photo of Nick by pose
4. Uses Google Genai to **swap the face** while preserving everything else
5. Outputs edited thumbnail

### The Key API Call

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)

# Send: reference photos + source thumbnail + prompt
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        reference_photo_1,   # Nick's face (reference)
        reference_photo_2,   # Nick's face (another angle)
        source_thumbnail,    # The thumbnail to edit
        prompt,              # "Swap the face with Nick's face"
    ],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    ),
)
```

### Why This Works (Google's Approach)

Google Genai uses a **single multimodal model** that:
- Understands images AND text together
- Can reference multiple input images
- Generates new images based on instructions
- Maintains face identity from reference photos

```
┌──────────────────────────────────────────────────────────┐
│                    GOOGLE GENAI                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                  │
│  │Reference│  │Reference│  │ Source  │  + "Swap face"   │
│  │ Photo 1 │  │ Photo 2 │  │Thumbnail│                  │
│  └────┬────┘  └────┬────┘  └────┬────┘                  │
│       │            │            │                        │
│       └────────────┴────────────┘                        │
│                    │                                     │
│                    ▼                                     │
│         ┌─────────────────────┐                         │
│         │  Single Multimodal  │                         │
│         │   Model (Gemini)    │                         │
│         └──────────┬──────────┘                         │
│                    │                                     │
│                    ▼                                     │
│            ┌─────────────┐                              │
│            │   Output    │                              │
│            │  Thumbnail  │                              │
│            └─────────────┘                              │
└──────────────────────────────────────────────────────────┘
```

---

## Local Alternative (Stable Diffusion Pipeline)

### The Challenge

There's no single local model that does what Google Genai does. Instead, you need a **pipeline of specialized models**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     LOCAL STABLE DIFFUSION PIPELINE                       │
│                                                                          │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────────────────┐│
│  │  Reference  │────▶│  Face Embedding │────▶│  IP-Adapter Face        ││
│  │   Photos    │     │  (InsightFace)  │     │  or InstantID           ││
│  └─────────────┘     └─────────────────┘     └───────────┬─────────────┘│
│                                                          │              │
│  ┌─────────────┐     ┌─────────────────┐                │              │
│  │   Source    │────▶│  ControlNet     │────────────────┤              │
│  │  Thumbnail  │     │  (structure)    │                │              │
│  └─────────────┘     └─────────────────┘                │              │
│                                                          │              │
│  ┌─────────────┐                                        │              │
│  │   Prompt    │────────────────────────────────────────┤              │
│  │  (text)     │                                        │              │
│  └─────────────┘                                        ▼              │
│                                              ┌─────────────────────┐   │
│                                              │  Stable Diffusion   │   │
│                                              │  XL + ControlNet    │   │
│                                              │  + IP-Adapter       │   │
│                                              └──────────┬──────────┘   │
│                                                         │              │
│                                                         ▼              │
│                                                  ┌─────────────┐       │
│                                                  │   Output    │       │
│                                                  │  Thumbnail  │       │
│                                                  └─────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Option 1: InstantID (Recommended)

**Best for:** Face-consistent generation from reference photos

InstantID is designed exactly for this use case - generate images that preserve a specific person's face identity.

#### Installation

```bash
# Create virtual environment
python -m venv sd_env
source sd_env/bin/activate

# Install dependencies
pip install torch torchvision
pip install diffusers transformers accelerate
pip install opencv-python insightface onnxruntime-gpu
pip install huggingface_hub

# Download models (one-time, ~15GB total)
python -c "
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline

# Base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0'
)

# InstantID models
hf_hub_download('InstantX/InstantID', 'ip-adapter.bin')
hf_hub_download('InstantX/InstantID', 'ControlNetModel/diffusion_pytorch_model.safetensors')
"
```

#### Code Implementation

```python
import cv2
import torch
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionXLInstantIDPipeline, ControlNetModel

# ============================================
# SETUP (do once)
# ============================================

# Face analyzer for embedding extraction
face_analyzer = FaceAnalysis(name='antelopev2', root='./')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Load ControlNet for face structure
controlnet = ControlNetModel.from_pretrained(
    "InstantX/InstantID",
    subfolder="ControlNetModel",
    torch_dtype=torch.float16
)

# Load pipeline
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.to("cuda")  # or "mps" for Mac

# Load IP-Adapter
pipe.load_ip_adapter_instantid("InstantX/InstantID", "ip-adapter.bin")


# ============================================
# FACE SWAP FUNCTION
# ============================================

def swap_face_instantid(
    reference_image: Image.Image,
    source_thumbnail: Image.Image,
    prompt: str = "professional YouTube thumbnail, high quality"
) -> Image.Image:
    """
    Swap face in thumbnail using InstantID.

    Args:
        reference_image: Photo of the person's face to use
        source_thumbnail: Thumbnail to edit
        prompt: Style/quality prompt

    Returns:
        Generated thumbnail with swapped face
    """
    # Extract face embedding from reference
    ref_cv2 = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    faces = face_analyzer.get(ref_cv2)

    if not faces:
        raise ValueError("No face detected in reference image")

    face_embedding = torch.tensor(faces[0].normed_embedding).unsqueeze(0)

    # Extract face keypoints for ControlNet (structure preservation)
    source_cv2 = cv2.cvtColor(np.array(source_thumbnail), cv2.COLOR_RGB2BGR)
    source_faces = face_analyzer.get(source_cv2)

    if source_faces:
        # Use source face structure
        face_kps = source_faces[0].kps
    else:
        # Fall back to reference structure
        face_kps = faces[0].kps

    # Generate
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted face",
        image_embeds=face_embedding,
        image=source_thumbnail,  # ControlNet uses this for structure
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5.0,
    ).images[0]

    return result
```

### Option 2: Roop/InsightFace (Simpler, Direct Face Swap)

**Best for:** Direct face replacement without regenerating the whole image

This approach directly swaps faces pixel-by-pixel rather than regenerating.

#### Installation

```bash
pip install insightface onnxruntime-gpu opencv-python

# Download face swap model
# Get inswapper_128.onnx from: https://github.com/facefusion/facefusion-assets
```

#### Code Implementation

```python
import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

# ============================================
# SETUP
# ============================================

# Face analyzer
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Face swapper model
swapper = insightface.model_zoo.get_model(
    'inswapper_128.onnx',
    download=True,
    download_zip=True
)


# ============================================
# SIMPLE FACE SWAP
# ============================================

def swap_face_roop(
    reference_image: Image.Image,
    source_thumbnail: Image.Image,
) -> Image.Image:
    """
    Direct face swap using InsightFace/Roop approach.

    Args:
        reference_image: Photo with the face to use
        source_thumbnail: Image where face will be replaced

    Returns:
        Thumbnail with swapped face
    """
    # Convert to CV2 format
    ref_cv2 = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
    source_cv2 = cv2.cvtColor(np.array(source_thumbnail), cv2.COLOR_RGB2BGR)

    # Detect faces
    ref_faces = face_app.get(ref_cv2)
    source_faces = face_app.get(source_cv2)

    if not ref_faces:
        raise ValueError("No face in reference image")
    if not source_faces:
        raise ValueError("No face in source thumbnail")

    # Swap face
    result = swapper.get(
        source_cv2,
        source_faces[0],
        ref_faces[0],
        paste_back=True
    )

    # Convert back to PIL
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)
```

### Option 3: IP-Adapter Face (Middle Ground)

**Best for:** Style-consistent generation with face similarity

#### Installation

```bash
pip install diffusers transformers accelerate
pip install ip-adapter
```

#### Code Implementation

```python
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterFaceIDPlusXL

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

# Load IP-Adapter Face
ip_model = IPAdapterFaceIDPlusXL(
    pipe,
    "h94/IP-Adapter-FaceID",
    "ip-adapter-faceid-plusv2_sdxl.bin",
    device="cuda"
)


def generate_with_face(
    reference_image: Image.Image,
    prompt: str,
    negative_prompt: str = "blurry, bad quality"
) -> Image.Image:
    """Generate image with face from reference."""

    result = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        face_image=reference_image,
        num_samples=1,
        width=1280,
        height=720,
        num_inference_steps=30,
        guidance_scale=7.5,
    )

    return result[0]
```

---

## Comparison

| Approach | Quality | Speed | VRAM | Best For |
|----------|---------|-------|------|----------|
| **Google Genai** | Excellent | Fast (API) | None | Production, no setup |
| **InstantID** | Very Good | ~30s | 12GB+ | Best local quality |
| **Roop/InsightFace** | Good | ~5s | 4GB | Direct swap, fast |
| **IP-Adapter Face** | Good | ~20s | 10GB+ | Style + face |

---

## Full Drop-in Replacement

Here's how to modify `recreate_thumbnails.py`:

### Before (Google Genai)

```python
from google import genai
from google.genai import types

API_KEY = os.getenv("NANO_BANANA_API_KEY")
MODEL = "gemini-3-pro-image-preview"

def recreate_thumbnail(source_image, reference_photos, ...):
    client = genai.Client(api_key=API_KEY)

    response = client.models.generate_content(
        model=MODEL,
        contents=reference_photos + [source_image, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    # ... extract image from response
```

### After (Roop - Simplest)

```python
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Global setup (do once)
_face_app = None
_swapper = None

def _init_face_swap():
    global _face_app, _swapper
    if _face_app is None:
        _face_app = FaceAnalysis(name='buffalo_l')
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        _swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

def recreate_thumbnail(source_image, reference_photos, ...):
    _init_face_swap()

    # Use first reference photo
    ref_image = reference_photos[0]

    # Convert to CV2
    ref_cv2 = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    source_cv2 = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)

    # Detect faces
    ref_faces = _face_app.get(ref_cv2)
    source_faces = _face_app.get(source_cv2)

    if not ref_faces or not source_faces:
        print("Face detection failed")
        return None

    # Swap
    result = _swapper.get(source_cv2, source_faces[0], ref_faces[0], paste_back=True)

    # Convert back
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
```

---

## Hardware Requirements

| Method | Min VRAM | Recommended | Notes |
|--------|----------|-------------|-------|
| Roop/InsightFace | 4GB | 6GB | CPU fallback works |
| IP-Adapter Face | 8GB | 12GB | Needs GPU |
| InstantID | 10GB | 16GB | Best quality |

### Mac (Apple Silicon)

```python
# Use MPS instead of CUDA
pipe.to("mps")

# Or CPU (slower)
pipe.to("cpu")
```

### Low VRAM Tips

```python
# Enable attention slicing
pipe.enable_attention_slicing()

# Enable CPU offload
pipe.enable_model_cpu_offload()

# Use float16
pipe = pipe.to(torch.float16)
```

---

## Which Should You Use?

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION TREE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Want simplest swap?                                         │
│  └── YES → Use Roop/InsightFace (Option 2)                  │
│                                                              │
│  Want best quality?                                          │
│  └── YES → Have 12GB+ VRAM?                                 │
│            ├── YES → Use InstantID (Option 1)               │
│            └── NO  → Use Google Genai (keep API)            │
│                                                              │
│  Want to regenerate entire thumbnail (not just face)?       │
│  └── YES → Use IP-Adapter Face (Option 3)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Locally

```bash
# 1. Install Roop (simplest option)
pip install insightface onnxruntime opencv-python

# 2. Test face swap
python3 -c "
from PIL import Image
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface

# Setup
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

# Load images
ref = cv2.imread('reference.jpg')
source = cv2.imread('thumbnail.jpg')

# Swap
ref_face = app.get(ref)[0]
src_face = app.get(source)[0]
result = swapper.get(source, src_face, ref_face, paste_back=True)

cv2.imwrite('output.jpg', result)
print('Saved to output.jpg')
"
```
