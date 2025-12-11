# Generate Thumbnails (Simplified)

Generate YouTube thumbnails from a face reference + prompt. Runs locally, completely free.

## Setup (One-time)

```bash
# Install dependencies
pip install -r execution/requirements_thumbnail_gen.txt

# First run downloads models (~7GB) - takes a few minutes
python execution/generate_thumbnail.py --face "your_photo.jpg" --prompt "test"
```

## Usage

```bash
# Basic generation
python execution/generate_thumbnail.py \
  --face ".tmp/reference_photos/nick.jpg" \
  --prompt "excited man explaining AI automation"

# Multiple variations
python execution/generate_thumbnail.py \
  --face "reference.jpg" \
  --prompt "person holding cash, shocked expression, money background" \
  -n 3

# Higher quality (slower)
python execution/generate_thumbnail.py \
  --face "reference.jpg" \
  --prompt "professional teaching coding" \
  --steps 50 --guidance 8.0

# Reproducible results
python execution/generate_thumbnail.py \
  --face "reference.jpg" \
  --prompt "tech entrepreneur" \
  --seed 42
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--face`, `-f` | required | Path to your face reference photo |
| `--prompt`, `-p` | required | What to generate |
| `--variations`, `-n` | 1 | Number of images to generate |
| `--steps` | 30 | Quality (20=fast, 50=quality) |
| `--guidance` | 7.0 | Prompt adherence (5-15) |
| `--seed` | random | For reproducible results |
| `--device` | cuda | cuda, mps (Mac), or cpu |

## Tips

- **Face photo**: Use a clear, well-lit photo. Front-facing works best.
- **Prompts**: Be descriptive. Include emotion, setting, lighting.
- **Variations**: Generate 3-5 and pick the best.
- **Text**: Add text in Canva/Figma after generation (AI text is unreliable).

## Comparison

| | Old (recreate_thumbnails.py) | New (generate_thumbnail.py) |
|---|---|---|
| Approach | Face-swap existing thumbnails | Generate from scratch |
| Cost | ~$0.20/image (API) | Free (local) |
| Complexity | 600 lines, pose detection | 200 lines, simple |
| Flexibility | Limited to source layout | Full creative control |
| Speed | 10-60s (API latency) | 20-40s (local GPU) |
