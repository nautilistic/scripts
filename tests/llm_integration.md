# LLM Integration: Anthropic → Ollama

This guide explains how the project uses LLMs for metadata generation and how to swap from Anthropic's paid API to Ollama (free, local).

## Current Usage (Anthropic)

### Where It's Used

**File:** `execution/simple_video_edit.py`
**Function:** `generate_metadata()` (lines 231-363)
**Purpose:** Generate YouTube title summaries and chapter timestamps from video transcripts

### How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Video       │ ──▶ │ Whisper     │ ──▶ │ Claude API  │ ──▶ │ Metadata    │
│ (input.mp4) │     │ transcribes │     │ generates   │     │ - summary   │
│             │     │ to text     │     │ metadata    │     │ - chapters  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Current Code (Anthropic)

```python
import anthropic

# Setup
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# API call
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=2000,
    messages=[{"role": "user", "content": prompt}]
)

# Extract response
response_text = response.content[0].text.strip()
```

### The Prompt

The prompt sends a timestamped transcript and asks for JSON output:

```python
prompt = f"""Analyze this video transcript and generate:
1. A summary for the YouTube description
2. YouTube chapters with timestamps

TRANSCRIPT (with timestamps in seconds):
{transcript_with_times}

VIDEO DURATION: {duration:.0f} seconds ({duration/60:.1f} minutes)

Respond in this exact JSON format:
{{
    "summary": "<2-4 sentence summary...>",
    "chapters": [
        {{"time": "00:00:00", "title": "Introduction"}},
        {{"time": "00:02:30", "title": "Topic Title Here"}}
    ]
}}
...
"""
```

### Requirements

```bash
pip install anthropic
```

**Environment variable:**
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Cost:** ~$0.01-0.05 per video (depending on transcript length)

---

## Free Alternative (Ollama)

### What is Ollama?

Ollama runs open-source LLMs locally on your machine. No API keys, no costs, no data sent to cloud.

**Recommended models:**
| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `llama3:8b` | 4.7GB | Good | Fast |
| `llama3:70b` | 40GB | Excellent | Slow |
| `mistral` | 4.1GB | Good | Fast |
| `qwen2:7b` | 4.4GB | Good | Fast |

### Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (one-time download)
ollama pull llama3:8b
```

### Code Changes

#### Before (Anthropic)

```python
import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def generate_metadata(words, cuts, duration, title):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""..."""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()
    # ... parse JSON ...
```

#### After (Ollama)

```python
import ollama

def generate_metadata(words, cuts, duration, title):
    prompt = f"""..."""  # Same prompt works!

    response = ollama.chat(
        model='llama3:8b',
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response['message']['content'].strip()
    # ... parse JSON (same code) ...
```

### Full Drop-in Replacement

Replace the `generate_metadata` function in `simple_video_edit.py`:

```python
def generate_metadata(
    words: list[dict],
    cuts: list[tuple[float, float]],
    duration: float,
    title: str
) -> dict:
    """
    Generate YouTube summary and chapters using Ollama (local LLM).
    Returns dict with {title, summary, chapters}.
    """
    if not words:
        return {
            "title": title,
            "summary": "Video content.",
            "chapters": "00:00:00 Introduction"
        }

    # Group words into ~30 second chunks
    chunks = []
    current_chunk = []
    chunk_start = 0.0

    for w in words:
        current_chunk.append(w)
        if current_chunk and w["end"] - chunk_start >= 30:
            chunk_text = " ".join(word["word"] for word in current_chunk)
            chunks.append({"start": chunk_start, "end": w["end"], "text": chunk_text})
            chunk_start = w["end"]
            current_chunk = []

    if current_chunk:
        chunk_text = " ".join(word["word"] for word in current_chunk)
        chunks.append({"start": chunk_start, "end": current_chunk[-1]["end"], "text": chunk_text})

    transcript_with_times = "\n".join(
        f"[{c['start']:.0f}s - {c['end']:.0f}s]: {c['text']}"
        for c in chunks
    )

    prompt = f"""Analyze this video transcript and generate:
1. A summary for the YouTube description
2. YouTube chapters with timestamps

TRANSCRIPT (with timestamps in seconds):
{transcript_with_times}

VIDEO DURATION: {duration:.0f} seconds ({duration/60:.1f} minutes)

Respond in this exact JSON format:
{{
    "summary": "<2-4 sentence summary describing what the video covers. Write in third person ('This video covers...'). Be specific about the content.>",
    "chapters": [
        {{"time": "00:00:00", "title": "Introduction"}},
        {{"time": "00:02:30", "title": "Topic Title Here"}}
    ]
}}

Chapter guidelines:
- Generate 5-15 chapters marking major topic transitions
- First chapter MUST be at 00:00:00
- Chapters should be 1-2+ minutes apart (except intro/outro)
- Concise titles (2-6 words)

Return ONLY the JSON, no other text."""

    print("Generating metadata with Ollama (llama3)...")

    # ============================================
    # OLLAMA CALL (replaces Anthropic)
    # ============================================
    import ollama

    response = ollama.chat(
        model='llama3:8b',  # or 'mistral', 'qwen2:7b', etc.
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response['message']['content'].strip()
    # ============================================

    # Parse JSON response (unchanged)
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        return {"title": title, "summary": "Video content.", "chapters": "00:00:00 Introduction"}

    try:
        data = json.loads(json_match.group())
        summary = data.get("summary", "Video content.")
        chapters_list = data.get("chapters", [{"time": "00:00:00", "title": "Introduction"}])
    except json.JSONDecodeError:
        return {"title": title, "summary": "Video content.", "chapters": "00:00:00 Introduction"}

    # Adjust timestamps for cuts (unchanged)
    adjusted_chapters = []
    sorted_cuts = sorted(cuts, key=lambda x: x[0]) if cuts else []

    for chapter in chapters_list:
        time_str = chapter.get("time", "00:00:00")
        chapter_title = chapter.get("title", "Chapter")

        match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})$', time_str)
        if not match:
            match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
            if match:
                minutes, seconds = match.groups()
                original_time = int(minutes) * 60 + int(seconds)
            else:
                adjusted_chapters.append(f"{time_str} {chapter_title}")
                continue
        else:
            hours, minutes, seconds = match.groups()
            original_time = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

        time_removed = 0
        for cut_start, cut_end in sorted_cuts:
            if cut_end <= original_time:
                time_removed += cut_end - cut_start
            elif cut_start < original_time:
                time_removed += original_time - cut_start

        adjusted_time = max(0, original_time - time_removed)

        hours = int(adjusted_time // 3600)
        minutes = int((adjusted_time % 3600) // 60)
        seconds = int(adjusted_time % 60)

        adjusted_chapters.append(f"{hours:02d}:{minutes:02d}:{seconds:02d} {chapter_title}")

    chapters_text = "\n".join(adjusted_chapters) if adjusted_chapters else "00:00:00 Introduction"

    return {
        "title": title,
        "summary": summary,
        "chapters": chapters_text
    }
```

### Requirements Change

```bash
# Remove
pip uninstall anthropic

# Add
pip install ollama
```

### Environment Variables

```bash
# Remove from .env
# ANTHROPIC_API_KEY=sk-ant-...

# No API key needed for Ollama!
```

---

## Comparison

| Aspect | Anthropic (Claude) | Ollama (Local) |
|--------|-------------------|----------------|
| **Cost** | ~$0.01-0.05/video | Free |
| **Speed** | ~2-5 seconds | ~5-30 seconds |
| **Quality** | Excellent | Good to Very Good |
| **Privacy** | Data sent to cloud | 100% local |
| **Internet** | Required | Not required |
| **Setup** | API key | Install + download model |

---

## Troubleshooting

### Ollama not responding

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve
```

### Model not found

```bash
# List installed models
ollama list

# Pull the model
ollama pull llama3:8b
```

### JSON parsing fails

Local models sometimes add extra text. The code already handles this with regex:

```python
json_match = re.search(r'\{[\s\S]*\}', response_text)
```

If issues persist, try a more capable model:

```python
response = ollama.chat(
    model='llama3:70b',  # Larger model, better JSON compliance
    messages=[{"role": "user", "content": prompt}]
)
```

### Slow performance

- Use a smaller model: `llama3:8b` instead of `llama3:70b`
- On Apple Silicon, Ollama uses GPU automatically
- On Linux with NVIDIA, ensure CUDA is configured

---

## Testing the Swap

```bash
# 1. Install Ollama and pull model
brew install ollama
ollama pull llama3:8b
ollama serve  # In separate terminal

# 2. Test Ollama directly
python3 -c "
import ollama
r = ollama.chat(model='llama3:8b', messages=[{'role': 'user', 'content': 'Say hello'}])
print(r['message']['content'])
"

# 3. Run video edit (after code changes)
python3 execution/simple_video_edit.py \
    --video .tmp/test.mp4 \
    --title "Test Video" \
    --no-upload
```
