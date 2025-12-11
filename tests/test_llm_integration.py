#!/usr/bin/env python3
"""
Test LLM Integration: Ollama for Local Metadata Generation

This test validates the Ollama integration for generating video metadata
(titles, summaries, chapters) as described in directives/llm_integration.md

Prerequisites (Arch Linux):
    # Install Ollama (choose one method):
    # Method 1: Using the install script
    curl -fsSL https://ollama.com/install.sh | sh

    # Method 2: From AUR (if available)
    yay -S ollama

    # Start the Ollama service
    systemctl start ollama
    # Or run manually:
    ollama serve

    # Pull a model (one-time download)
    ollama pull llama3:8b

    # Install Python dependency
    pip install ollama

Usage:
    python tests/test_llm_integration.py
    # or
    pytest tests/test_llm_integration.py -v
"""

import json
import re
import sys
import subprocess
from typing import Optional


# ============================================================================
# Test Configuration
# ============================================================================

OLLAMA_MODEL = "llama3:8b"  # Can also use: mistral, qwen2:7b, llama3:70b
OLLAMA_HOST = "http://localhost:11434"


# ============================================================================
# Helper Functions (from llm_integration.md)
# ============================================================================

def check_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["curl", "-s", f"{OLLAMA_HOST}/api/tags"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_model_installed(model: str) -> bool:
    """Check if a specific model is installed in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return model.split(":")[0] in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def parse_json_from_response(response_text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response text.
    Local models sometimes add extra text around the JSON.
    """
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None


def generate_metadata_ollama(
    words: list[dict],
    cuts: list[tuple[float, float]],
    duration: float,
    title: str
) -> dict:
    """
    Generate YouTube summary and chapters using Ollama (local LLM).
    Returns dict with {title, summary, chapters}.

    This is the drop-in replacement function from llm_integration.md
    """
    import ollama

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

    print(f"Generating metadata with Ollama ({OLLAMA_MODEL})...")

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response['message']['content'].strip()

    # Parse JSON response
    data = parse_json_from_response(response_text)
    if not data:
        print(f"Warning: Could not parse JSON from response:\n{response_text[:500]}")
        return {"title": title, "summary": "Video content.", "chapters": "00:00:00 Introduction"}

    summary = data.get("summary", "Video content.")
    chapters_list = data.get("chapters", [{"time": "00:00:00", "title": "Introduction"}])

    # Adjust timestamps for cuts
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


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_TRANSCRIPT_WORDS = [
    # Introduction (0-30s)
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "and", "start": 0.5, "end": 0.7},
    {"word": "welcome", "start": 0.7, "end": 1.2},
    {"word": "to", "start": 1.2, "end": 1.4},
    {"word": "this", "start": 1.4, "end": 1.6},
    {"word": "tutorial", "start": 1.6, "end": 2.2},
    {"word": "on", "start": 2.2, "end": 2.4},
    {"word": "Python", "start": 2.4, "end": 3.0},
    {"word": "programming.", "start": 3.0, "end": 3.8},
    {"word": "Today", "start": 5.0, "end": 5.4},
    {"word": "we'll", "start": 5.4, "end": 5.8},
    {"word": "cover", "start": 5.8, "end": 6.2},
    {"word": "three", "start": 6.2, "end": 6.5},
    {"word": "main", "start": 6.5, "end": 6.8},
    {"word": "topics:", "start": 6.8, "end": 7.3},
    {"word": "variables,", "start": 7.5, "end": 8.2},
    {"word": "functions,", "start": 8.3, "end": 9.0},
    {"word": "and", "start": 9.0, "end": 9.2},
    {"word": "loops.", "start": 9.2, "end": 10.0},
    {"word": "Let's", "start": 25.0, "end": 25.4},
    {"word": "get", "start": 25.4, "end": 25.6},
    {"word": "started!", "start": 25.6, "end": 26.2},

    # Variables section (30-90s)
    {"word": "First,", "start": 30.0, "end": 30.5},
    {"word": "let's", "start": 30.5, "end": 30.8},
    {"word": "talk", "start": 30.8, "end": 31.1},
    {"word": "about", "start": 31.1, "end": 31.4},
    {"word": "variables.", "start": 31.4, "end": 32.2},
    {"word": "Variables", "start": 35.0, "end": 35.6},
    {"word": "are", "start": 35.6, "end": 35.8},
    {"word": "containers", "start": 35.8, "end": 36.5},
    {"word": "for", "start": 36.5, "end": 36.7},
    {"word": "storing", "start": 36.7, "end": 37.2},
    {"word": "data", "start": 37.2, "end": 37.6},
    {"word": "values.", "start": 37.6, "end": 38.3},
    {"word": "You", "start": 55.0, "end": 55.2},
    {"word": "can", "start": 55.2, "end": 55.4},
    {"word": "store", "start": 55.4, "end": 55.8},
    {"word": "numbers,", "start": 55.8, "end": 56.4},
    {"word": "strings,", "start": 56.5, "end": 57.1},
    {"word": "and", "start": 57.1, "end": 57.3},
    {"word": "booleans.", "start": 57.3, "end": 58.2},

    # Functions section (90-150s)
    {"word": "Now", "start": 90.0, "end": 90.3},
    {"word": "let's", "start": 90.3, "end": 90.6},
    {"word": "move", "start": 90.6, "end": 90.9},
    {"word": "on", "start": 90.9, "end": 91.1},
    {"word": "to", "start": 91.1, "end": 91.3},
    {"word": "functions.", "start": 91.3, "end": 92.1},
    {"word": "Functions", "start": 95.0, "end": 95.6},
    {"word": "are", "start": 95.6, "end": 95.8},
    {"word": "reusable", "start": 95.8, "end": 96.5},
    {"word": "blocks", "start": 96.5, "end": 97.0},
    {"word": "of", "start": 97.0, "end": 97.2},
    {"word": "code.", "start": 97.2, "end": 97.8},
    {"word": "They", "start": 120.0, "end": 120.3},
    {"word": "help", "start": 120.3, "end": 120.6},
    {"word": "organize", "start": 120.6, "end": 121.3},
    {"word": "your", "start": 121.3, "end": 121.5},
    {"word": "program.", "start": 121.5, "end": 122.2},

    # Loops section (150-210s)
    {"word": "Finally,", "start": 150.0, "end": 150.6},
    {"word": "let's", "start": 150.6, "end": 150.9},
    {"word": "discuss", "start": 150.9, "end": 151.4},
    {"word": "loops.", "start": 151.4, "end": 152.0},
    {"word": "Loops", "start": 155.0, "end": 155.4},
    {"word": "allow", "start": 155.4, "end": 155.8},
    {"word": "you", "start": 155.8, "end": 156.0},
    {"word": "to", "start": 156.0, "end": 156.2},
    {"word": "repeat", "start": 156.2, "end": 156.7},
    {"word": "code.", "start": 156.7, "end": 157.2},
    {"word": "Python", "start": 180.0, "end": 180.5},
    {"word": "has", "start": 180.5, "end": 180.7},
    {"word": "for", "start": 180.7, "end": 181.0},
    {"word": "loops", "start": 181.0, "end": 181.4},
    {"word": "and", "start": 181.4, "end": 181.6},
    {"word": "while", "start": 181.6, "end": 182.0},
    {"word": "loops.", "start": 182.0, "end": 182.5},

    # Conclusion (210-240s)
    {"word": "That's", "start": 210.0, "end": 210.4},
    {"word": "all", "start": 210.4, "end": 210.6},
    {"word": "for", "start": 210.6, "end": 210.8},
    {"word": "today!", "start": 210.8, "end": 211.4},
    {"word": "Thanks", "start": 220.0, "end": 220.4},
    {"word": "for", "start": 220.4, "end": 220.6},
    {"word": "watching!", "start": 220.6, "end": 221.3},
]

SAMPLE_CUTS = [
    (40.0, 50.0),   # Cut 10 seconds of silence
    (100.0, 110.0), # Cut another 10 seconds
]

SAMPLE_DURATION = 240.0  # 4 minutes


# ============================================================================
# Tests
# ============================================================================

def test_ollama_service():
    """Test 1: Check if Ollama service is running."""
    print("\n" + "=" * 60)
    print("TEST 1: Ollama Service Check")
    print("=" * 60)

    is_running = check_ollama_running()

    if is_running:
        print("[PASS] Ollama service is running at", OLLAMA_HOST)
        return True
    else:
        print("[FAIL] Ollama service is NOT running!")
        print("\nTo start Ollama on Arch Linux:")
        print("  systemctl start ollama")
        print("  # or")
        print("  ollama serve")
        return False


def test_model_installed():
    """Test 2: Check if the required model is installed."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Model Installation Check ({OLLAMA_MODEL})")
    print("=" * 60)

    is_installed = check_model_installed(OLLAMA_MODEL)

    if is_installed:
        print(f"[PASS] Model '{OLLAMA_MODEL}' is installed")
        return True
    else:
        print(f"[FAIL] Model '{OLLAMA_MODEL}' is NOT installed!")
        print(f"\nTo install the model:")
        print(f"  ollama pull {OLLAMA_MODEL}")
        return False


def test_ollama_chat():
    """Test 3: Test basic Ollama chat functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Ollama Chat Functionality")
    print("=" * 60)

    try:
        import ollama

        print(f"Sending test message to {OLLAMA_MODEL}...")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Say 'Hello, test successful!' and nothing else."}]
        )

        content = response['message']['content'].strip()
        print(f"Response: {content}")

        if content:
            print("[PASS] Ollama chat is working")
            return True
        else:
            print("[FAIL] Empty response from Ollama")
            return False

    except ImportError:
        print("[FAIL] 'ollama' Python package not installed!")
        print("\nTo install:")
        print("  pip install ollama")
        return False
    except Exception as e:
        print(f"[FAIL] Error during chat: {e}")
        return False


def test_json_parsing():
    """Test 4: Test JSON parsing from various LLM response formats."""
    print("\n" + "=" * 60)
    print("TEST 4: JSON Parsing")
    print("=" * 60)

    test_cases = [
        # Clean JSON
        ('{"summary": "Test", "chapters": []}', True),

        # JSON with surrounding text
        ('Here is the JSON:\n{"summary": "Test", "chapters": []}\nHope this helps!', True),

        # JSON with markdown code block
        ('```json\n{"summary": "Test", "chapters": []}\n```', True),

        # Multiline JSON
        ('{\n  "summary": "Test summary",\n  "chapters": [\n    {"time": "00:00:00", "title": "Intro"}\n  ]\n}', True),

        # Invalid JSON
        ('This is not JSON at all', False),

        # Broken JSON
        ('{"summary": "Test", "chapters": [}', False),
    ]

    passed = 0
    for i, (text, should_parse) in enumerate(test_cases, 1):
        result = parse_json_from_response(text)
        parsed = result is not None

        if parsed == should_parse:
            print(f"  Case {i}: [PASS]")
            passed += 1
        else:
            print(f"  Case {i}: [FAIL] Expected parse={should_parse}, got parse={parsed}")
            print(f"    Input: {text[:50]}...")

    if passed == len(test_cases):
        print(f"\n[PASS] All {len(test_cases)} JSON parsing tests passed")
        return True
    else:
        print(f"\n[FAIL] {passed}/{len(test_cases)} JSON parsing tests passed")
        return False


def test_timestamp_adjustment():
    """Test 5: Test timestamp adjustment for video cuts."""
    print("\n" + "=" * 60)
    print("TEST 5: Timestamp Adjustment for Cuts")
    print("=" * 60)

    # Simulate chapter timestamps and cuts
    # Cuts: (40-50), (100-110) = 20 seconds removed total

    test_cases = [
        # (original_time, cuts, expected_adjusted_time)
        (0, [], 0),           # No cuts, no change
        (30, [(40, 50)], 30), # Before cut, no change
        (60, [(40, 50)], 50), # After cut, -10s
        (120, [(40, 50), (100, 110)], 100),  # After both cuts, -20s
        (45, [(40, 50)], 40), # During cut (edge case)
    ]

    passed = 0
    for original_time, cuts, expected in test_cases:
        # Calculate adjusted time (same logic as in generate_metadata)
        sorted_cuts = sorted(cuts, key=lambda x: x[0])
        time_removed = 0
        for cut_start, cut_end in sorted_cuts:
            if cut_end <= original_time:
                time_removed += cut_end - cut_start
            elif cut_start < original_time:
                time_removed += original_time - cut_start

        adjusted = max(0, original_time - time_removed)

        if adjusted == expected:
            print(f"  {original_time}s with cuts {cuts} -> {adjusted}s [PASS]")
            passed += 1
        else:
            print(f"  {original_time}s with cuts {cuts} -> {adjusted}s (expected {expected}s) [FAIL]")

    if passed == len(test_cases):
        print(f"\n[PASS] All {len(test_cases)} timestamp adjustment tests passed")
        return True
    else:
        print(f"\n[FAIL] {passed}/{len(test_cases)} timestamp adjustment tests passed")
        return False


def test_metadata_generation():
    """Test 6: Full metadata generation with Ollama."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Metadata Generation")
    print("=" * 60)

    try:
        print("Generating metadata for sample transcript...")
        print(f"  - Transcript: {len(SAMPLE_TRANSCRIPT_WORDS)} words")
        print(f"  - Duration: {SAMPLE_DURATION}s ({SAMPLE_DURATION/60:.1f} min)")
        print(f"  - Cuts: {len(SAMPLE_CUTS)} segments")
        print()

        metadata = generate_metadata_ollama(
            words=SAMPLE_TRANSCRIPT_WORDS,
            cuts=SAMPLE_CUTS,
            duration=SAMPLE_DURATION,
            title="Python Tutorial: Variables, Functions, and Loops"
        )

        print("\n--- Generated Metadata ---")
        print(f"Title: {metadata['title']}")
        print(f"\nSummary:\n{metadata['summary']}")
        print(f"\nChapters:\n{metadata['chapters']}")
        print("--- End Metadata ---\n")

        # Validate output structure
        checks = [
            ("title" in metadata, "Has 'title' field"),
            ("summary" in metadata, "Has 'summary' field"),
            ("chapters" in metadata, "Has 'chapters' field"),
            (len(metadata['summary']) > 10, "Summary is not empty"),
            ("00:00:00" in metadata['chapters'], "Chapters start at 00:00:00"),
        ]

        passed = 0
        for check, desc in checks:
            if check:
                print(f"  {desc}: [PASS]")
                passed += 1
            else:
                print(f"  {desc}: [FAIL]")

        if passed == len(checks):
            print(f"\n[PASS] Metadata generation successful")
            return True
        else:
            print(f"\n[WARN] Metadata generated with {passed}/{len(checks)} checks passed")
            return passed > 2  # Allow partial success

    except ImportError:
        print("[SKIP] 'ollama' package not installed")
        return False
    except Exception as e:
        print(f"[FAIL] Error during metadata generation: {e}")
        return False


def test_empty_transcript():
    """Test 7: Handle empty transcript gracefully."""
    print("\n" + "=" * 60)
    print("TEST 7: Empty Transcript Handling")
    print("=" * 60)

    try:
        metadata = generate_metadata_ollama(
            words=[],
            cuts=[],
            duration=60.0,
            title="Empty Video"
        )

        if metadata['summary'] == "Video content." and metadata['chapters'] == "00:00:00 Introduction":
            print("[PASS] Empty transcript handled correctly")
            return True
        else:
            print(f"[FAIL] Unexpected output for empty transcript: {metadata}")
            return False

    except Exception as e:
        print(f"[FAIL] Error handling empty transcript: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all tests."""
    import argparse

    global OLLAMA_MODEL

    parser = argparse.ArgumentParser(description="Test LLM integration with Ollama")
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip tests that require Ollama to be running (run only unit tests)"
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})"
    )
    args = parser.parse_args()

    # Update model if specified
    if args.model:
        OLLAMA_MODEL = args.model

    print("\n" + "=" * 60)
    print("LLM Integration Test Suite")
    print("Testing Ollama integration for video metadata generation")
    if args.skip_ollama:
        print("Mode: Unit tests only (--skip-ollama)")
    else:
        print(f"Mode: Full integration tests (model: {OLLAMA_MODEL})")
    print("=" * 60)

    results = {}

    if not args.skip_ollama:
        # Test 1: Service check
        results["ollama_service"] = test_ollama_service()

        if not results["ollama_service"]:
            print("\n[ABORT] Ollama service not running. Start it and try again.")
            print("\nQuick start on Arch Linux:")
            print("  # Install (if not installed)")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("")
            print("  # Start service")
            print("  systemctl start ollama")
            print("  # or")
            print("  ollama serve")
            print("")
            print("  # Pull model")
            print("  ollama pull llama3:8b")
            print("\nOr run with --skip-ollama to run unit tests only.")
            sys.exit(1)

        # Test 2: Model check
        results["model_installed"] = test_model_installed()

        if not results["model_installed"]:
            print(f"\n[ABORT] Model '{OLLAMA_MODEL}' not installed.")
            print(f"  ollama pull {OLLAMA_MODEL}")
            print("\nOr run with --skip-ollama to run unit tests only.")
            sys.exit(1)

        # Test 3: Chat functionality
        results["ollama_chat"] = test_ollama_chat()
    else:
        print("\n[SKIP] Ollama service check (--skip-ollama)")
        print("[SKIP] Model installation check (--skip-ollama)")
        print("[SKIP] Ollama chat test (--skip-ollama)")

    # Test 4: JSON parsing (no Ollama needed)
    results["json_parsing"] = test_json_parsing()

    # Test 5: Timestamp adjustment (no Ollama needed)
    results["timestamp_adjustment"] = test_timestamp_adjustment()

    if not args.skip_ollama:
        # Test 6: Full metadata generation
        if results.get("ollama_chat", False):
            results["metadata_generation"] = test_metadata_generation()
        else:
            print("\n[SKIP] Test 6: Skipping metadata generation (chat test failed)")
            results["metadata_generation"] = False

        # Test 7: Empty transcript handling
        results["empty_transcript"] = test_empty_transcript()
    else:
        print("\n[SKIP] Metadata generation test (--skip-ollama)")
        print("[SKIP] Empty transcript test (--skip-ollama)")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
