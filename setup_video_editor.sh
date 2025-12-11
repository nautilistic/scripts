#!/bin/bash
#
# Video Editor Setup Script
# For Arch Linux
#
# Installs all dependencies needed to run simple_video_edit.py:
#   - FFmpeg (video processing)
#   - Whisper (transcription)
#   - Ollama (metadata generation)
#   - Python packages
#
# Usage:
#   ./setup_video_editor.sh [OPTIONS]
#
# Options:
#   --faster-whisper   Use faster-whisper instead of openai-whisper
#   --skip-ollama      Skip Ollama setup (if already installed)
#   --skip-whisper     Skip Whisper installation
#   --skip-ffmpeg      Skip FFmpeg installation
#   --help             Show this help message
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_FASTER_WHISPER=false
SKIP_OLLAMA=false
SKIP_WHISPER=false
SKIP_FFMPEG=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

check_arch() {
    if ! command -v pacman &> /dev/null; then
        error "This script is designed for Arch Linux (pacman not found)"
        echo "For other distros, install manually:"
        echo "  - FFmpeg"
        echo "  - Python 3.8+"
        echo "  - pip install openai-whisper (or faster-whisper)"
        echo "  - Ollama: https://ollama.com"
        exit 1
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --faster-whisper)
            USE_FASTER_WHISPER=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --skip-whisper)
            SKIP_WHISPER=true
            shift
            ;;
        --skip-ffmpeg)
            SKIP_FFMPEG=true
            shift
            ;;
        --help)
            head -22 "$0" | tail -18
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main
# =============================================================================

header "Video Editor Setup"
echo "This will install dependencies for simple_video_edit.py"
echo ""

if [[ $EUID -eq 0 ]]; then
    warn "Running as root. On a normal system, run as regular user."
fi

check_arch

# -----------------------------------------------------------------------------
# Step 1: System packages (FFmpeg)
# -----------------------------------------------------------------------------

header "Step 1: FFmpeg"

if [[ "$SKIP_FFMPEG" == "true" ]]; then
    info "Skipping FFmpeg (--skip-ffmpeg)"
elif command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -1)
    success "FFmpeg already installed: $FFMPEG_VERSION"
else
    info "Installing FFmpeg..."
    sudo pacman -S --noconfirm --needed ffmpeg
    success "FFmpeg installed"
fi

# Verify ffprobe too
if command -v ffprobe &> /dev/null; then
    success "FFprobe available"
else
    warn "FFprobe not found (should come with FFmpeg)"
fi

# -----------------------------------------------------------------------------
# Step 2: Python packages
# -----------------------------------------------------------------------------

header "Step 2: Python Packages"

# Ensure pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    info "Installing python-pip..."
    sudo pacman -S --noconfirm python-pip
fi

PIP_CMD="pip"
command -v pip3 &> /dev/null && PIP_CMD="pip3"

# Core packages
info "Installing core Python packages..."
$PIP_CMD install --user --break-system-packages \
    python-dotenv \
    requests \
    2>/dev/null || \
$PIP_CMD install --user python-dotenv requests

success "Core packages installed (python-dotenv, requests)"

# -----------------------------------------------------------------------------
# Step 3: Whisper (transcription)
# -----------------------------------------------------------------------------

header "Step 3: Whisper (Transcription)"

if [[ "$SKIP_WHISPER" == "true" ]]; then
    info "Skipping Whisper (--skip-whisper)"
elif [[ "$USE_FASTER_WHISPER" == "true" ]]; then
    info "Installing faster-whisper..."

    # faster-whisper needs these
    if python3 -c "import faster_whisper" &> /dev/null; then
        success "faster-whisper already installed"
    else
        $PIP_CMD install --user --break-system-packages faster-whisper 2>/dev/null || \
        $PIP_CMD install --user faster-whisper
        success "faster-whisper installed"
    fi

    echo ""
    warn "NOTE: simple_video_edit.py uses openai-whisper by default."
    echo "      You may need to modify transcribe_video() to use faster-whisper."
    echo "      See: https://github.com/guillaumekln/faster-whisper"
else
    info "Installing openai-whisper..."

    if python3 -c "import whisper" &> /dev/null; then
        success "openai-whisper already installed"
    else
        # whisper needs torch
        $PIP_CMD install --user --break-system-packages openai-whisper 2>/dev/null || \
        $PIP_CMD install --user openai-whisper
        success "openai-whisper installed"
    fi
fi

# -----------------------------------------------------------------------------
# Step 4: Ollama (LLM for metadata)
# -----------------------------------------------------------------------------

header "Step 4: Ollama (LLM)"

if [[ "$SKIP_OLLAMA" == "true" ]]; then
    info "Skipping Ollama setup (--skip-ollama)"
elif [[ -f "$SCRIPT_DIR/tests/setup_and_test_ollama.sh" ]]; then
    info "Running Ollama setup script..."
    bash "$SCRIPT_DIR/tests/setup_and_test_ollama.sh" --unit-only
else
    # Fallback: install ollama manually
    info "Installing Ollama..."

    if command -v ollama &> /dev/null; then
        success "Ollama already installed"
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Install Python package
    $PIP_CMD install --user --break-system-packages ollama 2>/dev/null || \
    $PIP_CMD install --user ollama

    success "Ollama installed"
    echo ""
    warn "You still need to:"
    echo "  1. Start Ollama: ollama serve"
    echo "  2. Pull a model: ollama pull llama3:8b"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

header "Setup Complete!"

echo "Installed components:"
command -v ffmpeg &> /dev/null && echo "  [x] FFmpeg" || echo "  [ ] FFmpeg (missing)"
command -v ffprobe &> /dev/null && echo "  [x] FFprobe" || echo "  [ ] FFprobe (missing)"
python3 -c "import whisper" &> /dev/null && echo "  [x] openai-whisper" || \
    (python3 -c "import faster_whisper" &> /dev/null && echo "  [x] faster-whisper" || echo "  [ ] Whisper (missing)")
python3 -c "import ollama" &> /dev/null && echo "  [x] Ollama Python package" || echo "  [ ] Ollama package (missing)"
command -v ollama &> /dev/null && echo "  [x] Ollama CLI" || echo "  [ ] Ollama CLI (missing)"

echo ""
echo "Usage:"
echo "  # Make sure Ollama is running"
echo "  ollama serve  # (in another terminal, or use systemctl)"
echo ""
echo "  # Edit a video"
echo "  python3 execution/simple_video_edit.py \\"
echo "      --video /path/to/video.mp4 \\"
echo "      --title \"My Video\" \\"
echo "      --no-upload"
echo ""

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    success "Ollama service is running"
else
    warn "Ollama service is NOT running"
    echo "  Start it with: ollama serve"
    echo "  Or: systemctl start ollama"
fi
