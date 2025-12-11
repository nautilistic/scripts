#!/bin/bash
#
# Ollama LLM Integration Setup & Test Script
# For Arch Linux
#
# This script:
#   1. Installs Ollama (if not installed)
#   2. Starts the Ollama service
#   3. Pulls the required model
#   4. Installs Python dependencies
#   5. Runs the integration tests
#
# Usage:
#   ./tests/setup_and_test_ollama.sh [OPTIONS]
#
# Options:
#   --model MODEL    Use a different model (default: llama3:8b)
#   --skip-install   Skip Ollama installation (assume already installed)
#   --unit-only      Only run unit tests (skip Ollama-dependent tests)
#   --help           Show this help message
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

MODEL="${MODEL:-llama3:8b}"
OLLAMA_HOST="http://localhost:11434"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This is fine for containers/CI, but on a normal system run as a regular user."
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

SKIP_INSTALL=false
UNIT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --unit-only)
            UNIT_ONLY=true
            shift
            ;;
        --help)
            head -25 "$0" | tail -20
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
# Main Script
# =============================================================================

header "Ollama LLM Integration Setup"
echo "Model: $MODEL"
echo "Project: $PROJECT_DIR"

check_root

# -----------------------------------------------------------------------------
# Step 1: Check/Install Ollama
# -----------------------------------------------------------------------------

header "Step 1: Ollama Installation"

if [[ "$UNIT_ONLY" == "true" ]]; then
    info "Skipping Ollama installation (--unit-only mode)"
elif command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    success "Ollama is already installed ($OLLAMA_VERSION)"
elif [[ "$SKIP_INSTALL" == "true" ]]; then
    error "Ollama not found and --skip-install was specified"
    exit 1
else
    info "Ollama not found. Installing..."

    # Check for different installation methods
    if command -v yay &> /dev/null; then
        info "Installing via yay (AUR)..."
        yay -S --noconfirm ollama-bin 2>/dev/null || {
            warn "AUR install failed, falling back to official installer"
            curl -fsSL https://ollama.com/install.sh | sh
        }
    elif command -v paru &> /dev/null; then
        info "Installing via paru (AUR)..."
        paru -S --noconfirm ollama-bin 2>/dev/null || {
            warn "AUR install failed, falling back to official installer"
            curl -fsSL https://ollama.com/install.sh | sh
        }
    else
        info "Installing via official installer..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    if command -v ollama &> /dev/null; then
        success "Ollama installed successfully"
    else
        error "Failed to install Ollama"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 2: Start Ollama Service
# -----------------------------------------------------------------------------

if [[ "$UNIT_ONLY" == "true" ]]; then
    info "Skipping Ollama service (--unit-only mode)"
else
    header "Step 2: Starting Ollama Service"

    # Check if already running
    if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
        success "Ollama service is already running"
    else
        info "Starting Ollama service..."

        # Try systemd first
        if systemctl is-enabled ollama &> /dev/null 2>&1; then
            info "Starting via systemd..."
            sudo systemctl start ollama
            sleep 2
        else
            # Start manually in background
            info "Starting ollama serve in background..."
            ollama serve &> /tmp/ollama.log &
            OLLAMA_PID=$!

            # Wait for it to be ready
            info "Waiting for Ollama to be ready..."
            for i in {1..30}; do
                if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
                    break
                fi
                sleep 1
                echo -n "."
            done
            echo ""
        fi

        # Verify it's running
        if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
            success "Ollama service is running"
        else
            error "Failed to start Ollama service"
            echo "Try running manually: ollama serve"
            exit 1
        fi
    fi

    # -----------------------------------------------------------------------------
    # Step 3: Pull Model
    # -----------------------------------------------------------------------------

    header "Step 3: Pulling Model ($MODEL)"

    # Check if model exists
    if ollama list 2>/dev/null | grep -q "^${MODEL%%:*}"; then
        success "Model '$MODEL' is already downloaded"
    else
        info "Downloading model '$MODEL'..."
        info "This may take a while depending on your internet connection..."
        echo ""

        if ollama pull "$MODEL"; then
            success "Model '$MODEL' downloaded successfully"
        else
            error "Failed to download model '$MODEL'"
            exit 1
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Step 4: Install Python Dependencies
# -----------------------------------------------------------------------------

header "Step 4: Python Dependencies"

# Check for pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    warn "pip not found. Installing python-pip..."
    sudo pacman -S --noconfirm python-pip
fi

PIP_CMD="pip"
command -v pip3 &> /dev/null && PIP_CMD="pip3"

# Check if ollama package is installed
if python3 -c "import ollama" &> /dev/null; then
    success "Python 'ollama' package is already installed"
else
    info "Installing Python 'ollama' package..."
    $PIP_CMD install --user ollama --break-system-packages 2>/dev/null || \
    $PIP_CMD install --user ollama || {
        error "Failed to install ollama Python package"
        echo "Try: pip install --user ollama"
        exit 1
    }
    success "Python 'ollama' package installed"
fi

# -----------------------------------------------------------------------------
# Step 5: Run Tests
# -----------------------------------------------------------------------------

header "Step 5: Running Tests"

cd "$PROJECT_DIR"

if [[ "$UNIT_ONLY" == "true" ]]; then
    info "Running unit tests only (--unit-only)..."
    python3 tests/test_llm_integration.py --skip-ollama
else
    info "Running full integration tests with model: $MODEL"
    python3 tests/test_llm_integration.py --model "$MODEL"
fi

TEST_EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

header "Setup Complete"

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    success "All tests passed!"
    echo ""
    echo "You can now use Ollama in your project:"
    echo ""
    echo "  import ollama"
    echo "  response = ollama.chat("
    echo "      model='$MODEL',"
    echo "      messages=[{'role': 'user', 'content': 'Hello!'}]"
    echo "  )"
    echo "  print(response['message']['content'])"
    echo ""
else
    warn "Some tests failed. Check output above for details."
fi

echo "Useful commands:"
echo "  ollama list          # List installed models"
echo "  ollama run $MODEL    # Interactive chat"
echo "  ollama serve         # Start service manually"
echo "  systemctl status ollama  # Check service status"
echo ""

exit $TEST_EXIT_CODE
