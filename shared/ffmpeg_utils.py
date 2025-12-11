#!/usr/bin/env python3
"""
Shared FFmpeg Utilities

Common video encoding utilities used across multiple projects.
"""

import subprocess

# Video encoding settings
# Hardware encoding (h264_videotoolbox) is 5-10x faster on Apple Silicon
# Falls back to software (libx264) if hardware unavailable
HARDWARE_ENCODER = "h264_videotoolbox"
SOFTWARE_ENCODER = "libx264"
HARDWARE_BITRATE = "10M"  # 10 Mbps for hardware encoding (no CRF support)
SOFTWARE_CRF = "18"  # CRF 18 for software encoding (high quality)


def check_hardware_encoder_available() -> bool:
    """Check if h264_videotoolbox hardware encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        return "h264_videotoolbox" in result.stdout
    except Exception:
        return False


def get_video_encoder_args(use_hardware: bool = True) -> list[str]:
    """
    Get FFmpeg video encoder arguments.

    Returns args for hardware encoding if available and requested,
    otherwise falls back to software encoding.
    """
    if use_hardware and check_hardware_encoder_available():
        # Hardware encoding: use bitrate (no CRF support)
        # 10 Mbps is good for 1080p, scales well for 4K
        return ["-c:v", HARDWARE_ENCODER, "-b:v", HARDWARE_BITRATE]
    else:
        # Software encoding: use CRF for quality-based encoding
        return ["-c:v", SOFTWARE_ENCODER, "-preset", "fast", "-crf", SOFTWARE_CRF]


# Cache the encoder check result (only check once per run)
_hardware_encoder_available = None

def get_cached_encoder_args() -> list[str]:
    """Get encoder args with cached hardware availability check."""
    global _hardware_encoder_available
    if _hardware_encoder_available is None:
        _hardware_encoder_available = check_hardware_encoder_available()
        if _hardware_encoder_available:
            print(f"Hardware encoding enabled (h264_videotoolbox)")
        else:
            print(f"Using software encoding (libx264)")
    return get_video_encoder_args(_hardware_encoder_available)
