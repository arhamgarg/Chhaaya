"""Audio conversion with ffmpeg."""

import subprocess


def to_ogg_opus(wav: bytes) -> bytes:
    """Convert WAV to mono OGG/Opus, which WhatsApp shows as a voice note."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0"]
        + ["-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", "-f", "ogg"]
        + ["pipe:1"],
        input=wav,
        capture_output=True,
        timeout=30,  # seconds; a stuck ffmpeg would stall the only worker
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")
    return result.stdout
