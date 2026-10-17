import math
import struct
import subprocess
import wave
from io import BytesIO

import pytest

from chhaaya.audio import to_ogg_opus


def tone(seconds: float = 1.0, rate: int = 22050) -> bytes:
    """A mono 16-bit WAV like the ones Sarvam returns."""
    out = BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        samples = (
            int(8000 * math.sin(2 * math.pi * 440 * n / rate))
            for n in range(int(seconds * rate))
        )
        wav.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return out.getvalue()


def test_wav_becomes_an_ogg_opus_file_that_decodes_to_the_same_length():
    ogg = to_ogg_opus(tone(1.0))

    assert ogg.startswith(b"OggS")
    assert b"OpusHead" in ogg[:100]
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-"],
        input=ogg,
        capture_output=True,
        check=True,
    ).stdout
    seconds = len(decoded) / 2 / 16000
    assert seconds == pytest.approx(1.0, abs=0.1)


def test_something_that_is_not_audio_fails_with_ffmpegs_message():
    with pytest.raises(RuntimeError, match="ffmpeg"):
        to_ogg_opus(b"this is not audio")
