"""Step 2: Audio transcription using faster-whisper (optional — skips if no audio/speech)."""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from app.config import get_settings


@dataclass
class TranscriptSegment:
    start: float  # seconds
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    language: str = ""
    duration_s: float = 0.0

    @property
    def has_speech(self) -> bool:
        return len(self.full_text.strip()) >= 50

    def text_for_range(self, start_s: float, end_s: float) -> str:
        """Get transcript text for a time range."""
        parts = []
        for seg in self.segments:
            if seg.end >= start_s and seg.start <= end_s:
                parts.append(seg.text)
        return " ".join(parts)

    def formatted_transcript(self, start_s: float = 0, end_s: float = float("inf")) -> str:
        """Format transcript with timestamps for LLM consumption."""
        lines = []
        for seg in self.segments:
            if seg.end >= start_s and seg.start <= end_s:
                mm_start = int(seg.start // 60)
                ss_start = int(seg.start % 60)
                lines.append(f"[{mm_start}:{ss_start:02d}] {seg.text.strip()}")
        return "\n".join(lines)


class Transcriber:
    """Extract and transcribe audio from video files using faster-whisper."""

    def __init__(self):
        settings = get_settings()
        self._model_size = getattr(settings, "WHISPER_MODEL", "medium")
        self._device = getattr(settings, "WHISPER_DEVICE", "auto")
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self._device
            if device == "auto":
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(f"Loading Whisper model: {self._model_size} on {device} ({compute_type})")
            self._model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute_type,
            )
        return self._model

    def _extract_audio(self, video_path: str) -> str | None:
        """Extract audio from video to a temporary WAV file. Returns path or None."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vn",                    # no video
                    "-acodec", "pcm_s16le",   # 16-bit PCM
                    "-ar", "16000",           # 16kHz (what Whisper expects)
                    "-ac", "1",               # mono
                    tmp.name,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning(f"ffmpeg audio extraction failed: {result.stderr[:200]}")
                os.unlink(tmp.name)
                return None

            # Check if file has actual content (not just header)
            size = os.path.getsize(tmp.name)
            if size < 10000:  # less than 10KB = probably silence
                logger.info("Audio track too small, likely silent video")
                os.unlink(tmp.name)
                return None

            return tmp.name

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Audio extraction error: {e}")
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            return None

    def transcribe(self, video_path: str) -> TranscriptionResult:
        """
        Transcribe audio from a video file.
        Returns TranscriptionResult (empty if no audio/speech).
        """
        result = TranscriptionResult()

        # Step 1: Extract audio
        audio_path = self._extract_audio(video_path)
        if audio_path is None:
            logger.info("No audio found — proceeding without transcript")
            return result

        try:
            # Step 2: Transcribe with faster-whisper
            model = self._get_model()
            segments_iter, info = model.transcribe(
                audio_path,
                beam_size=5,
                language=None,           # auto-detect
                vad_filter=True,         # skip silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )

            result.language = info.language
            result.duration_s = info.duration

            logger.info(f"Whisper: language={info.language}, "
                        f"prob={info.language_probability:.2f}, "
                        f"duration={info.duration:.0f}s")

            segments = []
            full_parts = []
            for seg in segments_iter:
                ts = TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                )
                segments.append(ts)
                full_parts.append(ts.text)

            result.segments = segments
            result.full_text = " ".join(full_parts)

            logger.info(f"Transcribed {len(segments)} segments, "
                        f"{len(result.full_text)} chars")

        except Exception as e:
            logger.error(f"Transcription failed: {e}")

        finally:
            # Clean up temp audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        return result
