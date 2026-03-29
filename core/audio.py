import uuid
import logging
from pathlib import Path

import httpx

from core.config import ELEVENLABS_API_KEY

logger = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_AUDIO_DIR = APP_ROOT / "static" / "audio"
DEFAULT_VOICE_ID = "bh4qskdfSl83na9IzVGC"
MODEL_FALLBACKS = (
    "eleven_multilingual_v2",
    "eleven_turbo_v2_5",
    "eleven_monolingual_v1",
)
VOICE_SETTINGS = {
    # Keeps cloned voice identity/accent strong while staying expressive.
    "stability": 0.45,
    "similarity_boost": 0.9,
    "style": 0.35,
    "use_speaker_boost": True,
}


def _tts_request(text: str, voice_id: str, model_id: str) -> bytes:
    """Calls ElevenLabs directly and returns mp3 bytes."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": VOICE_SETTINGS,
    }
    with httpx.Client(timeout=45.0) as client:
        response = client.post(url, params={"output_format": "mp3_44100_128"}, headers=headers, json=payload)
        response.raise_for_status()
        return response.content


def generate_bowser_audio(text: str, voice_id: str = DEFAULT_VOICE_ID) -> tuple[str, str]:
    """
    Generates ElevenLabs TTS and writes it under static/audio.
    Returns tuple: (audio_url, error_message).
    """
    if not ELEVENLABS_API_KEY:
        return "", "ELEVENLABS_API_KEY missing"

    clean_text = (text or "").strip()
    if not clean_text:
        return "", "No text provided for TTS"

    STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    file_path = STATIC_AUDIO_DIR / filename

    # Force the custom cloned voice so accent/character remains consistent.
    candidate_voices = [DEFAULT_VOICE_ID]

    last_error = "Unknown ElevenLabs error"
    for candidate_voice in candidate_voices:
        for model_id in MODEL_FALLBACKS:
            try:
                audio_bytes = _tts_request(clean_text, candidate_voice, model_id)
                if not audio_bytes:
                    raise RuntimeError("Empty audio payload from ElevenLabs")
                file_path.write_bytes(audio_bytes)
                cache_bust = uuid.uuid4().hex[:8]
                return f"/static/audio/{filename}?v={cache_bust}", ""
            except Exception as exc:
                last_error = f"{candidate_voice}/{model_id}: {exc}"
                logger.warning(f"TTS attempt failed [{candidate_voice}/{model_id}]: {exc}")

    logger.error(f"ElevenLabs TTS failed after retries: {last_error}")
    return "", last_error
