import os
import uuid
import logging
from elevenlabs.client import ElevenLabs
from core.config import ELEVENLABS_API_KEY

logger = logging.getLogger(__name__)

# Initialize ElevenLabs Client
if ELEVENLABS_API_KEY:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
else:
    client = None

def generate_bowser_audio(text: str) -> str:
    """
    Generates TTS and saves to static audio folder. Returns the file path.
    """
    if not client:
        return ""
        
    try:
        # Generate audio using the "Brian" voice, or another default if needed.
        # Open router might use a distinct voice ID. For MVP we just use an ID that's close or default 'Rachel'/'Drew'
        # 'JBFqnCBcg6Bd4IY78fC7' is the ID for a known voice or we can just pass a string of a name like 'Drew' or 'Adam'
        audio_generator = client.generate(
            # "Adam" or custom voice ID for Bowser
            voice="pNInz6obpgDQGcFmaJgB",
            text=text,
            model="eleven_multilingual_v2"
        )
        
        # Save to static/audio directory
        os.makedirs("static/audio", exist_ok=True)
        filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join("static", "audio", filename)
        
        with open(filepath, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
                
        return f"/static/audio/{filename}"
    except Exception as e:
        logger.error(f"ElevenLabs TTS Error: {e}")
        return ""
