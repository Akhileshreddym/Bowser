import os
from dotenv import load_dotenv

load_dotenv()

# Google ADK uses GOOGLE_API_KEY env var automatically
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
