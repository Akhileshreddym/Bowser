import asyncio
import time
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core.vision import VisionTracker
from core.agents import (
    patient_agent, get_patient_record,
    clinical_agent, analyze_pose_data, reset_clinical_state, _clinical_state,
    assistant_agent, format_therapy_context,
    pacer_agent, calculate_shadow_drive
)
from core.audio import generate_bowser_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components
tracker = VisionTracker(camera_index="off")

# ADK Session & Runner for the Therapy Assistant (the only agent that needs LLM inference)
session_service = InMemorySessionService()
assistant_runner = Runner(
    agent=assistant_agent,
    app_name="therapy_app",
    session_service=session_service,
)

# Global State
class AppState:
    def __init__(self):
        self.patient_info = get_patient_record("patient_123")
        self.clinical_state = {
            "active_exercise": "None",
            "reps_count": 0,
            "form_quality": "unknown",
            "is_in_rep": False
        }
        self.director_command = "[STOP]"
        self.director_dialogue = "Welcome! Scan a bracelet to begin."
        self.roomba_output = "drive 0,0"
        self.latest_audio_url = ""
        self.is_active = False
        self.kid_speech = ""
        self.session_id = "session_default"

state = AppState()
active_connections: list[WebSocket] = []

async def broadcast_state():
    if not active_connections:
        return
    message = json.dumps({
        "patient": state.patient_info,
        "clinical": state.clinical_state,
        "command": state.director_command,
        "dialogue": state.director_dialogue,
        "roomba": state.roomba_output,
        "audio_url": state.latest_audio_url
    })
    state.latest_audio_url = ""
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send to websocket: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        await broadcast_state()
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            if cmd.get("action") == "scan_bracelet":
                patient_id = cmd.get("patient_id", "patient_123")
                state.patient_info = get_patient_record(patient_id)
                state.is_active = True
                # Reset clinical tracking for new patient
                reset_clinical_state()
                state.clinical_state = _clinical_state.copy()
                # Create a fresh ADK session for this patient
                state.session_id = f"session_{patient_id}_{int(time.time())}"
                session = await session_service.create_session(
                    app_name="therapy_app",
                    user_id="therapist",
                    session_id=state.session_id,
                )
                asyncio.create_task(trigger_assistant_update())
            elif cmd.get("action") == "stop":
                state.is_active = False
                state.director_dialogue = "Session stopped."
                state.roomba_output = "drive 0,0"
                await broadcast_state()
            elif cmd.get("action") == "kid_speech":
                state.kid_speech = cmd.get("text")
                if state.is_active:
                    asyncio.create_task(trigger_assistant_update())
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def trigger_assistant_update():
    """Invokes the ADK Therapy Assistant Agent via the Runner."""
    logger.info("Triggering ADK assistant agent...")

    # Build the context message for the LLM
    info = state.patient_info
    context = (
        f"Patient: {info.get('name', 'Unknown')}, "
        f"Goal: {info.get('therapy_goal', 'General')}, "
        f"Progress: {state.clinical_state.get('reps_count', 0)}/{info.get('target_reps', 10)}"
    )
    if state.kid_speech:
        context += f". Kid just said: '{state.kid_speech}'"

    state.kid_speech = ""

    try:
        # Run the ADK agent
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=context)]
        )
        
        final_response = ""
        async for event in assistant_runner.run_async(
            user_id="therapist",
            session_id=state.session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text

        if final_response:
            # Parse command and dialogue from the response
            reply = final_response.strip()
            command = "[STOP]"
            dialogue = reply
            if reply.startswith("[") and "]" in reply:
                end_idx = reply.find("]")
                command = reply[:end_idx+1].strip()
                dialogue = reply[end_idx+1:].strip()

            state.director_command = command
            state.director_dialogue = dialogue

    except Exception as e:
        logger.error(f"ADK Agent Error: {e}")
        state.director_dialogue = "I'm having trouble connecting right now."
        state.director_command = "[STOP]"

    # Generate TTS
    if state.director_dialogue:
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(
            None, generate_bowser_audio, state.director_dialogue
        )
        state.latest_audio_url = audio_url

    await broadcast_state()

def generate_video():
    """Generator for MJPEG streaming."""
    last_reps = 0
    while True:
        frame_bytes, vision_data = tracker.process_frame()

        # Call the clinical tool function directly (deterministic, no LLM needed)
        goal = state.patient_info.get("therapy_goal", "Squats")
        new_clinical_state = analyze_pose_data(
            squat_angle=vision_data.get("squat_angle", 180),
            arm_angle=vision_data.get("arm_angle", 180),
            therapy_goal=goal
        )

        if new_clinical_state["reps_count"] > last_reps:
            last_reps = new_clinical_state["reps_count"]
            state.clinical_state = new_clinical_state.copy()
            if state.is_active and main_loop is not None:
                asyncio.run_coroutine_threadsafe(trigger_assistant_update(), main_loop)

        state.clinical_state = new_clinical_state.copy()

        if state.is_active:
            # Call the pacer tool function directly (deterministic, no LLM needed)
            result = calculate_shadow_drive(
                center_x=vision_data.get("center_x", 0.5),
                depth=vision_data.get("distance_depth", 128)
            )
            new_roomba_out = result["drive_command"]
            if new_roomba_out != state.roomba_output:
                state.roomba_output = new_roomba_out

        if frame_bytes is None:
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.post("/api/camera/{index}")
def set_camera(index: str):
    tracker.set_camera(index)
    return {"status": "ok", "camera_index": index}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

async def periodic_broadcast():
    while True:
        await asyncio.sleep(1.0)
        await broadcast_state()

main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(periodic_broadcast())
    # Create a default session
    await session_service.create_session(
        app_name="therapy_app",
        user_id="therapist",
        session_id=state.session_id,
    )
