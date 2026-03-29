import asyncio
import time
import json
import logging
import os
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
    pacer_agent, calculate_shadow_drive, translate_intent_to_drive
)
from core.audio import generate_bowser_audio
from core.hardware import send_command, send_stop, drive_string_to_command, configure_esp32

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
tracker = VisionTracker(camera_index=0) # Default to first camera

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
        self.latest_audio_error = ""
        self.latest_audio_event_id = 0
        self.is_active = False
        self.target_locked = False
        self.follow_enabled = True
        self.kid_speech = ""
        self.session_id = "session_default"
        self.last_audio_time = 0.0
        self.reset_video_reps = False  # Flag to tell video thread to reset last_reps
        self.session_started_at = time.time()
        self.session_completed_at = 0.0
        self.agent_trace = []
        self.agent_status = {
            "patient_agent": {"events": 0, "last": "idle", "status": "ready"},
            "clinical_agent": {"events": 0, "last": "idle", "status": "ready"},
            "director_agent": {"events": 0, "last": "idle", "status": "ready"},
            "pacer_agent": {"events": 0, "last": "idle", "status": "ready"},
            "audio_agent": {"events": 0, "last": "idle", "status": "ready"},
        }
        self.llm_failures = 0
        self.last_llm_error = ""
        self.llm_inflight = 0
        self.last_esp32_cmd = 4
        self.last_esp32_cmd_at = 0.0

state = AppState()
active_connections: list[WebSocket] = []
assistant_slots = asyncio.Semaphore(2)


def _safe_ratio(numer: float, denom: float) -> float:
    return round((numer / denom) if denom else 0.0, 3)


def _session_summary() -> dict:
    target_reps = int(state.patient_info.get("target_reps", 10) or 10)
    reps = int(state.clinical_state.get("reps_count", 0) or 0)
    elapsed = max(0, int(time.time() - state.session_started_at))
    completion = min(1.0, _safe_ratio(reps, target_reps))
    completed = reps >= target_reps and target_reps > 0
    if completed and state.session_completed_at == 0.0:
        state.session_completed_at = time.time()

    return {
        "reps": reps,
        "target_reps": target_reps,
        "completion": completion,
        "elapsed_seconds": elapsed,
        "completed": completed,
    }


def _health_snapshot() -> dict:
    return {
        "vision": True,
        "audio_ready": bool(os.getenv("ELEVENLABS_API_KEY", "").strip()),
        "llm_ready": bool(os.getenv("GOOGLE_API_KEY", "").strip()),
        "robot_follow_enabled": state.follow_enabled,
        "llm_inflight": state.llm_inflight,
        "llm_failures": state.llm_failures,
    }


def record_agent_event(agent: str, message: str, status: str = "ok"):
    slot = state.agent_status.setdefault(agent, {"events": 0, "last": "idle", "status": "ready"})
    slot["events"] += 1
    slot["last"] = message[:120]
    slot["status"] = status
    state.agent_trace.append({
        "t": int(time.time()),
        "agent": agent,
        "message": message[:180],
        "status": status,
    })
    state.agent_trace = state.agent_trace[-25:]

async def broadcast_state():
    if not active_connections:
        return
    message = json.dumps({
        "patient": state.patient_info,
        "clinical": state.clinical_state,
        "command": state.director_command,
        "dialogue": state.director_dialogue,
        "roomba": state.roomba_output,
        "summary": _session_summary(),
        "health": _health_snapshot(),
        "agent_status": state.agent_status,
        "agent_trace": state.agent_trace,
        "audio_url": state.latest_audio_url,
        "audio_error": state.latest_audio_error,
        "audio_event_id": state.latest_audio_event_id,
        "target_locked": state.target_locked,
        "is_active": state.is_active
    })
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send to websocket: {e}")


def publish_audio_event(audio_url: str, audio_error: str = ""):
    """Publishes a new audio event for frontend playback/debugging."""
    state.latest_audio_url = audio_url or ""
    state.latest_audio_error = audio_error or ""
    state.latest_audio_event_id += 1
    if audio_error:
        record_agent_event("audio_agent", f"TTS failed: {audio_error}", status="degraded")
    else:
        record_agent_event("audio_agent", "TTS generated and queued")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    record_agent_event("patient_agent", "UI client connected")
    try:
        await broadcast_state()
        while True:
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
            except json.JSONDecodeError:
                record_agent_event("patient_agent", "Invalid websocket payload", status="degraded")
                continue
            if cmd.get("action") in {"scan_bracelet", "scan_qr", "scan_profile"}:
                patient_id = cmd.get("patient_id")
                if not patient_id:
                     patient_id = cmd.get("id") # Fallback for different JSON keys
                
                if patient_id:
                    logger.info(f"══════ SWITCHING PATIENT TO: {patient_id} ══════")
                    state.patient_info = get_patient_record(patient_id)
                    record_agent_event("patient_agent", f"Loaded profile '{patient_id}'")
                state.is_active = True
                state.target_locked = False

                # Reset clinical tracking for new patient
                reset_clinical_state()
                state.clinical_state = _clinical_state.copy()
                state.reset_video_reps = True  # Tell video thread to reset its counter

                # CRITICAL: Reset TTS throttle so new patient greeting ALWAYS speaks
                state.last_audio_time = 0.0
                state.latest_audio_url = ""
                state.latest_audio_error = ""
                state.session_started_at = time.time()
                state.session_completed_at = 0.0

                patient_name = state.patient_info.get('name', 'Unknown')
                therapy_goal = state.patient_info.get('therapy_goal', 'General')
                normalized_patient = patient_id or "unknown"
                state.session_id = f"session_{normalized_patient}_{int(time.time())}"
                await session_service.create_session(
                    app_name="therapy_app",
                    user_id="therapist",
                    session_id=state.session_id,
                )
                record_agent_event("director_agent", "New ADK session created")

                greeting = await generate_human_greeting(state.session_id, patient_name, therapy_goal)
                if not greeting:
                    greeting = f"Hey {patient_name}, ready to crush {therapy_goal} together?"

                state.director_dialogue = greeting
                state.director_command = "[STOP]"

                # Immediate UI update with new patient data
                await broadcast_state()

                # Generate TTS for the greeting RIGHT NOW (don't wait for next rep)
                voice_id = state.patient_info.get("voice_id", "bh4qskdfSl83na9IzVGC")
                logger.info(f"Generating greeting TTS for {patient_name} with voice {voice_id}")
                loop = asyncio.get_event_loop()
                audio_url, audio_error = await loop.run_in_executor(
                    None, generate_bowser_audio, greeting, voice_id
                )
                publish_audio_event(audio_url, audio_error)
                if audio_error:
                    logger.warning(f"Greeting TTS error: {audio_error}")
                state.last_audio_time = time.time()
                await broadcast_state()

                # Don't immediately call assistant — let the greeting play first
            elif cmd.get("action") == "stop":
                state.is_active = False
                state.director_dialogue = "Session stopped."
                state.roomba_output = "drive 0,0"
                record_agent_event("director_agent", "Session stopped")
                await broadcast_state()
            elif cmd.get("action") == "kid_speech":
                state.kid_speech = cmd.get("text")
                if state.is_active:
                    record_agent_event("director_agent", "Received kid speech input")
                    asyncio.create_task(trigger_assistant_update())
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        record_agent_event("patient_agent", "UI client disconnected", status="ready")

async def trigger_assistant_update():
    """Invokes the ADK Therapy Assistant Agent via the Runner."""
    # Snapshot patient info at invocation time to prevent race conditions
    current_patient = state.patient_info.copy()
    current_session = state.session_id
    
    logger.info(f"Triggering ADK assistant agent for {current_patient.get('name', 'Unknown')}...")
    record_agent_event("director_agent", "Preparing ADK prompt")

    # Build the context message for the LLM
    context = (
        f"CURRENT PATIENT NAME: {current_patient.get('name', 'Unknown')}. "
        f"EXERCISE GOAL: {current_patient.get('therapy_goal', 'General')}. "
        f"PROGRESS: {state.clinical_state.get('reps_count', 0)}/{current_patient.get('target_reps', 10)} reps. "
        f"IMPORTANT: Address the child by their name {current_patient.get('name', 'Unknown')} and their specific exercise."
    )
    if state.kid_speech:
        context += f" The child just said: '{state.kid_speech}'"
        state.kid_speech = "" # Clear after consuming

    final_response = ""
    try:
        async with assistant_slots:
            state.llm_inflight += 1
            # Run the ADK agent
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=context)]
            )
            record_agent_event("director_agent", "Calling ADK runner (parallel slot acquired)")
            async for event in assistant_runner.run_async(
                user_id="therapist",
                session_id=current_session,
                new_message=content,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text
            record_agent_event("director_agent", "ADK runner response complete")

        # If patient changed while we were waiting for LLM, discard
        if state.session_id != current_session:
            logger.warning(f"Patient changed during LLM call — discarding stale response for {current_patient.get('name')}")
            record_agent_event("director_agent", "Discarded stale ADK response", status="degraded")
            return

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
            logger.info(f"LLM response for {current_patient.get('name')}: {dialogue[:80]}...")
            record_agent_event("director_agent", f"Command {command} with dialogue")
        else:
            state.director_command = "[STOP]"
            state.director_dialogue = f"Keep going {current_patient.get('name', 'champ')}! You are doing great."
            record_agent_event("director_agent", "Empty ADK response, used deterministic fallback", status="degraded")

    except Exception as e:
        logger.error(f"ADK Agent Error: {e}")
        state.llm_failures += 1
        state.last_llm_error = str(e)
        state.director_dialogue = f"I'm still with you, {current_patient.get('name', 'champ')}! Keep moving!"
        state.director_command = "[STOP]"
        record_agent_event("director_agent", f"ADK error fallback: {e}", status="degraded")
    finally:
        state.llm_inflight = max(0, state.llm_inflight - 1)

    # Generate TTS with 4s throttle
    if state.director_dialogue:
        current_time = time.time()
        if current_time - state.last_audio_time < 4.0:
            logger.info(f"ElevenLabs Throttle: Skipping voice generation (only {current_time - state.last_audio_time:.1f}s since last)")
            await broadcast_state()
            return

        state.last_audio_time = current_time
        voice_id = state.patient_info.get("voice_id", "bh4qskdfSl83na9IzVGC")
        loop = asyncio.get_event_loop()
        audio_url, audio_error = await loop.run_in_executor(
            None, generate_bowser_audio, state.director_dialogue, voice_id
        )
        publish_audio_event(audio_url, audio_error)
        if audio_error:
            logger.warning(f"Assistant TTS error: {audio_error}")

    await broadcast_state()


async def generate_human_greeting(session_id: str, patient_name: str, therapy_goal: str) -> str:
    """Generate a warmer greeting line via Gemini/ADK."""
    prompt = (
        f"Create one warm, natural greeting for a child named {patient_name}. "
        f"They are about to do {therapy_goal}. "
        "Keep it to one sentence, no movement commands, no brackets."
    )
    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    result = ""
    try:
        async with assistant_slots:
            state.llm_inflight += 1
            async for event in assistant_runner.run_async(
                user_id="therapist",
                session_id=session_id,
                new_message=content,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    result = event.content.parts[0].text or ""
            record_agent_event("director_agent", "Generated Gemini greeting")
    except Exception as e:
        record_agent_event("director_agent", f"Greeting generation fallback: {e}", status="degraded")
    finally:
        state.llm_inflight = max(0, state.llm_inflight - 1)

    # Strip optional [COMMAND] prefix if model still adds it.
    cleaned = result.strip()
    if cleaned.startswith("[") and "]" in cleaned:
        cleaned = cleaned[cleaned.find("]") + 1 :].strip()
    return cleaned

def generate_video():
    """Generator for MJPEG streaming."""
    last_reps = 0
    while True:
        frame_bytes, vision_data = tracker.process_frame()
        
        if frame_bytes is None:
            time.sleep(0.1)
            continue

        # Reset video reps counter when patient switches
        if state.reset_video_reps:
            last_reps = 0
            state.reset_video_reps = False

        # If session is NOT active, just stream the raw feed (with heartbeat) and skip logic
        if not state.is_active:
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
             continue

        # -- CLINICAL LOGIC (only when active) --
        goal = state.patient_info.get("therapy_goal", "Squats")
        new_clinical_state = analyze_pose_data(
            squat_angle=vision_data.get("squat_angle", 180),
            arm_angle=vision_data.get("arm_angle", 180),
            therapy_goal=goal
        )

        if new_clinical_state["reps_count"] > last_reps:
            last_reps = new_clinical_state["reps_count"]
            state.clinical_state = new_clinical_state.copy()
            record_agent_event(
                "clinical_agent",
                f"Rep {last_reps}/{state.patient_info.get('target_reps', 10)} "
                f"({state.clinical_state.get('active_exercise', 'exercise')})"
            )
            if state.is_active and main_loop is not None:
                asyncio.run_coroutine_threadsafe(trigger_assistant_update(), main_loop)

        state.clinical_state = new_clinical_state.copy()
        target_reps = int(state.patient_info.get("target_reps", 10) or 10)
        if target_reps > 0 and state.clinical_state["reps_count"] >= target_reps and state.session_completed_at == 0.0:
            state.session_completed_at = time.time()
            state.director_command = "[STOP]"
            state.director_dialogue = "Victory! Session complete. Outstanding work!"
            record_agent_event("clinical_agent", "Target reps completed")
            if main_loop is not None:
                asyncio.run_coroutine_threadsafe(trigger_assistant_update(), main_loop)

        # -- AUTONOMOUS LOGIC --
        person_detected = vision_data.get("person_detected", False)
        t_pose = vision_data.get("t_pose", False)
        whole_body = vision_data.get("whole_body_visible", False)

        # T-Pose Lock Logic: Always update lock status if T-pose detected
        if t_pose:
            if not state.target_locked:
                state.target_locked = True
                state.is_active = True
                state.director_dialogue = "Target locked. I will follow you closely!"
                # Immediate audio feedback for the lock
                if main_loop is not None:
                     asyncio.run_coroutine_threadsafe(trigger_assistant_update(), main_loop)
        
        # Drive calculation
        if person_detected and state.follow_enabled:
            # Determine mode based on lock status
            mode = "follow" if state.target_locked else "approach"
            
            result = calculate_shadow_drive(
                center_x=vision_data.get("center_x", 0.5),
                depth=vision_data.get("distance_depth", 128),
                shoulder_width=vision_data.get("shoulder_width", 0.2),
                mode=mode,
                whole_body_visible=whole_body,
                yolo_height=vision_data.get("yolo_height", 0.0)
            )

            # A2A handoff: Director intent can override pacer on command moments.
            new_roomba_out = result["drive_command"]
            if state.director_command and state.director_command != "[STOP]":
                intent = translate_intent_to_drive(state.director_command).get("roomba_command", "")
                if intent.startswith("ROOMBA_CMD: "):
                    intent_drive = intent.replace("ROOMBA_CMD: ", "").strip()
                    if intent_drive != "drive 0,0":
                        new_roomba_out = intent_drive
                        record_agent_event("pacer_agent", f"Override from director {state.director_command}")
            
            if new_roomba_out != state.roomba_output:
                state.roomba_output = new_roomba_out
                record_agent_event("pacer_agent", f"Drive output -> {new_roomba_out}")
                if main_loop is not None:
                    cmd = drive_string_to_command(new_roomba_out)
                    state.last_esp32_cmd = cmd
                    state.last_esp32_cmd_at = time.time()
                    asyncio.run_coroutine_threadsafe(send_command(cmd), main_loop)
        else:
            # No person detected: STOP
            if state.roomba_output != "drive 0,0":
                state.roomba_output = "drive 0,0"
                record_agent_event("pacer_agent", "No person detected, emergency stop", status="degraded")
                if main_loop is not None:
                    asyncio.run_coroutine_threadsafe(send_command(4), main_loop)

        if frame_bytes is None:
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.post("/api/camera/{index}")
def set_camera(index: str):
    tracker.set_camera(index)
    return {"status": "ok", "camera_index": index}

@app.post("/api/esp32")
async def set_esp32(data: dict):
    """Runtime configuration hook for robot endpoint."""
    ws_url = str(data.get("ws_url", "")).strip()
    if ws_url:
        configure_esp32(ws_url)
        record_agent_event("pacer_agent", f"ESP32 target updated: {ws_url}")
    return {"status": "ok", "ws_url": ws_url or "unchanged"}

@app.post("/api/follow/unlock")
async def follow_unlock():
    """Explicitly release following lock, requiring new T-pose to resume."""
    state.follow_enabled = False
    state.target_locked = False
    state.is_active = False
    state.director_dialogue = "Follow target released. Perform T-pose to re-lock."
    state.roomba_output = "drive 0,0"
    await send_stop()
    return {"status": "unlocked"}

@app.post("/api/esp32/toggle")
async def esp32_toggle():
    """Toggle the Roomba's following ability without stopping the session."""
    state.follow_enabled = not state.follow_enabled
    if not state.follow_enabled:
        await send_stop()
        state.roomba_output = "drive 0,0"
    
    return {
        "status": "ok", 
        "follow_enabled": state.follow_enabled,
        "label": "STOP" if state.follow_enabled else "RESUME"
    }

@app.post("/api/esp32/stop")
async def esp32_stop():
    """Deprecated: use toggle or specific state setters."""
    return await esp32_toggle()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/remote")
def remote_control():
    return FileResponse("static/remote.html")

@app.post("/api/roomba/{cmd}")
async def roomba_command(cmd: int):
    """(Deprecated) Handled by frontend directly."""
    names = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]
    return {"status": "ok", "command": names[cmd] if 0 <= cmd <= 4 else "UNKNOWN"}

@app.get("/api/roomba/ping")
async def roomba_ping():
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    """Judge-facing reliability snapshot."""
    return {
        "status": "ok",
        "health": _health_snapshot(),
        "summary": _session_summary(),
        "agents": state.agent_status,
    }


@app.get("/api/session/summary")
async def session_summary():
    """Returns outcome metrics suitable for scorecard display/export."""
    summary = _session_summary()
    summary["patient"] = state.patient_info.get("name", "Unknown")
    summary["goal"] = state.patient_info.get("therapy_goal", "General movement")
    summary["llm_failures"] = state.llm_failures
    summary["last_llm_error"] = state.last_llm_error
    return summary

async def periodic_broadcast():
    while True:
        await asyncio.sleep(1.0)
        await broadcast_state()

main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    esp32_env = os.getenv("ESP32_WS_URL", "").strip()
    if esp32_env:
        configure_esp32(esp32_env)
        record_agent_event("pacer_agent", f"ESP32 URL from env: {esp32_env}")
    asyncio.create_task(periodic_broadcast())
    # Create a default session
    await session_service.create_session(
        app_name="therapy_app",
        user_id="therapist",
        session_id=state.session_id,
    )
