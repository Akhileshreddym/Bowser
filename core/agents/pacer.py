from google.adk.agents import Agent

# ──────────────────────────────────────────
# Pacer Agent — Roomba Motor Controller
# ──────────────────────────────────────────

_COMMAND_MAP = {
    "[MOVE FORWARD]": "drive 100,0",
    "[TURN LEFT]": "drive 50,50",
    "[TURN RIGHT]": "drive 50,-50",
    "[SPIN]": "drive 100,-100",
    "[BACKUP]": "drive -100,0",
    "[STOP]": "drive 0,0"
}

def translate_intent_to_drive(intent_command: str) -> dict:
    """Translates a high-level movement intent command into Roomba drive bytes.

    Args:
        intent_command: A movement intent in brackets (e.g. '[SPIN]', '[MOVE FORWARD]').

    Returns:
        dict: Contains 'roomba_command' with the raw drive string.
    """
    cmd = _COMMAND_MAP.get(intent_command, "drive 0,0")
    return {"roomba_command": f"ROOMBA_CMD: {cmd}"}

def calculate_shadow_drive(
    center_x: float, 
    depth: float, 
    shoulder_width: float = 0.2, 
    mode: str = "approach", 
    whole_body_visible: bool = False,
    yolo_height: float = 0.0
) -> dict:
    """Calculates continuous Roomba drive output to approach or follow a person.
    Targets a 2ft (60cm) distance for safety.
    """
    # Proportional control for steering (always active if person detected)
    x_error = center_x - 0.5
    turn = -x_error * 30  # Softer steering to prevent losing the human

    # Velocity control based on mode
    velocity = 0
    
    # Target Values for 2ft (Target Social Distance)
    TARGET_SHOULDER_WIDTH = 0.16
    TOO_CLOSE_SHOULDER_WIDTH = 0.22
    TARGET_DEPTH = 135.0
    
    # ── EMERGENCY STOP / SAFETY ──
    # 1. YOLO Height: If person takes up >85% of frame, they are TOO CLOSE
    y_h = yolo_height
    # 2. Depth Map: If median depth is very close (>220 disparity)
    d_v = depth
    
    if y_h > 0.85 or d_v > 220 or shoulder_width > TOO_CLOSE_SHOULDER_WIDTH:
        # HARD STOP
        return {"drive_command": "drive 0,0"}

    if mode == "follow":
        # Follow mode: keep a specific distance
        target = TARGET_DEPTH if whole_body_visible else TARGET_DEPTH - 20
        depth_error = depth - target
        velocity = -depth_error * 0.4
    else:
        # "Approach" mode: Approach slowly to 2ft and then STOP
        if shoulder_width < TARGET_SHOULDER_WIDTH:
            velocity = 20 # Decelerated approach speed
        else:
            velocity = 0 # BRAKE APPLIED (Reached 2ft)

    # Multi-person or extreme proximity caution
    if shoulder_width > 0.4:
        velocity = 0
        turn = 0

    # Clamp velocities for smoother motion
    velocity = max(-30, min(35, int(velocity)))
    turn = max(-25, min(25, int(turn)))

    # Enhanced dead zone for stability
    if abs(velocity) < 10 and abs(turn) < 10:
        return {"drive_command": "drive 0,0"}

    return {"drive_command": f"drive {velocity},{turn}"}

pacer_agent = Agent(
    name="pacer_agent",
    model="gemini-2.5-flash",
    description="Agent that translates spatial tracking data and movement intents into Roomba motor drive commands.",
    instruction=(
        "You are a robotic pacer agent. You translate either high-level movement intents "
        "(like [SPIN]) or continuous spatial tracking data (center_x, depth) into raw "
        "Roomba differential drive commands. Use the provided tools to compute outputs."
    ),
    tools=[translate_intent_to_drive, calculate_shadow_drive],
)
