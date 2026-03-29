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

def calculate_shadow_drive(center_x: float, depth: float, shoulder_width: float = 0.2) -> dict:
    """Calculates continuous Roomba drive output to follow a person.

    Uses horizontal position and depth to steer towards the target location.
    If multiple people are in frame (wide shoulder width), the robot will behave cautiously.

    Args:
        center_x: Normalized horizontal center (0.0=left, 1.0=right).
        depth: Depth estimation value where larger means closer.
        shoulder_width: Normalized shoulder width; very large values usually indicate multiple people.

    Returns:
        dict: Contains 'drive_command' with the drive string.
    """
    # target region is center and medium distance
    target_center = 0.5
    target_depth = 150.0

    # Minimal distance safety (2ft rule): if too close in depth, go backwards
    min_depth_for_2ft = 90.0  # empirical, adjust to camera/depth scale

    if depth >= min_depth_for_2ft:
        # too close: retreat
        return {"drive_command": "drive -40,0"}

    # multi-person caution: if shoulder width is large, slow down and keep center
    if shoulder_width > 0.3:
        return {"drive_command": "drive 0,0"}

    # proportional control for smoother tracking
    x_error = center_x - target_center
    depth_error = depth - target_depth

    # Rotation command to point toward person (left/right)
    # Positive for left turn, negative for right turn
    turn = int(max(-80, min(80, -x_error * 220)))

    # Forward/backward speed: positive = forward (approach), negative = back away
    base_speed = int(max(-100, min(100, -depth_error * 0.7)))

    # Dead zone around ideal window to avoid oscillation
    if abs(x_error) < 0.05 and abs(depth_error) < 15:
        return {"drive_command": "drive 0,0"}

    # If we're mostly aligned, keep straight ahead without steering jitter
    if abs(x_error) < 0.1:
        turn = 0

    # Convert tiny residual values to significant movement
    if base_speed == 0 and abs(depth_error) > 15:
        base_speed = 30 if depth_error < 0 else -30

    return {"drive_command": f"drive {base_speed},{turn}"}

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
