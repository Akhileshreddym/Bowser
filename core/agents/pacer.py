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
    """Calculates continuous Roomba drive output to approach a person to 2 ft.

    Uses horizontal position and depth to steer towards the target location at 2 ft distance.

    Args:
        center_x: Normalized horizontal center (0.0=left, 1.0=right).
        depth: Depth estimation value.
        shoulder_width: Normalized shoulder width.

    Returns:
        dict: Contains 'drive_command' with the drive string.
    """
    target_depth = 90.0  # Approximate 2 ft in depth units

    # Multi-person caution
    if shoulder_width > 0.3:
        return {"drive_command": "drive 0,0"}

    # Proportional control
    x_error = center_x - 0.5
    depth_error = depth - target_depth

    turn = -x_error * 50  # Turn gain
    velocity = -depth_error * 0.5  # Velocity gain

    # Clamp
    velocity = max(-50, min(50, int(velocity)))
    turn = max(-50, min(50, int(turn)))

    # Dead zone
    if abs(velocity) < 5 and abs(turn) < 5:
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
