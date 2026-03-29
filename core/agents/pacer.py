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

def calculate_shadow_drive(center_x: float, depth: float) -> dict:
    """Calculates continuous Roomba differential drive output to shadow a patient.

    Uses the patient's horizontal position (center_x) and depth distance
    to compute real-time motor commands.

    Args:
        center_x: The normalized horizontal center of the patient (0.0=left, 1.0=right).
        depth: The depth estimation value from the neural network (0-255, higher=closer).

    Returns:
        dict: Contains 'drive_command' with the drive string.
    """
    base_speed = 0
    turn = 0

    if center_x > 0.6:
        turn = -50
    elif center_x < 0.4:
        turn = 50

    if depth < 120:
        base_speed = 100
    elif depth > 180:
        base_speed = -100

    if base_speed == 0 and turn == 0:
        return {"drive_command": "drive 0,0"}

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
