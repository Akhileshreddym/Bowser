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
    whole_body_visible: bool = False
) -> dict:
    """Calculates continuous Roomba drive output to approach or follow a person.

    Args:
        center_x: Normalized horizontal center (0.0=left, 1.0=right).
        depth: Depth estimation value.
        shoulder_width: Normalized shoulder width.
        mode: "approach" (go to person and stop) or "follow" (active tracking).
        whole_body_visible: Whether the full person is currently in frame.

    Returns:
        dict: Contains 'drive_command' with the drive string.
    """
    # Proportional control for steering (always active if person detected)
    x_error = center_x - 0.5
    turn = -x_error * 60  # Slightly higher turn gain for responsiveness

    # Velocity control based on mode
    velocity = 0
    
    if mode == "follow":
        # Active following: target a specific depth/distance
        target_depth = 100.0 if whole_body_visible else 80.0
        depth_error = depth - target_depth
        velocity = -depth_error * 0.6
    else:
        # Default "Approach" mode: go closer until whole body is visible and at a good distance
        # If whole body is NOT visible, we might be too close (or they are partially out)
        if not whole_body_visible:
            if shoulder_width > 0.25:
                # Too close! Stop or back up slightly
                velocity = -10 
            else:
                # Far away but ankles cut off? Move forward slowly to find them
                velocity = 20
        else:
            # Whole body visible! If shoulder width is small, we are far away
            if shoulder_width < 0.18:
                velocity = 30 # Move forward to close the gap
            else:
                velocity = 0 # Stay here, we can see them fully

    # Multi-person or extreme proximity caution
    if shoulder_width > 0.4:
        velocity = 0
        turn = 0

    # Clamp
    velocity = max(-40, min(50, int(velocity)))
    turn = max(-50, min(50, int(turn)))

    # Dead zone to prevent jitter
    if abs(velocity) < 8 and abs(turn) < 8:
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
