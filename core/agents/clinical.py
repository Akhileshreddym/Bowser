from google.adk.agents import Agent
import time

# ──────────────────────────────────────────
# Clinical Agent — Exercise Tracking Logic
# ──────────────────────────────────────────

# Internal state (module-level so the tool functions can access it)
_clinical_state = {
    "active_exercise": "None",
    "reps_count": 0,
    "form_quality": "unknown",
    "is_in_rep": False,
    "latest_angle": 180.0,
}
_last_rep_time = 0.0
_REP_COOLDOWN = 0.8  # Minimum seconds between counted reps

def analyze_pose_data(squat_angle: float, arm_angle: float, therapy_goal: str) -> dict:
    """Analyzes raw pose angles from computer vision to track exercise repetitions.

    Uses hysteresis thresholds to prevent false-positive counting.
    Dynamically selects which joint angles to measure based on the therapy goal.

    Args:
        squat_angle: The knee angle in degrees (180 = standing straight).
        arm_angle: The shoulder elevation angle in degrees (low = arm at side, high = arm raised).
        therapy_goal: The patient's assigned therapy goal string.

    Returns:
        dict: Clinical state with active_exercise, reps_count, form_quality, and is_in_rep.
    """
    global _clinical_state, _last_rep_time

    goal = (therapy_goal or "").lower()

    if "arm raise" in goal:
        angle = arm_angle
        _clinical_state["active_exercise"] = "Arm Raises"
        # Arm at side ≈ 20-40°, Arm raised ≈ 150-170°
        threshold_enter = 140   # Must raise above this to start a rep
        threshold_exit = 60     # Must lower below this to complete a rep
        enter_high = True       # "enter" means going HIGH
    elif "side reach" in goal or "side" in goal:
        angle = arm_angle
        _clinical_state["active_exercise"] = "Side Reach"
        # Similar to lateral raise cycle: high for entry, low for completion
        threshold_enter = 120
        threshold_exit = 55
        enter_high = True
    else:
        angle = squat_angle
        _clinical_state["active_exercise"] = "Squats"
        threshold_enter = 100   # Must bend below this to start a rep
        threshold_exit = 150    # Must stand above this to complete a rep
        enter_high = False      # "enter" means going LOW

    now = time.time()
    _clinical_state["latest_angle"] = round(float(angle), 2)

    if not enter_high:
        # Squats: enter when angle goes LOW, exit when angle goes HIGH
        if angle < threshold_enter:
            if not _clinical_state["is_in_rep"]:
                _clinical_state["is_in_rep"] = True
                _clinical_state["form_quality"] = "good"
        elif angle > threshold_exit:
            if _clinical_state["is_in_rep"] and (now - _last_rep_time) > _REP_COOLDOWN:
                _clinical_state["reps_count"] += 1
                _clinical_state["is_in_rep"] = False
                _last_rep_time = now
    else:
        # Arm Raises: enter when angle goes HIGH, exit when angle goes LOW
        if angle > threshold_enter:
            if not _clinical_state["is_in_rep"]:
                _clinical_state["is_in_rep"] = True
                _clinical_state["form_quality"] = "good"
        elif angle < threshold_exit:
            if _clinical_state["is_in_rep"] and (now - _last_rep_time) > _REP_COOLDOWN:
                _clinical_state["reps_count"] += 1
                _clinical_state["is_in_rep"] = False
                _last_rep_time = now

    return _clinical_state.copy()

def reset_clinical_state() -> dict:
    """Resets the clinical tracking state back to defaults.

    Returns:
        dict: The reset clinical state.
    """
    global _clinical_state
    _clinical_state = {
        "active_exercise": "None",
        "reps_count": 0,
        "form_quality": "unknown",
        "is_in_rep": False,
        "latest_angle": 180.0,
    }
    return _clinical_state.copy()

clinical_agent = Agent(
    name="clinical_agent",
    model="gemini-2.5-flash",
    description="Agent that analyzes real-time pose data to track physical therapy exercise repetitions and form quality.",
    instruction=(
        "You are a clinical biomechanics agent. You receive raw joint angle data from a "
        "computer vision system and use the analyze_pose_data tool to track exercise "
        "repetitions with medical precision. Never guess — always use the tool."
    ),
    tools=[analyze_pose_data, reset_clinical_state],
)
