from google.adk.agents import Agent

# ──────────────────────────────────────────
# Therapy Assistant Agent — LLM Conversational Agent
# ──────────────────────────────────────────

def format_therapy_context(
    patient_name: str,
    therapy_goal: str,
    target_reps: int,
    current_reps: int,
    kid_speech: str
) -> dict:
    """Formats the current therapy session data into a readable context for the assistant.

    Args:
        patient_name: The child's name.
        therapy_goal: The assigned exercise goal.
        target_reps: How many reps the child needs to complete.
        current_reps: How many reps have been completed so far.
        kid_speech: What the child just said via microphone (empty string if nothing).

    Returns:
        dict: A formatted context summary.
    """
    ctx = {
        "patient": patient_name,
        "goal": therapy_goal,
        "progress": f"{current_reps}/{target_reps}",
        "kid_said": kid_speech if kid_speech else "(nothing)"
    }
    return ctx

assistant_agent = Agent(
    name="therapy_assistant_agent",
    model="gemini-2.5-flash",
    description="A friendly pediatric physical therapy robot assistant that guides children through exercises.",
    instruction=(
        "You are a friendly, encouraging pediatric physical therapy robot assistant. "
        "Your job is to gently guide kids through their physical therapy exercises. "
        "You must be cheerful, patient, and sweet. "
        "If the kid talks to you, respond directly to what they said while maintaining "
        "your friendly persona and gently encouraging them to exercise. "
        "Keep your responses very short (1-2 sentences max) and positive. "
        "IMPORTANT: Start your response with a movement command in brackets: "
        "[MOVE FORWARD], [TURN LEFT], [TURN RIGHT], [SPIN], [STOP], or [BACKUP]. "
        "Then put your dialogue after it. "
        "Example: [SPIN] Great job! You are doing so well! "
        "Use the format_therapy_context tool when you need to check the session data."
    ),
    tools=[format_therapy_context],
)
