from google.adk.agents import Agent

# ──────────────────────────────────────────
# Patient Record Agent
# ──────────────────────────────────────────
# Mock patient database
_PATIENT_DB = {
    "akhilesh": {
        "name": "Akhilesh",
        "age": 11,
        "therapy_goal": "Arm Raises X20",
        "target_reps": 20,
        "voice_id": "bh4qskdfSl83na9IzVGC"
    },
    "bailey": {
        "name": "Bailey",
        "age": 9,
        "therapy_goal": "Squats X20",
        "target_reps": 20,
        "voice_id": "bh4qskdfSl83na9IzVGC"
    },
    "john": {
        "name": "John",
        "age": 12,
        "therapy_goal": "Side Reach X20",
        "target_reps": 20,
        "voice_id": "bh4qskdfSl83na9IzVGC"
    }
}

def get_patient_record(patient_id: str) -> dict:
    """Retrieves the patient profile from the hospital database.
    
    Args:
        patient_id: The unique identifier for the patient (e.g. 'patient_123').
    
    Returns:
        dict: The patient profile containing name, age, therapy_goal, and target_reps.
    """
    return _PATIENT_DB.get(patient_id, {
        "name": "Unknown",
        "age": 0,
        "therapy_goal": "General movement",
        "target_reps": 5
    })

patient_agent = Agent(
    name="patient_record_agent",
    model="gemini-2.5-flash",
    description="Agent that retrieves patient medical records and therapy plans from the hospital database.",
    instruction=(
        "You are a patient record retrieval agent. When asked about a patient, "
        "use the get_patient_record tool to look up their profile and return it."
    ),
    tools=[get_patient_record],
)
