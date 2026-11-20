
# LAB SHEET 6 - PART D
# GOOGLE CALENDAR AI AGENT

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, HumanMessage, ToolMessage
)

from rag_agent import calculator, search_student_memory, rag_search
from memory_agent import initialize_database
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
TIMEZONE = ZoneInfo("Asia/Colombo")


# STEP 1: Authenticate Google Calendar

def get_calendar_service():
    credentials = None
    token_path = BASE_DIR / "token.json"
    credentials_path = BASE_DIR / "credentials.json"

    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(
            str(token_path), SCOPES
        )

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    "credentials.json is missing."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )

            credentials = flow.run_local_server(
               host="localhost",
               port=8088,
               open_browser=True
            )

        token_path.write_text(
            credentials.to_json(), encoding="utf-8"
        )

    return build(
        "calendar", "v3", credentials=credentials
    )


# STEP 2: Check today's calendar

@tool
def get_today_schedule() -> str:
    """Retrieve today's events from the user's Google Calendar."""

    try:
        service = get_calendar_service()

        now = datetime.now(TIMEZONE)
        start = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=1)

        result = service.events().list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = result.get("items", [])

        if not events:
            return "No events scheduled for today."

        lines = []
        for event in events:
            event_start = event["start"].get(
                "dateTime", event["start"].get("date")
            )
            lines.append(
                f"{event.get('summary', 'Untitled')}: {event_start}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Calendar error: {e}"


# STEP 3: Create a calendar event

@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str
) -> str:
    """Create a Google Calendar event.

    Args:
        title: Event title.
        start_time: ISO datetime such as 2026-11-16T10:00:00+05:30.
        end_time: ISO datetime such as 2026-11-16T11:00:00+05:30.
    """

    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)

        if start.tzinfo is None or end.tzinfo is None:
            return "Error: Both times must include a timezone."

        if end <= start:
            return "Error: End time must be after start time."

        service = get_calendar_service()

        event = {
            "summary": title,
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": "Asia/Colombo"
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": "Asia/Colombo"
            }
        }

        created = service.events().insert(
            calendarId="primary",
            body=event
        ).execute()

        return (
            f"Event created successfully: {created['summary']}\n"
            f"Event link: {created.get('htmlLink', '')}"
        )

    except Exception as e:
        return f"Calendar creation error: {e}"


# STEP 4: Register all five tools

tools = [
    calculator,
    search_student_memory,
    rag_search,
    get_today_schedule,
    create_calendar_event
]

tool_map = {t.name: t for t in tools}

llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# STEP 5: Run agent

def run_agent(user_input):
    messages = [
        SystemMessage(content=(
            "You are a university AI assistant. "
            "Use calculator for arithmetic, "
            "search_student_memory for stored user facts, "
            "rag_search for university PDF questions, "
            "get_today_schedule for today's calendar events, "
            "and create_calendar_event for creating events. "
            "The user's timezone is Asia/Colombo. "
            "Do not invent calendar events. "
            "Never create an event without user approval."
        )),
        HumanMessage(content=user_input)
    ]

    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("\nAgent:", response.content)
            return

        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]

            print("\n[TOOL SELECTED]", name)
            print("[ARGUMENTS]", args)

            if name == "create_calendar_event":
                approval = input(
                    "Create this calendar event? (yes/no): "
                ).strip().lower()

                if approval != "yes":
                    result = "Event creation cancelled by user."
                else:
                    try:
                        result = tool_map[name].invoke(args)
                    except Exception as e:
                        result = f"Tool error: {e}"
            elif name in tool_map:
                try:
                    result = tool_map[name].invoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {name}"

            print("[TOOL RESULT]", result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    print("Agent stopped after reaching the tool-call limit.")


# STEP 6: Start application

if __name__ == "__main__":
    initialize_database()

    print("GOOGLE CALENDAR UNIVERSITY AI AGENT")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        if user_input:
            try:
                run_agent(user_input)
            except Exception as e:
                print("Agent error:", e)
