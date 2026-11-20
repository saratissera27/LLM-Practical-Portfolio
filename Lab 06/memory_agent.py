
# LAB SHEET 6 - PART B
# AI AGENT WITH SQLITE MEMORY

import ast
import operator
import re
import sqlite3
from pathlib import Path
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
from langchain_ollama import ChatOllama


# STEP 1: Initialize Ollama

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# STEP 2: Initialize SQLite database

DB_PATH = Path(__file__).resolve().parent / "student_memory.db"


def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL UNIQUE,
                information TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


# STEP 3: Save information to memory

def save_memory(category, information):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO memories (category, information, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(category) DO UPDATE SET
                information = excluded.information,
                created_at = excluded.created_at
        """, (
            category,
            information,
            datetime.now().isoformat()
        ))

    print("[MEMORY SAVED]", information)


# STEP 4: Detect user information

def detect_memory(user_input):

    patterns = {
        "name": r"\bmy name is (.+)",
        "university": r"\bmy university is (.+)",
        "subject": r"\bmy favorite subject is (.+)",
        "study": r"\bi study (.+)"
    }

    for category, pattern in patterns.items():

        match = re.search(
            pattern,
            user_input,
            re.IGNORECASE
        )

        if match:
            information = match.group(1).strip().rstrip(".!?")

            if information:
                save_memory(category, information)
                return category

    return None


# STEP 5: Memory Search Tool

@tool
def search_student_memory(query: str) -> str:
    """Search stored user information such as name,
    university, studies and favorite subject.
    """

    query_lower = query.lower()

    categories = {
        "name": ["name", "called"],
        "university": ["university", "campus"],
        "subject": ["favorite subject", "favourite subject"],
        "study": ["study", "studying"]
    }

    selected_category = None

    for category, keywords in categories.items():
        if any(word in query_lower for word in keywords):
            selected_category = category
            break

    with sqlite3.connect(DB_PATH) as conn:

        if selected_category:
            rows = conn.execute(
                "SELECT category, information FROM memories "
                "WHERE category = ?",
                (selected_category,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT category, information FROM memories"
            ).fetchall()

    if not rows:
        return "No matching memories found."

    return "\n".join(
        f"{category}: {information}"
        for category, information in rows
    )


# STEP 6: Calculator Tool

math_operations = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}


def safe_calculate(expression):

    node = ast.parse(expression, mode="eval").body

    def evaluate(n):

        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value

        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -evaluate(n.operand)

        if isinstance(n, ast.BinOp) and type(n.op) in math_operations:
            return math_operations[type(n.op)](
                evaluate(n.left),
                evaluate(n.right)
            )

        raise ValueError("Unsupported mathematical expression")

    return evaluate(node)


@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions such as 25*4."""

    try:
        result = safe_calculate(expression)
        return f"Calculation result: {result}"

    except Exception as e:
        return f"Calculation error: {e}"


# STEP 7: Register tools

tools = [
    calculator,
    search_student_memory
]

tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)


# STEP 8: Short-term conversation history

conversation_history = []


# STEP 9: Run the AI Agent

def run_agent(user_input):

    saved_category = detect_memory(user_input)

    # Retrieve stored facts when relevant
    memory_keywords = [
        "my name",
        "my university",
        "my favorite subject",
        "my favourite subject",
        "what do i study",
        "what am i studying",
        "remember about me"
    ]

    memory_context = ""

    if any(
        keyword in user_input.lower()
        for keyword in memory_keywords
    ):
        memory_context = search_student_memory.invoke({
            "query": user_input
        })

        print("[TOOL SELECTED] search_student_memory")
        print("[TOOL RESULT]", memory_context)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful student AI assistant. "
                "Remember the conversation context. "
                "Use the calculator tool for mathematical calculations. "
                "Use the memory search tool to retrieve stored facts. "
                "Do not invent user information. "
                "Use the provided memory context when answering. "
                f"\nStored memory context:\n{memory_context}"
            )
        )
    ]

    messages.extend(conversation_history)
    messages.append(HumanMessage(content=user_input))

    # Process up to 5 tool-calling rounds
    for _ in range(5):

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:

            for call in response.tool_calls:

                name = call["name"]
                arguments = call["args"]

                if name in tool_map:
                    result = str(
                        tool_map[name].invoke(arguments)
                    )
                else:
                    result = f"Unknown tool: {name}"

                print("[TOOL SELECTED]", name)
                print("[TOOL RESULT]", result)

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=call["id"]
                    )
                )

            continue

        print("\nAgent:", response.content)

        conversation_history.append(
            HumanMessage(content=user_input)
        )

        conversation_history.append(
            AIMessage(content=str(response.content))
        )

        # Keep recent conversation turns
        if len(conversation_history) > 20:
            del conversation_history[:-20]

        return

    print("Agent stopped after reaching the tool-call limit.")


# STEP 10: Start the application

if __name__ == "__main__":

    initialize_database()

    print("=" * 45)
    print("INTELLIGENT STUDENT AI AGENT \n SHORT-TERM MEMORY \n LONG-TERM SQLITE \n MEMORY WORKING MEMORY")
    print("=" * 45)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input:
            try:
                run_agent(user_input)

            except Exception as e:
                print("Agent Error:", e)
