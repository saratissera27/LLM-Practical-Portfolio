# MULTI-TOOL AI AGENT

import ast
import json
import operator
import os
import platform
import re
import subprocess
import time

from datetime import datetime
from pathlib import Path

import psutil
import pyautogui

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_ollama import ChatOllama


# Initialize Ollama LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# TOOL 1 - CALCULATOR

math_operations = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}


def calculate_expression(expression):
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

        raise ValueError("Unsupported expression")

    return evaluate(node)


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions such as 25*4."""

    try:
        result = calculate_expression(expression)

        subprocess.Popen("calc.exe")
        time.sleep(2)

        pyautogui.write(str(result), interval=0.1)

        return f"Calculation result: {result}"

    except Exception as e:
        return f"Calculator error: {e}"

# TOOL 2 - DATE AND TIME

@tool
def get_date_time() -> str:
    """Retrieve the current date and time."""

    now = datetime.now()

    return now.strftime(
        "Current Date: %Y-%m-%d\n"
        "Current Time: %H:%M:%S"
    )

# TOOL 3 - NOTEPAD

@tool
def open_notepad(text: str) -> str:
    """Open Windows Notepad and write user-provided text."""

    try:
        subprocess.Popen("notepad.exe")
        time.sleep(3)

        pyautogui.write(text, interval=0.1)

        return f"Notepad opened and wrote: {text}"

    except Exception as e:
        return f"Notepad error: {e}"

# TOOL 4 - FILE SEARCH

@tool
def search_files(directory: str, extension: str) -> str:
    """Search for files by extension in a directory."""

    try:
        folder = Path(directory)

        if not folder.is_dir():
            return f"Directory not found: {directory}"

        extension = extension.strip().lstrip(".").lower()

        if not extension or not extension.isalnum():
            return "Invalid file extension."

        matching_files = []

        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith("." + extension):
                    matching_files.append(
                        str(Path(root) / filename)
                    )

                    if len(matching_files) >= 50:
                        break

            if len(matching_files) >= 50:
                break

        if not matching_files:
            return "No matching files found."

        return "\n".join(matching_files)

    except Exception as e:
        return f"File search error: {e}"

# TOOL 5 - FILE CREATION

@tool
def create_file(directory: str, filename: str, content: str) -> str:
    """Create a text file in a specified directory.

    Args:
        directory: Full Windows folder path where the file must be created.
        filename: Name of the text file, such as test.txt.
        content: Text to write inside the file.
    """

    try:
        folder = Path(directory)

        print("\n[DEBUG] Requested directory:", repr(directory))
        print("[DEBUG] Filename:", repr(filename))
        print("[DEBUG] Content:", repr(content))

        # Check the destination folder
        if not folder.is_dir():
            return f"ERROR: Directory does not exist: {directory}"

        # Prevent filenames containing another path
        if Path(filename).name != filename:
            return "ERROR: Filename must not contain a folder path."

        if not filename.lower().endswith(".txt"):
            return "ERROR: Only .txt files are supported."

        # Construct the complete file path
        full_path = folder / filename

        print("[DEBUG] Final path:", full_path)

        if full_path.exists():
            return f"File already exists: {full_path.resolve()}"

        # Create file in the requested folder
        with full_path.open("x", encoding="utf-8") as file:
            file.write(content)

        # Verify file creation
        if full_path.is_file():
            return (
                f"SUCCESS: File created at "
                f"{full_path.resolve()}"
            )

        return "ERROR: File creation could not be verified."

    except Exception as e:
        return f"File creation error: {e}"

# TOOL 6 - SYSTEM INFORMATION

@tool
def system_information() -> str:
    """Retrieve operating system, processor and RAM details."""

    os_name = platform.system()
    os_version = platform.version()
    processor = platform.processor()

    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024 ** 3)

    return (
        f"Operating System: {os_name}\n"
        f"OS Version: {os_version}\n"
        f"Processor: {processor}\n"
        f"Total RAM: {ram_gb:.2f} GB"
    )

# REGISTER ALL TOOLS

tools = [
    calculator,
    get_date_time,
    open_notepad,
    search_files,
    create_file,
    system_information
]

tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

# AGENT EXECUTION LOOP

def execute_tool(name, arguments):

    if name not in tool_map:
        return f"Unknown tool: {name}"

    if not isinstance(arguments, dict):
        return "Invalid tool arguments."

    try:
        # Ask permission before changing files or opening Notepad
        if name in ["open_notepad", "create_file"]:
            print("\nRequested action:", name)
            print("Arguments:", arguments)

            approval = input("Allow this action? (yes/no): ")

            if approval.strip().lower() != "yes":
                return "Action cancelled."

        return str(tool_map[name].invoke(arguments))

    except Exception as e:
        return f"Tool execution error: {e}"


def run_agent(user_input):

    messages = [
        SystemMessage(
            content=(
                "You are a Windows AI assistant. "
                "Use the appropriate tool for the user's request. "
                "Available tools: calculator, get_date_time, "
                "open_notepad, search_files, create_file, "
                "system_information. "
                "For file searches, extract the directory "
                "and extension from the request. "
                "For file creation, extract the filename "
                "and content. "
                "Do not invent tool results."
            )
        ),
        HumanMessage(content=user_input)
    ]

    for _ in range(5):

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Handle normal LangChain tool calls
        if response.tool_calls:

            for call in response.tool_calls:
                name = call["name"]
                arguments = call["args"]

                result = execute_tool(name, arguments)

                print("\n[TOOL SELECTED]", name)
                print("[TOOL RESULT]", result)

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=call["id"]
                    )
                )

            continue

        # Fallback for Ollama JSON-text tool calls
        content = response.content

        if isinstance(content, str):
            try:
                match = re.search(
                    r'\{.*\}',
                    content,
                    re.DOTALL
                )

                if match:
                    data = json.loads(match.group())

                    name = data.get("name")
                    arguments = data.get(
                        "parameters",
                        data.get("arguments", {})
                    )

                    if name in tool_map:
                        result = execute_tool(name, arguments)

                        print("\n[TOOL SELECTED]", name)
                        print("[TOOL RESULT]", result)
                        print("Agent:", result)
                        return

            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        print("\nAgent:", content)
        return

    print("Agent stopped after reaching the tool-call limit.")

# RUN THE MULTI-TOOL AGENT

print("=" * 45)
print("MULTI-TOOL AI AGENT USING OLLAMA")
print("=" * 45)

while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Agent stopped.")
        break

    if user_input:
        try:
            run_agent(user_input)

        except Exception as e:
            print("Agent Error:", e)

