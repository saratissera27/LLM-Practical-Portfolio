import ast
import operator
import subprocess
import time

import pyautogui
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage


# STEP 1: Initialize the Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# STEP 2: Define safe arithmetic operations
operations = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}


def evaluate_expression(expression):
    node = ast.parse(expression, mode="eval").body

    def calculate(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value

        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -calculate(n.operand)

        if isinstance(n, ast.BinOp) and type(n.op) in operations:
            return operations[type(n.op)](
                calculate(n.left),
                calculate(n.right)
            )

        raise ValueError("Unsupported expression")

    return calculate(node)


# STEP 3: Create calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions such as 25*4 or 10+5."""

    try:
        result = evaluate_expression(expression)

        # Open Windows Calculator
        subprocess.Popen("calc.exe")
        time.sleep(2)

        # Display the result in Calculator
        pyautogui.hotkey("ctrl", "a")
        pyautogui.write(str(result), interval=0.1)

        return f"Calculation result: {result}"

    except Exception as e:
        return f"Calculator error: {e}"


# STEP 4: Register the tool with the LLM
tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

tool_map = {tool.name: tool for tool in tools}


# STEP 5: Agent execution loop

# STEP 5: Agent execution loop
def run_agent(user_input):

    messages = [
        SystemMessage(
            content=(
                "You are a calculator assistant. "
                "Use the calculator tool for every arithmetic request. "
                "Never output tool calls as JSON text."
            )
        ),
        HumanMessage(content=user_input)
    ]

    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = response.tool_calls

        # Handle models that return tool calls as JSON text
        if not tool_calls:
            import json
            import re

            content = response.content

            if isinstance(content, str):
                try:
                    match = re.search(r'\{.*\}', content, re.DOTALL)

                    if match:
                        data = json.loads(match.group())

                        if data.get("name") == "calculator":
                            args = data.get(
                                "parameters",
                                data.get("arguments", {})
                            )

                            result = calculator.invoke(args)

                            print("Tool: calculator")
                            print("Result:", result)
                            print("Agent:", result)
                            return

                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            print("Agent:", response.content)
            return

        for tool_call in tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            if name not in tool_map:
                result = "Unknown tool"
            else:
                result = tool_map[name].invoke(args)

            print("Tool:", name)
            print("Result:", result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

    print("Agent stopped after reaching the tool-call limit.")

# STEP 6: Run and test
while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    run_agent(user_input)