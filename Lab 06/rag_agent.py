
# LAB SHEET 6 - PART C
# RAG-POWERED AI AGENT

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
from langchain_ollama import ChatOllama

# Reuse tools and memory from Part B
from memory_agent import (
    calculator,
    search_student_memory,
    initialize_database,
    detect_memory
)

# Reuse RAG from Lab 5
from rag_pipeline import ask_university


# STEP 1: Initialize Ollama

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# STEP 2: Create the RAG Search Tool

@tool
def rag_search(question: str) -> str:
    """Search university PDF documents using RAG.

    Use this tool for questions about university
    information, course details, academic policies,
    AI, machine learning and uploaded PDF content.
    """

    try:
        result = ask_university(question)

        answer = str(result.get("answer", ""))

        sources = result.get("sources", [])

        if sources:
            source_text = "\n".join(
                f"- {s.get('document', 'Unknown')}, "
                f"Page {s.get('page', '?')}"
                for s in sources
            )
        else:
            source_text = "No document sources found."

        return (
            f"RAG Answer:\n{answer}\n\n"
            f"Sources:\n{source_text}"
        )

    except Exception as e:
        return f"RAG Search Error: {e}"


# STEP 3: Register the three tools

tools = [
    calculator,
    search_student_memory,
    rag_search
]

tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)


# STEP 4: Conversation history

conversation_history = []


# STEP 5: Run the agent

def run_agent(user_input):

    # Save supported user facts to SQLite
    detect_memory(user_input)

    messages = [
        SystemMessage(
            content=(
                "You are a University AI Assistant. "
                "You have three tools. "
                "Use calculator for arithmetic. "
                "Use search_student_memory for personal "
                "information previously saved by the user. "
                "Use rag_search for university questions "
                "and questions about uploaded documents. "
                "Never invent university information. "
                "For university questions, use the RAG "
                "tool and preserve its source citations. "
                "If a tool returns an error, report it "
                "instead of inventing a successful result."
            )
        )
    ]

    messages.extend(conversation_history)
    messages.append(HumanMessage(content=user_input))

    # Handle up to five rounds of tool calling
    for _ in range(5):

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:

            for call in response.tool_calls:

                name = call["name"]
                arguments = call["args"]

                print("\n[TOOL SELECTED]", name)

                if name in tool_map:
                    try:
                        result = str(
                            tool_map[name].invoke(arguments)
                        )
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = f"Unknown tool: {name}"

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

        # Keep the last 10 conversation turns
        if len(conversation_history) > 20:
            del conversation_history[:-20]

        return

    print("Agent stopped after reaching the tool-call limit.")


# STEP 6: Start the application

if __name__ == "__main__":

    initialize_database()

    print("=" * 45)
    print("RAG-POWERED UNIVERSITY AI AGENT")
    print("Tools: Calculator | Memory | RAG")
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
