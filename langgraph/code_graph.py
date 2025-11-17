# flake8: noqa

import os
from typing_extensions import TypedDict
from openai import OpenAI
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, ValidationError

load_dotenv()

# -----------------------------
# GEMINI-COMPATIBLE CLIENT
# -----------------------------
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# -----------------------------
# Response schemas
# -----------------------------
class ClassifyMessageResponse(BaseModel):
    is_coding_question: bool


class CodeAccuracyResponse(BaseModel):
    accuracy_percentage: str


# -----------------------------
# State type
# -----------------------------
class State(TypedDict):
    user_query: str
    llm_result: str | None
    accuracy_percentage: str | None
    is_coding_question: bool | None


# -----------------------------
# Nodes
# -----------------------------
def classify_message(state: State):
    print("⚠️ classify_message")
    query = state["user_query"]

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is
    related to coding question or not.
    Return the response in specified JSON boolean only.
    """

    response = client.beta.chat.completions.parse(
        model="gemini-2.5-flash",
        response_format=ClassifyMessageResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )

    is_coding_question = response.choices[0].message.parsed.is_coding_question
    state["is_coding_question"] = is_coding_question

    return state


def route_query(state: State) -> Literal["general_query", "coding_query"]:
    """
    This function returns the next node name (string).
    It's used by add_conditional_edges on the classify_message node.
    """
    print("⚠️ route_query")
    is_coding = state.get("is_coding_question", False)
    return "coding_query" if is_coding else "general_query"


def general_query(state: State):
    print("⚠️ general_query")
    query = state["user_query"]

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": query}
        ],
    )

    state["llm_result"] = response.choices[0].message.content
    return state


def coding_query(state: State):
    print("⚠️ coding_query")
    query = state["user_query"]

    SYSTEM_PROMPT = "You are a Coding Expert Agent"

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
    )

    state["llm_result"] = response.choices[0].message.content
    return state


def coding_validate_query(state: State):
    print("⚠️ coding_validate_query")
    query = state["user_query"]
    llm_code = state["llm_result"]

    SYSTEM_PROMPT = f"""
        You are expert in calculating accuracy of the code according to the question.
        Return the percentage of accuracy
        
        User Query: {query}
        Code: {llm_code}
    """

    response = client.beta.chat.completions.parse(
        model="gemini-2.5-flash",
        response_format=CodeAccuracyResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    )

    state["accuracy_percentage"] = response.choices[0].message.parsed.accuracy_percentage
    return state



# -----------------------------
# Build graph
# -----------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("classify_message", classify_message)
# Note: route_query is NOT added as a node — it's the conditional decision function
# Add the query-handling nodes:
graph_builder.add_node("general_query", general_query)
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_validate_query", coding_validate_query)

# Graph edges:
graph_builder.add_edge(START, "classify_message")
# Use classify_message node + conditional decision function to pick next node
graph_builder.add_conditional_edges("classify_message", route_query)

graph_builder.add_edge("general_query", END)
graph_builder.add_edge("coding_query", "coding_validate_query")
graph_builder.add_edge("coding_validate_query", END)

graph = graph_builder.compile()


# -----------------------------
# Runner
# -----------------------------
def main():
    user = input("> ")

    _state: State = {
        "user_query": user,
        "accuracy_percentage": None,
        "is_coding_question": None,
        "llm_result": None
    }

    for event in graph.stream(_state):
        import json
        print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
