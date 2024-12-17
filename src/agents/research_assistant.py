import os
from dotenv import load_dotenv
from typing import Literal, TypedDict, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from core import get_model, settings
from neo4j import GraphDatabase

from agents.models import models

# Neo4j connection setup
load_dotenv()
neo4j_username = "neo4j"
neo4j_password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=(neo4j_username, neo4j_password))


class QuestionState(MessagesState):
    question_id: int
    question: str
    description: str
    correct_answer: str
    attempts: int
    messages: list[HumanMessage | AIMessage]


instructions = f"""
    You are a helpful AP Computer Science A coding assistant. You will present questions to the user and evaluate their answers.

    A few things to remember:
    - Use the query_neo4j tool when generating a new question.
    - Present the question clearly to the user.
    - Evaluate the user's answer based on the correct answer(s) provided.
    - If the answer is incorrect, explain why without revealing the correct answer on the first attempt.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[QuestionState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: QuestionState, config: RunnableConfig) -> QuestionState:
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def generate_question_id(state: QuestionState) -> str:
    return f'LoLju0t00s{state["question_id"]}'


def generate_seed_question(state: QuestionState) -> QuestionState:
    questionId = generate_question_id(state)
    query = """
            MATCH (q:Question) WHERE (q.LolQuestionIndex.startsWith({questionId: $questionId}))
            RETURN q.QuestionTitle, q.QuestionDescription, q.GoldenSolution
            ORDER BY rand()
            LIMIT 1
            """
    record, _, _ = driver.execute_query(query, {"questionId": questionId})
    result = record.single()
    if result:
        title, description, solution = result
        return {
            "question": title,
            "description": description,
            "correct_answer": solution,
            "attempts": 0,
            "messages": [AIMessage(content=f"{title}\n\n{description}")],
            "question_id": state["question_id"],
        }
    else:
        return {
            "question": "No question available",
            "description": "Please try again later",
            "correct_answer": "",
            "attempts": 0,
            "messages": [AIMessage(content="Sorry, no question is available at the moment.")],
            "question_id": state["question_id"],
        }


async def check_answer(state: QuestionState, config: RunnableConfig) -> QuestionState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)

    user_answer = state["messages"][-1].content
    prompt = f"""
    Question: {state['question']}
    Correct answer(s): {state['correct_answer']}
    User's answer: {user_answer}

    Is the user's answer correct? If so, please say so and praise the user enthusiastically.
    If not, explain why it's wrong without revealing the correct answer or using the word 'correct'.
    """

    response = await model_runnable.ainvoke({"messages": [AIMessage(content=prompt)]}, config)

    if "correct" in response.content.lower():
        state["messages"].append(AIMessage(content=response.content))
        state["question_id"] += 1
        return state
    else:
        state["attempts"] += 1
        if state["attempts"] < 2:
            state["messages"].append(
                AIMessage(content=f"{response.content} You have one more attempt.")
            )
        else:
            state["messages"].append(
                AIMessage(
                    content=f"{response.content} The correct answer was: {state['correct_answer']}. Let's move on to the next question."
                )
            )
        return state


# Initialize the state graph
graph = StateGraph(QuestionState)

# Add nodes to the graph
graph.add_node("generate_question", generate_seed_question)
graph.add_node("check_answer", check_answer)


# Define edges
def should_continue(state: QuestionState) -> str:
    if state["attempts"] == 2 or state["messages"][-1].content.startswith("Correct!"):
        return "new_question"
    return "continue"


graph.add_edge("generate_question", "check_answer")
graph.add_conditional_edges(
    "check_answer",
    should_continue,
    {"new_question": "generate_question", "continue": "check_answer"},
)

# Set the entry point
graph.set_entry_point("generate_question")

# Compile the graph
app = graph.compile()
