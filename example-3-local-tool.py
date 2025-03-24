from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import UserProxyAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
import random
import asyncio
from dotenv import load_dotenv
import os

# === Car stock agent tools ===

def get_car_stock(input_brand: str) -> str:
    """Get amount of cars in stock for a brand"""
    # Get random number between 1 and 100 for car stock amount
    car_stock_amount = random.randint(1, 100)
    return f"Car stock for {input_brand} is {car_stock_amount}"

def rent_car(input_brand: str) -> str:
    """Rent a car"""
    return f"Car {input_brand} rented"

async def main() -> None:

    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        model="gpt-4o",
        api_version="2024-06-01",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
    )

    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        handoffs=["car_stock_expert", "car_features_expert", "user_proxy"],
        tools=[rent_car],
        system_message="""You are a research planning coordinator.
        Coordinate a car rental plan for the user. You will get their request, then handoff to the appropriate expert. Ask the user for permission before renting something.
        You have the following experts at your disposal:
        - Car features expert: For getting recommendations of what car brands a user should rent based on their requirements
        - Car stock expert: For getting what brands of cars are currently available to rent
        - User proxy: For getting input from the user
        Always send your plan first, then handoff to appropriate agent.
        Always handoff to a single agent at a time.
        Say TERMINATE when rental is complete.""",
    )

    user_proxy = UserProxyAgent("user_proxy")

    car_stock_expert = AssistantAgent(
        "car_stock_expert",
        model_client=model_client,
        handoffs=["planner"],
        tools=[get_car_stock, rent_car],
        system_message="""You are a car stock expert.
        For any asked for car brands you will provide the amount of cars in stock.
        Always handoff back to planner when analysis is complete.""",
    )

    car_features_expert = AssistantAgent(
        "car_features_expert",
        model_client=model_client,
        handoffs=["planner"],
        system_message="""You are a car features expert
        Based on a the user's input, you will recommend a car brand or brands to rent.
        Always handoff back to planner when analysis is complete.""",
    )

    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
    team = Swarm([planner, car_stock_expert, car_features_expert, user_proxy], termination_condition=termination)

    task = "I want to rent a fast but safe car."

    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]

asyncio.run(main())