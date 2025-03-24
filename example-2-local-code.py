import asyncio
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelFamily, ModelInfo
from dotenv import load_dotenv
import os

async def main() -> None:

    # model_client = AzureOpenAIChatCompletionClient(
    #     azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    #     model="gpt-4o",
    #     api_version="2024-06-01",
    #     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    #     api_key=os.getenv("API_KEY"),
    # )

    model_client = OllamaChatCompletionClient(model="gemma3:latest",
                                              model_info={
                                                  "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
                                              })

    user_proxy = UserProxyAgent("user_proxy")

    agent1 = AssistantAgent("assistant1", model_client=model_client, 
        system_message="You are an AI assistant who helps with tasks by generating Python code to solve them."
        "You place any pip install commands first before you explain the code."
        "You just put the code required, and don't include an explanation at the end."
        "You always annotate your markdown blocks with which language you are using, like so: ```python or ```bash")

    code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    await code_executor.start()

    code_executor_agent = CodeExecutorAgent(
        "code_executor",
        code_executor=code_executor,
    )

    termination = TextMentionTermination("exit", sources=["user_proxy"])
    # Web surfer and user proxy take turns in a round-robin fashion.
    team = RoundRobinGroupChat([agent1, code_executor_agent, user_proxy], termination_condition=termination)
    try:
        # Start the team and wait for it to terminate.
        await Console(team.run_stream(task="Output the high, low and close prices for MSFT over the last 4 days."))
    finally:
        await agent1.close()
        await model_client.close()

asyncio.run(main())

