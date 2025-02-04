from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.agents.strategies import DefaultTerminationStrategy
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.selection.sequential_selection_strategy import *
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel import Kernel

class MyAgentGroupChat:
    def __init__(self):

        # Define the Kernel
        kernel = Kernel()
        added_models = ["qwen2.5-coder:1.5b", "llama3.2:3b", "deepseek-r1:8b", "phi4:latest"]
        for model_id in added_models:
            kernel.add_service(OllamaChatCompletion(service_id=model_id, ai_model_id=model_id))

        qwen_agent = ChatCompletionAgent(
            service_id="qwen2.5-coder:1.5b",
            kernel=kernel, 
            name="QwenAgent", 
            instructions="""You represent answers from the qwen2.5 model.
            You should be as short and concise as possible. Give a 1 or 2 sentence answer only.
            """,
        )

        llama_agent = ChatCompletionAgent(
            service_id="llama3.2:3b",
            kernel=kernel, 
            name="LlamaAgent", 
            instructions="""You represent answers from the llama3.1 model.
            You will add another assistant reply, that is short to help better answer the original user question.
            """,
        )

        deepseek_agent = ChatCompletionAgent(
            service_id="deepseek-r1:8b",
            kernel=kernel, 
            name="DeepseekAgent", 
            instructions="""You represent answers from the deepseek-r1:8b model.
            Even if their is a last message from an assistant please add another one.
            Expand on the last reply from the assistant and make it longer.
            """,
        )

        phi4_agent = ChatCompletionAgent(
            service_id="phi4:latest",
            kernel=kernel, 
            name="Phi4Agent", 
            instructions="""You represent answers from the phi4:latest model.
            Even if their is a last message from an assistant please add another one.
            You will see a chat history from a user and an assistant. Your job is to output whether the question is fully answered (In that case you write "The question is fully answered")
            OR if it is not fully answered, please write the reason why.
            Please give a long and verbose answer.
            """,
        )

        # selection_function = KernelFunctionFromPrompt(
        #     function_name="selection",
        #     prompt=f"""
        #     Determine which participant takes the next turn in a conversation based on the the most recent participant.
        #     State only the name of the participant to take the next turn.
        #     Never select 'User' as the next participant.

        #     Choose only from these participants:
        #     - PersonalAssistant
        #     - CarInventoryAgent
        #     - CarInfoAgent

        #     Message History:
        #     {{{{$history}}}}
        #     """,
        # )

        # TERMINATION_KEYWORD = "yes"

        # termination_function = KernelFunctionFromPrompt(
        #     function_name="termination",
        #     prompt=f"""
        #         Read through the history below. If the original question asked by the user is sufficiently answered then reply {TERMINATION_KEYWORD}.
        #         If the personal assistant has asked for the help from any other agent and they haven't replied yet, then do not reply {TERMINATION_KEYWORD}.

        #         History:
        #         {{{{$history}}}}
        #         """,
        # )

        chat = AgentGroupChat(
            agents=[qwen_agent, llama_agent, deepseek_agent, phi4_agent],
            # selection_strategy=KernelFunctionSelectionStrategy(
            #     agents=[personal_assistant_agent],
            #     function=selection_function,
            #     kernel=kernel,
            #     result_parser=lambda result: str(result.value[0]) if result.value is not None else "PersonalAssistant",
            #     agent_variable_name="agents",
            #     history_variable_name="history",
            # ),
            selection_strategy=SequentialSelectionStrategy(
                agents=[qwen_agent, llama_agent, deepseek_agent, phi4_agent],
            ),
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=4),
        )

        self.chat = chat
        self.kernel = kernel
        self.execution_settings = OllamaChatPromptExecutionSettings()

    async def ask_question(self, input_chat_history):
        for input_chat_message in input_chat_history:
            await self.chat.add_chat_message(ChatMessageContent(role=input_chat_message['role'], name=input_chat_message['name'], content=input_chat_message['content']))

        async for response in self.chat.invoke():
            print("===RESPONSE===")
            print(f"# {response.role} - {response.name or '*'}: '{response.content}'")
            yield {"role": response.role, "name": response.name or '*', "content": response.content}

        print("===DONE===")

    async def ask_question_old(self, input_chat_history):
        chat_history = ChatHistory()

        for message in input_chat_history:
            if message["role"] == "assistant":
                chat_history.add_assistant_message(message["content"])
            elif message["role"] == "user":
                chat_history.add_user_message(message["content"])
        
        chat_completion = self.kernel.get_service("ollamaservice")

        response = await chat_completion.get_chat_message_content(
            chat_history=chat_history,
            settings=self.execution_settings,
        )

        return response.items[0].text

        
