import streamlit as st
from openai import OpenAI
from my_agent_group_chat import MyAgentGroupChat
import asyncio
from semantic_kernel.contents.utils.author_role import AuthorRole

st.title("💬 Entourage")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "name": "PersonalAssistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

async def get_question_response(question):
    agent_group_chat = MyAgentGroupChat()
    chat_history = st.session_state['messages']
    
    async for response in agent_group_chat.ask_question(chat_history):
        st.session_state.messages.append({"role": response['role'], 
                                     'name': response['name'], 
                                     "content": response['content']})
        st.chat_message(response['name']).write(response['name'] + ":\n\n" + response['content'])

    st.chat_message('ai').write("Answer from AIs is complete")

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "name": AuthorRole.USER, "content": prompt})
    st.chat_message("user").write(prompt)
    asyncio.run(get_question_response(prompt))