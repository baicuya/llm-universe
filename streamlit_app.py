import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
# sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中
# from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())    # read local .env file


#export OPENAI_API_KEY=
#os.environ["OPENAI_API_BASE"] = 'https://api.chatgptid.net/v1'
# zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
# llm = ChatZhipuAI(model="glm-4", temperature=0, api_key=zhipuai_api_key)


def generate_response(input_text):
    # llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    llm = ChatZhipuAI(model="glm-4", temperature=0, api_key=zhipuai_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output


# Streamlit 应用程序界面
def main():
    st.title('🌙(>^ω^<) your assistant(>^ω^<)🌙')
    zhipuai_api_key = st.sidebar.text_input('ZHIPUAI_API_KEY', type='password')

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "chat":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
