from langchain_openai import AzureChatOpenAI  # type: ignore
import os
import streamlit as st

# Load API configuration
OPENAI_DEPLOYMENT_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
OPENAI_MODEL_NAME = st.secrets.get("AZURE_OPENAI_MODEL_NAME", "gpt-4o")  # fallback to gpt-4o

# Initialize Azure OpenAI Client
llm = AzureChatOpenAI(
    temperature=0.1,
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    model_name=OPENAI_MODEL_NAME,
    azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
    openai_api_version=OPENAI_API_VERSION,
    openai_api_key=OPENAI_API_KEY
)

def get_openai_response(messages):
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"
