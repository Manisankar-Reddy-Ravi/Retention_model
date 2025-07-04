from langchain_openai import AzureChatOpenAI
import streamlit as st
import os

# Try to load from Streamlit secrets first (for Streamlit Cloud)
try:
    OPENAI_DEPLOYMENT_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
    OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
    OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
    OPENAI_MODEL_NAME = st.secrets["AZURE_OPENAI_MODEL_NAME"]
except Exception:
    # Fallback to local .env (for local development)
    from dotenv import load_dotenv
    load_dotenv()

    OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")

# Initialize AzureChatOpenAI
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
