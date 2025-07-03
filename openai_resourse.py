from langchain_openai import AzureChatOpenAI  # type: ignore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
OPENAI_DEPLOYMENT_ENDPOINT = "https://advancedanalyticsopenaikey.openai.azure.com/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ← updated
OPENAI_API_VERSION = "2024-12-01-preview"
OPENAI_DEPLOYMENT_NAME = "gpt-4o"
OPENAI_MODEL_NAME = "gpt-4o"

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
