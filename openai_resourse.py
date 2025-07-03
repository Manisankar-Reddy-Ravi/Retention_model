from langchain_openai import AzureChatOpenAI  # type: ignore
 
# Azure OpenAI Configuration (Hardcoded API Key)
OPENAI_DEPLOYMENT_ENDPOINT = "https://advancedanalyticsopenaikey.openai.azure.com/"
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_API_VERSION = "2024-12-01-preview"
OPENAI_DEPLOYMENT_NAME = "gpt-4o"
OPENAI_MODEL_NAME = "gpt-4o"
 
#OPENAI_DEPLOYMENT_ENDPOINT = "https://az-openai-document-question-answer-service.openai.azure.com/"
#OPENAI_API_KEY = "5d24331966b648738e5003caad552df8"
#OPENAI_API_VERSION = "2023-05-15"
#OPENAI_DEPLOYMENT_NAME = "az-gpt_35_model"
#OPENAI_MODEL_NAME = "gpt-3.5-turbo"
 
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
 