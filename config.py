# config.py
import os

MODEL_DIR = 'trained_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'dummy_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, 'feature_columns_for_ui.pkl')
TOP_CITIES_PATH = os.path.join(MODEL_DIR, 'top_cities.pkl')
TOP_ZIP_CODES_PATH = os.path.join(MODEL_DIR, 'top_zip_codes.pkl')

DATASET_PATH = "Dataset.csv"
ORGANIZATION_LOGO_PATH = "logo.png"

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = "https://advancedanalyticsopenaikey.openai.azure.com/"
AZURE_OPENAI_API_KEY = "FqFd4DBx1W97MSVjcZvdQsmLhI80hXjl48iWYmZ4W3NutUlWvf0JQQJ99BDACYeBjFXJ3w3AAABACOGl3xo"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_MODEL = "gpt-4o"
