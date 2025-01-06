# Configuration settings

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
api_key = os.environ['OA_API']
os.environ['OPENAI_API_KEY'] = api_key


class Config:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

    # Collection settings
    COLLECTION_NAME = "knowledge_base"

    # File paths
    DOCUMENT_DIRECTORY = "./data/documents/"  # Main directory for all documents
    LOCAL_QDRANT_PATH = "./local_qdrant"

    # Document type directories (subdirectories)
    PDF_DIRECTORY = os.path.join(DOCUMENT_DIRECTORY, "pdfs")
    PPT_DIRECTORY = os.path.join(DOCUMENT_DIRECTORY, "presentations")

    # Supported file extensions
    SUPPORTED_EXTENSIONS = ['.pdf', '.ppt', '.pptx']

    # Search settings
    SEARCH_LIMIT = 10
    SIMILARITY_THRESHOLD = 0.7

    # Text processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 8
    TIMEOUT = 1800.0