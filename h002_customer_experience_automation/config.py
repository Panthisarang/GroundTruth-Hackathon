import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB", "customer_experience_db")

# Collections
USERS_COLLECTION = "users"
ORDERS_COLLECTION = "orders"
STORES_COLLECTION = "locations"
COUPONS_COLLECTION = "coupons"
CHAT_HISTORY_COLLECTION = "chat_history"
EMBEDDINGS_COLLECTION = "embeddings"

# LLM Configuration
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))

# Rate Limiting
MAX_DAILY_LLM_CALLS = int(os.getenv("MAX_DAILY_LLM_CALLS", 20))

# RAG Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model from sentence-transformers
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3
SECRET_KEY="f790dc92ae373b9d1483d5cef5114f6a7fa71e1d4dd82afceb43ac4cb62b221c"

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
