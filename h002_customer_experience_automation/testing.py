import os
from dotenv import load_dotenv
from llm_client import llm_client

# Load environment variables
load_dotenv()

# Test LLM connection
try:
    response = llm_client.call_llm(
        [{"role": "user", "content": "Hello, can you hear me?"}],
        system_prompt="You are a helpful assistant."
    )
    print("✅ LLM Response:", response)
except Exception as e:
    print(f"❌ Error calling LLM: {e}")
    print("Make sure you've set OPENAI_API_KEY in your .env file")