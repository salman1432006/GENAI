import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Locate .env
env_path = BASE_DIR / ".env"

print("Project directory:", BASE_DIR)
print("Checking .env file:", env_path)
print("Does .env exist?", env_path.exists())

# Load environment variables
load_dotenv(dotenv_path=env_path)

# Read API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("ERROR: API key not found! Check your .env file.")

print("API key successfully loaded!")

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Prompt
prompt = """
Explain Artificial Intelligence in simple words.
Give five real-world examples.
"""

try:
    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    print("\n========== LLM RESPONSE ==========\n")
    print(response.output_text)
    print("\n==================================")

except Exception as e:
    print("API Error:", e)
