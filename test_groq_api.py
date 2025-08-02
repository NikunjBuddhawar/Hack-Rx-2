import requests
import os

# Replace this with your actual API key (or load from env)
GROQ_API_KEY = "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"

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": "Hello! What's the capital of France?"}
        ],
        "temperature": 0.2
    },
    timeout=60
)

print("Status Code:", response.status_code)
print("Response:", response.json())
