import requests

url = "https://api.dedaluslabs.ai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dsk-test-08203fd590ff-07111fe8102042a07a5cac3421d197ac"
}

payload = {
    "model": "openai/gpt-5",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
}

r = requests.post(url, headers=headers, json=payload)
print(r.status_code)
print(r.json())
