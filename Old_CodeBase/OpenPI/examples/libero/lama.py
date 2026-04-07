import ollama
import os
from pathlib import Path

client = ollama.Client()
MODEL_NAME = 'qwen3-vl:8b'  
IMAGE_PATH = "/Home/Desktop/sluta/openpi//Desktop/test.png"
PROMPT = "Based on this image, what do you see?"


stream = client.generate(
    model=MODEL_NAME, 
    prompt=PROMPT,
    images=[IMAGE_PATH],
    stream=True
)
        
print("AI: ", end="", flush=True)
for chunk in stream:
    print(chunk['response'], end="", flush=True)            
            