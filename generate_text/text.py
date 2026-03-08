# https://developers.openai.com/api/docs
from openai import OpenAI
import os
from colorama import Fore
import base64
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.pretty import pprint


client = OpenAI()

# =================== RESPONSES API =====================

console=Console()
console.rule("[bold green]Generating Content with responses [/bold green]")

response = client.responses.create(
    model="gpt-4.1",
    input="Write a two-sentence horror story about a haunted houses.",
    temperature=0.7
)

print(response.output_text)

# =================== COMPLETIONS API =====================

console=Console()
console.rule("[bold green]Generating Content with completions [/bold green]")

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a one-sentence greeting story about a nice woman.",
    temperature=0.7,
    max_tokens=60
)

print(response.choices[0].text)

# =================== CHAT COMPLETIONS API =====================

console=Console()
console.rule("[bold green]Generating Content with chat completions [/bold green]")

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role":"user", "content":"Write a one-sentence greeting story about a nice woman."}],
    temperature=0.7,
    max_tokens=60
)

print(response.choices[0].message.content)