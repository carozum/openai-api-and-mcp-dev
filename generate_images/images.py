from openai import OpenAI
import os
from colorama import Fore
import base64
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.pretty import pprint



client = OpenAI() 

# =================== IMAGES GPT5 API =====================

console=Console()
console.rule("[bold green]Generating Images GPT5with responses [/bold green]")


response = client.responses.create(
    model="gpt-5",
    input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type": "image_generation"}],
)

# Save the image to a file
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

img = response.data[0]

if img.b64_json:
    image_bytes = base64.b64decode(img.b64_json)
    with open("image.png", "wb") as f:
        f.write(image_bytes)

    console.print(Fore.GREEN + "Image saved as image.png")
else:
    console.print(Fore.RED + "No base64 data in response. Try checking response_format or use img.url instead.")

# =================== IMAGES DALLE API =====================
# https://developers.openai.com/api/reference/python/resources/images/methods/generate
# console=Console()
# console.rule("[bold green]Generating Images DALLE with responses [/bold green]")


# img = client.images.generate(
#     # model="gpt-image-1.5",
#     model="dall-e-3",
#     prompt="A realistic baby seagull",
#     n=1,
#     size="1024x1024",
#     response_format="b64_json"
# )

# img = img.data[0]

# if img.b64_json:
#     image_bytes = base64.b64decode(img.b64_json)
#     with open("output_seagull.png", "wb") as f:
#         f.write(image_bytes)


