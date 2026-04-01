import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(model="gpt-5.4", input="Say 'API key works!'")

print(response.output[0].content[0].text)
