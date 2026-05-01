import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def get_llm_client():
    if LLM_PROVIDER == "groq":
        from groq import Groq
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
