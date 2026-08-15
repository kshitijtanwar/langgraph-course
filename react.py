from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    Multiply the provided number by 3.

    Args:
        num: The numeric value to multiply by 3.

    Returns:
        The input number multiplied by 3 as a float.
    """
    return float(num * 3)


tools = [TavilySearch(max_retries=1), triple]


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash").bind_tools(tools)
