from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
            "Note: You have access to a web search tool that you can use to research relevant information, current trends, and best practices for creating effective tweets. Use the search tool when appropriate to gather up-to-date context, improve accuracy, and apply proven tweet-writing best practices.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
            "Note: You have access to a web search tool that you can use to research relevant information, current trends, and best practices for creating effective tweets. Use the search tool when appropriate to gather up-to-date context, improve accuracy, and apply proven tweet-writing best practices.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOllama(model="gemma4:31b-cloud").bind_tools(
    tools=[TavilySearch(max_retries=2, max_results=5)]
)
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
