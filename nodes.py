from dotenv import load_dotenv
from langchain import messages
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()


SYSTEM_MESSAGES = """You are a helpful assistant with access to a set of tools. You can use these tools to perform various tasks and provide accurate information to the user. When responding, consider the available tools and their capabilities to enhance your answers."""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGES}, *state["messages"]]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)
