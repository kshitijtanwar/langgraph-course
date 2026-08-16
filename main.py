from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflection_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"
LAST = -1


def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECT


def generate_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    result = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=result.content)]}


builder = StateGraph(MessageGraph)

builder.add_node(GENERATE, generate_node)
builder.set_entry_point(GENERATE)
builder.add_node(REFLECT, reflection_node)

builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})

builder.add_edge(REFLECT, GENERATE)


app = builder.compile()
# app.get_graph().draw_mermaid_png(output_file_path="flow.png")


def main():
    input = [HumanMessage(content="""I'm learning about Reflection Agents 

This tweet was generated and refined by a Reflection Agent—an AI agent that critiques its own output and iteratively improves it.
------------------------------------------------
At the end, return only the final polished tweet that I can post.
            """)]
    result = app.invoke({"messages": input})
    print(result["messages"][LAST].content)
    print("\nExecution completed!")


if __name__ == "__main__":
    main()
