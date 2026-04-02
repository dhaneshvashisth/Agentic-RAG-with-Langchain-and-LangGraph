from dotenv import load_dotenv
from typing import TypedDict, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, START, END

load_dotenv()

COLLECTION_NAME = "youtube_test_collection"

myllm = ChatOpenAI(model="gpt-4o-mini")


client = QdrantClient("localhost", port=6333)

vectordb = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=OpenAIEmbeddings()
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

class AgentState(TypedDict):
    question: str
    context: Optional[str]  
    answer: Optional[str]


def retrieve_node(state: AgentState):

    docs = retriever.invoke(state["question"])

    context = "\n".join([d.page_content for d in docs])

    return {"context": context}




def research_node(state: AgentState):

    prompt = f"""
        You are a research assistant and your name is Aelous. Your job is to answer questions strictly based on the transcript context provided below.
        
        STRICT RULES:
        - Answer ONLY using the information present in the context.
        - If the context does not contain enough information to answer the question, respond exactly with: "The transcript does not contain information about this."
        - Do NOT use any external knowledge, assumptions, or information not present in the context.
        - Do NOT speculate or infer beyond what is explicitly stated.
        - Do NOT answer questions unrelated to the transcript content.
        - If asked about anything else just greet user and say you dont know the answer.
        - If asked about you like who or how are you just tell user your name  and tell you are good and following say how you can help him like you can answer questions about information present in the transcript available only.
        
        CONTEXT (from YouTube transcript):
        {state["context"]}
        
        QUESTION:
        {state["question"]}
        
        ANSWER (based strictly on the context above): """

    response = myllm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("research", research_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "research")
graph.add_edge("research", END)

workflow = graph.compile()


# ----------------------------------
# Autonomous Research Loop
# ----------------------------------

def run_agent():

    print("\nAutonomous YouTube Research Agent")
    print("type 'exit' to stop\n")

    while True:

        question = input("\nAsk research question: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = workflow.invoke({
            "question": question,
            "context": None,
            "answer": None
        })

        print("\nAnswer:\n")
        print(result["answer"])


if __name__ == "__main__":

    run_agent()