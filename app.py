#!/usr/bin/env python3
"""
PWR AI Research Assistant
An AI-powered research assistant specialized in Pressurized Water Reactors (PWRs)
"""

import os
import sys
from typing import Annotated, Sequence
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import io


# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Please set your API key: export GROQ_API_KEY='your-key-here'"
    )

DB_DIR = "./pwr_vector_db"
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.0


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = (
    "You are a world class AI Research Assistant specializing in Pressurized Water Reactors (PWRs), "
    "with the expertise of a Senior Nuclear Systems Design Engineer and Nuclear Safety Researcher.\n\n"

    "Your role is to assist users in understanding, analyzing, and explaining technical concepts related to "
    "Pressurized Water Reactors. You should provide scientifically accurate, well structured, and evidence based "
    "responses suitable for engineers, researchers, and graduate students.\n\n"

    "You have access to three external tools:\n"
    "1. A local knowledge base containing uploaded PWR documents, reactor safety standards, design reports, "
    "technical manuals, anomaly detection research papers, and nuclear accident studies.\n"
    "2. A live web search tool for retrieving recent publications, regulations, and information not available "
    "in the local knowledge base.\n"
    "3. A Python calculator for performing engineering calculations and numerical analysis.\n\n"

    "TOOL USAGE POLICY:\n"
    "1. Always search the local knowledge base FIRST whenever the user's question may be answered using the uploaded documents.\n"
    "2. Use the web search tool ONLY if the requested information is recent, unavailable in the local documents, or explicitly requires current information.\n"
    "3. Always use the Python calculator for numerical calculations, reactor physics equations, thermal hydraulic calculations, unit conversions, and engineering computations. Never perform arithmetic mentally.\n\n"

    "When answering using the local knowledge base:\n"
    "• Use only information supported by the retrieved documents.\n"
    "• Do not infer or add technical details that are not explicitly stated.\n"
    "• Cite the document name and page number whenever available.\n"
    "• If multiple retrieved chunks conflict, mention the discrepancy instead of choosing one.\n"

    "YOUR LOCAL KNOWLEDGE BASE CONTAINS INFORMATION ABOUT:\n"
    "• Pressurized Water Reactor design and operation\n"
    "• Reactor safety standards and engineering documentation\n"
    "• Thermal hydraulics and reactor physics\n"
    "• Nuclear safety analysis\n"
    "• Anomaly detection methods for nuclear systems\n"
    "• Fault diagnosis in PWRs\n"
    "• Nuclear accident analysis and mitigation\n"
    "• Research papers related to PWR operation and safety\n\n"

    "RESPONSE GUIDELINES:\n"
    "1. Use precise nuclear engineering terminology.\n"
    "2. Never fabricate technical specifications, reactor parameters, safety limits, or numerical values.\n"
    "3. If the information cannot be found in either the local knowledge base or web search, clearly state that the information is unavailable.\n"
    "4. When answering from the local knowledge base, reference the retrieved document whenever possible.\n"
    "5. For calculations, clearly explain the methodology, define all variables, present equations using LaTeX, and report the final result with appropriate units.\n"
    "6. Distinguish clearly between information retrieved from documents and your own explanatory reasoning.\n"
    "7. If multiple documents provide complementary information, synthesize them into a single coherent answer.\n"
    "8. When discussing nuclear accidents or safety incidents, maintain an objective, evidence based tone and avoid speculation.\n"
    "9. If the user's question is ambiguous, ask a clarifying question before using tools.\n\n"

    "Your highest priorities are technical accuracy, correct tool usage, transparency, and evidence based reasoning."
)


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def initialize_llm():
    """Initialize the ChatGroq LLM"""
    return ChatGroq(
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def setup_academic_web_search():
    """Setup web search tool"""
    ddg_search = DuckDuckGoSearchRun()

    @tool
    def academic_web_search(query: str) -> str:
        """
        Searches the live web for recent and authoritative information related to
        Pressurized Water Reactors (PWRs), including reactor design, safety standards,
        nuclear accidents, anomaly detection, thermal hydraulics, reactor physics,
        regulatory guidance, and current research.

        Use this tool when the required information is not available in the local
        knowledge base or when the user requests recent developments, regulations,
        publications, or news.

        Args:
            query (str): A clear and specific search query.
        """
        try:
            return ddg_search.run(query)
        except Exception as e:
            return f"Search failed with error: {str(e)}"

    return academic_web_search


def setup_python_calculator():
    """Setup Python calculator tool"""
    @tool
    def python_calculator(code: str) -> str:
        """
        Executes Python code in a local environment and returns the standard output (stdout).

        Use this tool whenever numerical computation is required, including engineering
        calculations, reactor physics, thermal hydraulics, heat transfer, fluid flow,
        neutronics, unit conversions, data analysis, or mathematical evaluation.
        Always use this tool instead of performing arithmetic or calculations mentally.

        Args:
            code (str): Valid Python code to execute. Use print() to display the final
            results and any intermediate values that should be returned.
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        redirected_error = sys.stderr = io.StringIO()

        try:
            namespace = {}
            exec(code, namespace)

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            output = redirected_output.getvalue()
            error = redirected_error.getvalue()

            if error:
                return f"Execution Error:\n{error}"

            return (
                output
                if output.strip()
                else "Code executed successfully, but nothing was printed. Remember to use print()."
            )

        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return f"Python Runtime Error: {str(e)}"

    return python_calculator


def setup_vector_store():
    """Build or load the vector database"""
    print("Setting up vector store...")

    # Initialize embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # Check if vector store already exists
    if os.path.exists(DB_DIR):
        print(f"Loading existing vector database from {DB_DIR}")
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings_model
        )
        return vector_store

    # Find all PDF files in current directory
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]

    if not pdf_files:
        print("WARNING: No PDF files found in current directory.")
        print("The knowledge base retriever will return empty results.")
        print("To populate the knowledge base, add PDF files to the project root directory.")
        # Return an empty vector store
        return Chroma(
            embedding_function=embeddings_model,
            persist_directory=DB_DIR
        )

    print(f"Found {len(pdf_files)} PDF files. Building vector database...")

    # Setup text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    all_chunks = []

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file}...")
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()

            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["document"] = pdf_file
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error processing {pdf_file}: {str(e)}")
            continue

    if not all_chunks:
        print("No chunks created from PDFs. Vector store will be empty.")
        return Chroma(
            embedding_function=embeddings_model,
            persist_directory=DB_DIR
        )

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print("Building vector database...")

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings_model,
        persist_directory=DB_DIR
    )

    print("Vector database created and persisted successfully.")
    return vector_store


def setup_knowledge_base_retriever(vector_store):
    """Setup the knowledge base retrieval tool"""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 8
        }
    )

    query_pwr_knowledge_base = create_retriever_tool(
        retriever,
        "query_pwr_knowledge_base",
        "Search the uploaded PWR documents for reactor specifications, safety standards, anomaly detection, nuclear accidents, thermal hydraulics, neutronics, and engineering information. Always use this tool before web search."
    )

    return query_pwr_knowledge_base


# ============================================================================
# AGENT GRAPH SETUP
# ============================================================================

def setup_agent_graph(llm, tools_list):
    """Setup the LangGraph workflow"""

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools_list)

    # Define should_continue function
    def should_continue(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        return "end"

    # Define call_model function
    def call_model(state: AgentState):
        messages = state["messages"]
        formatted_messages = prompt_template.format_messages(messages=messages)
        response = llm_with_tools.invoke(formatted_messages)
        return {"messages": [response]}

    # Create tool node
    tool_node = ToolNode(tools_list)

    # Create workflow graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================================
# CHAT INTERFACE
# ============================================================================

def create_chat_function(pwr_agent):
    """Create the chat function for Gradio"""
    def pwr_chat(message, history):
        messages = []

        # Process conversation history
        if history:
            if isinstance(history[0], dict):
                # New Gradio format
                for item in history:
                    if item["role"] == "user":
                        messages.append(HumanMessage(content=item["content"]))
                    elif item["role"] == "assistant":
                        messages.append(AIMessage(content=item["content"]))
            else:
                # Old tuple format
                for user_msg, assistant_msg in history:
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=assistant_msg))

        messages.append(HumanMessage(content=message))

        try:
            result = pwr_agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 50}
            )

            response = result["messages"][-1].content

            if isinstance(response, list):
                response = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in response
                )

            return str(response)

        except Exception as e:
            return f"Error: {str(e)}"

    return pwr_chat


def launch_interface(pwr_agent):
    """Launch the Gradio chat interface"""
    pwr_chat = create_chat_function(pwr_agent)

    demo = gr.ChatInterface(
        fn=pwr_chat,
        title="PWR Research Assistant",
        description=(
            "AI Research Assistant for Pressurized Water Reactors (PWRs)\n\n"
            "This assistant combines a domain specific knowledge base, live web search, "
            "and Python based engineering calculations to support technical research, "
            "reactor analysis, safety assessment, and evidence based question answering."
        ),
        chatbot=gr.Chatbot(height=500)
    )

    demo.launch(debug=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 80)
    print("PWR AI Research Assistant")
    print("=" * 80)
    print()

    # Initialize LLM
    print("Initializing LLM...")
    llm = initialize_llm()
    print(f"LLM initialized: {MODEL_NAME}")
    print()

    # Setup vector store
    print("Setting up vector database...")
    vector_store = setup_vector_store()
    print()

    # Setup tools
    print("Setting up tools...")
    academic_web_search = setup_academic_web_search()
    python_calculator = setup_python_calculator()
    query_pwr_knowledge_base = setup_knowledge_base_retriever(vector_store)

    tools_list = [python_calculator, query_pwr_knowledge_base, academic_web_search]
    print(f"Tools ready: {len(tools_list)} tools loaded")
    print()

    # Setup agent graph
    print("Compiling agent graph...")
    pwr_agent = setup_agent_graph(llm, tools_list)
    print("Agent compiled successfully")
    print()

    # Launch interface
    print("Launching Gradio interface...")
    print("=" * 80)
    launch_interface(pwr_agent)


if __name__ == "__main__":
    main()
