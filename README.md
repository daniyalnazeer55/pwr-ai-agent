# PWR AI Research Assistant

An AI-powered research assistant specialized in Pressurized Water Reactors (PWRs). This agent combines domain-specific knowledge, live web search, and engineering calculations to provide accurate technical support for reactor analysis, safety assessment, and nuclear systems research.

## Overview

This project implements an intelligent agent powered by LangGraph and LLMs that specializes in Pressurized Water Reactor (PWR) technical knowledge. The agent is equipped with three primary tools and a sophisticated decision-making system to guide users through complex nuclear engineering questions.

## Features

- **PWR Domain Expertise**: Trained on a comprehensive vector database of PWR technical documents, safety standards, design reports, anomaly detection research, and nuclear accident studies
- **Multi-Tool Architecture**: Combines local knowledge base search, web search, and Python-based calculations
- **Vector Database Search**: Fast and accurate retrieval from uploaded PWR documents using semantic search
- **Live Web Search**: Access to current research, regulations, and recent publications via DuckDuckGo
- **Engineering Calculator**: Executes Python code for complex reactor physics, thermal hydraulics, and numerical computations
- **Interactive Chat Interface**: Gradio-based web UI for easy interaction with the agent
- **Message History Management**: Maintains conversation context across multiple turns
- **Tool Chain Orchestration**: Intelligent routing between different tools based on query requirements

## System Architecture

The agent is built with a multi-component architecture:

1. **Language Model**: ChatGroq with Llama 3.1 8B Instant as the reasoning engine
2. **Vector Store**: Chroma database with Hugging Face embeddings for semantic search
3. **Tool Layer**: 
   - Academic Web Search (DuckDuckGo)
   - Python Calculator
   - PWR Knowledge Base Retriever
4. **Graph Layer**: LangGraph StateGraph for workflow orchestration
5. **Interface Layer**: Gradio ChatInterface for user interaction

## Installation

### Prerequisites

- Python 3.9 or higher
- Groq API key
- Git

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/daniyalnazeer55/pwr-ai-agent.git
cd pwr-ai-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Groq API key:
```bash
export GROQ_API_KEY="your-api-key-here"
```

## Quick Start

### Running the Interactive Agent

1. Prepare your PWR documents (PDF format):
   - Place all PDF files in the project root directory
   - The agent expects documents like technical reports, safety standards, research papers

2. Run the application:
```bash
python app.py
```

3. Open your browser to the Gradio interface URL

4. Ask your PWR-related questions and the agent will respond with evidence-based answers

### Example Queries

```
"Explain what DNBR stands for and why it matters in PWR thermal-hydraulics."

"Calculate the Reynolds number for PWR coolant flowing through a pipe with diameter 0.20 m, 
velocity 5.0 m/s, density 700 kg/m³, and dynamic viscosity 9 × 10⁻⁵ Pa·s."

"What are the latest safety developments in PWR design for 2026?"

"Search the knowledge base for information about PWR cooling system design."
```

## Configuration

### System Prompt

The agent uses a sophisticated system prompt that enforces:
- Technical accuracy in nuclear engineering
- Proper tool usage (knowledge base first, then web search)
- Citation of sources from retrieved documents
- Transparent reasoning and calculation methodology

### Vector Database Settings

The knowledge base is built with the following settings:
- Chunk size: 750 characters
- Chunk overlap: 120 characters
- Embedding model: BAAI/bge-base-en-v1.5
- Retrieval method: Maximal Marginal Relevance (MMR)
- Top-k results: 3
- Fetch-k value: 8

### LLM Parameters

- Model: llama-3.1-8b-instant
- Temperature: 0.0 (deterministic responses)
- Max tokens per response: Handled by LangGraph recursion limit

## Tools Explained

### 1. Query PWR Knowledge Base

Searches the local vector database of PWR documents using semantic similarity.

**Usage**: Automatically invoked when the query is likely answerable from uploaded documents

**Search Type**: Maximal Marginal Relevance (MMR) for better result diversity

### 2. Academic Web Search

Performs live web searches using DuckDuckGo to find recent publications, regulations, and current information.

**Usage**: When information is recent, unavailable in local documents, or explicitly requested

**Scope**: PWR design, safety standards, nuclear accidents, anomaly detection, thermal hydraulics, reactor physics, and regulatory guidance

### 3. Python Calculator

Executes Python code in a sandboxed environment for numerical computations.

**Usage**: Engineering calculations, reactor physics equations, thermal hydraulic calculations, unit conversions, data analysis

**Safety**: Input is executed in an isolated namespace with error handling

## Tool Usage Policy

The agent follows strict guidelines for tool selection:

1. **Always search the local knowledge base first** for user queries that may be answered using uploaded documents
2. **Use web search only** if information is recent, unavailable locally, or explicitly requested
3. **Use the calculator** for all numerical calculations instead of mental arithmetic
4. **Never fabricate** technical specifications, reactor parameters, or safety limits

## Adding PWR Documents

1. Prepare PDF files with PWR technical content
2. Place them in the project root directory
3. Ensure files match the expected names or modify the `build_or_load_vector_db()` function
4. Delete the existing `./pwr_vector_db` directory if you want to rebuild the database
5. Run the application to rebuild the vector store

## Recommended File Structure

```
pwr-ai-agent/
├── README.md
├── requirements.txt
├── app.py                          # Main application entry point
├── AI_Agent.ipynb                  # Jupyter notebook (development)
├── pwr_vector_db/                  # Chroma vector database (generated)
│   ├── chroma.sqlite3
│   └── data/
├── Documents/                      # Place PWR PDF documents here
│   └── *.pdf
└── venv/                           # Virtual environment
```

## Development

The project is developed as a Jupyter notebook and can be converted to a Python script for production use.

### Converting from Notebook to Python

To run the agent outside of Jupyter:

1. The notebook cells are organized in logical blocks
2. Modify the Google Colab sections to use environment variables
3. Create an `app.py` file that imports and runs the agent

### Key Development Blocks

- Cells 0-4: Dependencies and LLM setup
- Cells 5-7: Agent state and prompt definition
- Cells 8-10: Tool definitions
- Cells 11-13: Vector database setup
- Cells 14-19: Agent graph compilation
- Cells 20-21: Agent invocation and debugging
- Cells 22-27: Demo application with Gradio

## Response Guidelines

The agent follows strict response standards:

1. Uses precise nuclear engineering terminology
2. Never fabricates technical specifications or numerical values
3. Clearly states when information is unavailable
4. References document sources when using the knowledge base
5. Shows methodology and equations for all calculations
6. Distinguishes between retrieved information and explanatory reasoning
7. Synthesizes multiple sources into coherent answers
8. Maintains an objective, evidence-based tone for safety discussions
9. Asks clarifying questions when queries are ambiguous

## Error Handling

The agent includes error handling for:
- Web search failures with detailed error messages
- Python code execution errors with traceback information
- Missing or corrupt documents in the knowledge base
- Invalid LLM responses

## Performance Considerations

- Vector search uses MMR to balance relevance and diversity
- Message history is maintained per conversation
- Recursion limit set to 50 for tool chain execution
- Chunk size of 750 characters balances context and precision

## Dependencies

See `requirements.txt` for complete list. Key packages:

- **LangChain**: LLM framework and tool integration
- **LangGraph**: Agentic workflow orchestration
- **ChatGroq**: Groq API client for LLMs
- **Chroma**: Vector database
- **Hugging Face Transformers**: Embedding models
- **Gradio**: Web UI framework
- **PyPDF**: PDF document loading
- **DuckDuckGo**: Web search integration

## Troubleshooting

### Vector Database Issues
- Delete `pwr_vector_db` directory and restart to rebuild
- Ensure PDFs are in the correct format (not scanned images)

### API Key Issues
- Verify Groq API key is set correctly
- Check API key has necessary permissions

### Memory Issues
- Reduce chunk size in vector database if memory limited
- Reduce max number of retrieved documents

### No Response from Agent
- Check recursion limit in config
- Verify LLM model is available
- Check internet connection for web search

## Safety and Compliance

This agent is designed for research and educational purposes. When using for critical nuclear safety decisions:

1. Always verify agent responses against official documentation
2. Do not rely solely on AI output for safety-critical decisions
3. Consult qualified nuclear engineers for operational decisions
4. Follow all regulatory requirements and standards

## License

MIT License

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Citation

If you use this project in research, please cite:

```
PWR AI Research Assistant
A specialized AI agent for Pressurized Water Reactor technical support
https://github.com/daniyalnazeer55/pwr-ai-agent
```


## Disclaimer

This tool is for informational and research purposes only. It should not be used as a substitute for professional nuclear engineering consultation or official regulatory guidance. Always consult qualified experts and official sources for critical decisions related to nuclear reactor operations or safety.
