# Script Generation System

## Overview

This project is a modular, multi-agent research and content generation pipeline built with **LangGraph** and **LangChain**. It autonomously discovers trending topics, conducts deep research using web search and full-page extraction, synthesizes comprehensive reports, generates critique and iterative improvements, and finally produces creative image prompts for visual content.

**Key Features:**
- **Topic Scout:** Finds trending topics from Reddit and Twitter/X or lets the user specify a topic.
- **Planner Agent:** Strategically plans research and crafts high-quality search queries.
- **Searcher Agent:** Executes web searches using Tavily, robustly handling errors and maximizing result quality.
- **Extractor Agent:** Pulls full-page content from URLs using Tavily Extract.
- **Writer Agent:** Synthesizes detailed, structured reports from all gathered material.
- **Judge Agent:** Critiques reports as a target audience, suggesting improvements and new research angles.
- **Iterative Loop:** Planner, Searcher, Extractor, Writer, and Judge collaborate for three research iterations.
- **Image Prompt Generator:** Creates detailed, creative text-to-image prompts for each source in the report.

## Architecture

```
User/Scout → Planner → Searcher → Extractor → Writer → [Judge → Planner] (loop x2) → Final Writer → Image Prompt Generator
```

- **Memory:** All agents share state, so context and history are preserved across iterations.
- **Streaming:** Outputs from each agent are shown in real time for transparency and debugging.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/scriptgen.git
cd scriptgen
```

### 2. Install Dependencies

```bash
pip install -U langchain langchain-tavily python-dotenv
```

### 3. API Keys

- **Tavily:** [Get your API key here](https://app.tavily.com/)
- **Google Gemini:** [Get your API key here](https://aistudio.google.com/app/apikey)

Create a `.env` file in your project root:

```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=your-gemini-key
```

## Usage

```bash
python workflow.py
```

- **Choose a topic:**  
  - Enter manually  
  - Or let the Topic Scout agent find a trending topic for you

- **Watch in real time:**  
  - Each agent’s output is printed as the workflow progresses

- **Outputs:**  
  - `final_research_report_.md` — a comprehensive Markdown report
  - `image_prompts_for_.md` — detailed text-to-image prompts for each report source

## Project Structure

```
scriptgen/
├── workflow.py         # Main orchestrator script
├── image_prompt.py       # Image prompt generation module
├── .env                           # API keys
└── README.md                      # This file
```

## Key Technologies

- **LangGraph** — for building stateful, multi-agent workflows
- **LangChain** — for LLM orchestration and tool integration
- **Tavily Search & Extract** — for web search and full-page content extraction
- **Google Gemini** — for all LLM-powered agents (planning, writing, critique, image prompts)


## License

MIT License

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Tavily](https://tavily.com/)
- [Google Gemini](https://aistudio.google.com/)
