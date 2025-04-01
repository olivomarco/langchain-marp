"""
Langchain Agent with Brave Search & Markdown Extraction
This script implements a Langchain agent that:
1. Searches the web using Brave Search API
2. Downloads web content from search results
3. Transforms HTML content to Markdown using MarkdownifyTransformer
"""

from langchain_core.prompts import PromptTemplate
import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from readability import Document

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

import pyhtml2md
import time
import argparse

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Langchain Agent for MARP presentations.")
parser.add_argument("-q", "--query", type=str, required=True, help="The query to search and generate a MARP presentation for.")
args = parser.parse_args()

# Use the query from the command-line arguments
query = args.query

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize Brave Search
brave_search_tool = BraveSearch.from_api_key(
    api_key=BRAVE_API_KEY, search_kwargs={"count": 10})


def search_web(query: str) -> List[Dict]:
    """Search the web using Brave Search API."""
    time.sleep(1)   # Add a delay to avoid rate limiting on free tier
    print(f"Searching the web for: {query}")
    search_results = brave_search_tool.run(query)
    return search_results


def extract_markdown_from_url(url: str) -> Optional[str]:
    """
    Download content from a URL and convert it to markdown.

    Args:
        url: The URL to download content from

    Returns:
        Markdown content if successful, None otherwise
    """
    try:
        # Load the webpage
        loader = AsyncHtmlLoader(url)
        docs = loader.load()

        if not docs:
            return None

        doc = Document(docs[0].page_content)

        return pyhtml2md.convert(doc.summary())
    except Exception as e:
        print(f"Error extracting markdown from {url}: {str(e)}")
        return None


def sanitize_filename(query: str) -> str:
    """
    Sanitize the query to create a safe filename.

    Args:
        query: The input query string.

    Returns:
        A sanitized string safe for use as a filename.
    """
    # Replace spaces with underscores and remove invalid characters
    return re.sub(r'[^a-zA-Z0-9_\-]', '', query.replace(' ', '_'))


# Initialize the LLM
if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT_NAME:
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-10-21",
        temperature=0
    )
elif OPENAI_API_KEY:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
else:
    raise ValueError(
        "No valid OpenAI or Azure OpenAI API key configuration found.")

# Create the agent
memory = MemorySaver()
model = llm
tools = [
    Tool(
        name="BraveSearch",
        description="Useful for searching the web for information.",
        func=search_web,
    ),
    Tool(
        name="MarkdownExtractor",
        description="Extract markdown content from a webpage URL.",
        func=extract_markdown_from_url,
    )
]
agent_executor = create_react_agent(model, tools, checkpointer=memory)


# Get the prompt to use - you can modify this!
prompt = PromptTemplate(input_variables=["agent_scratchpad", "input"], template="""
Create a MARP presentation using CommonMark based on research. Research all documents that you need to fully understand a topic.
Only output valid Markdown MARP text, do not add anything else.
Please be a bit verbose in the slides, include images, tables, bulleted and ordered lists, links, emoticons and code samples where applicable.
Be creative with the graphics, stick with the content that is provided and remember that you can include images and icons full-page if this helps the creative flow.
Content of the presentation shall be very effective and engaging, and shall follow a logical flow typical of the world-class presentations.

MARP file must start with the following header, unchanged:

---
marp: true
theme: default
paginate: true
_class: lead
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
---

Topic to research is: {input}

{agent_scratchpad}
""")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": query})

# Sanitize the query for the filename
safe_query = sanitize_filename(query)

# Save result['output'] to a file in the outputs/ folder
with open(f"outputs/{safe_query}.md", "w") as f:
    f.write(result['output'])
