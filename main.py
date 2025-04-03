"""
Langchain Agent with DuckDuckGo Search & Markdown Extraction
This script implements a Langchain agent that:
1. Searches the web using DuckDuckGo Search API
2. Downloads web content from search results
3. Transforms HTML content to Markdown using MarkdownifyTransformer
4. Processes local documents (TXT, MD, PDF, DOCX)
"""

from langchain_core.prompts import PromptTemplate
import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from readability import Document

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

import pyhtml2md
import time
import argparse
import pathlib

# Load environment variables
load_dotenv()

# Default background image URL
DEFAULT_BACKGROUND_URL = 'https://marp.app/assets/hero-background.svg'

# Function to create prompt templates with dynamic background image URL


def create_search_prompt(bg_url):
    return PromptTemplate(input_variables=["agent_scratchpad", "input"], template=f"""
Create a MARP presentation using CommonMark based on research. Research all documents that you need to fully understand a topic.
Only output valid Markdown MARP text, do not add anything else.
Please be a bit verbose in the slides, include images from research material, tables, bulleted and ordered lists, links, emoticons and code samples where applicable.
Be creative with the graphics, stick with the content that is provided and remember that you can include images and icons full-page if this helps the creative flow.
Content of the presentation shall be very effective and engaging, and shall follow a logical flow typical of the world-class presentations.
Image URLs shall be taken directly from the researched content, do not make them up.

MARP file must start with the following header, unchanged:

---
marp: true
theme: default
paginate: true
_class: lead
backgroundColor: #fff
backgroundImage: url('{bg_url}')
---

Topic to research is: {{input}}

{{agent_scratchpad}}
""")


def create_url_prompt(bg_url):
    return PromptTemplate(input_variables=["agent_scratchpad", "input"], template=f"""
Create a MARP presentation using CommonMark based on the following url: {{input}}.
First, download and extract the content from the url. Then, elaborate the slides.
Only output valid Markdown MARP text, do not add anything else.
Please be a bit verbose in the slides, include images, tables, bulleted and ordered lists, links, emoticons and code samples where applicable.
Be creative with the graphics, stick with the content that you extract and remember that you can include images and icons full-page if this helps the creative flow.
Content of the presentation shall be very effective and engaging, and shall follow a logical flow typical of world-class presentations.
Image URLs shall be taken from the url scraped, do not make them up.

MARP file must start with the following header, unchanged:

---
marp: true
theme: default
paginate: true
_class: lead
backgroundColor: #fff
backgroundImage: url('{bg_url}')
---

{{agent_scratchpad}}
""")


def create_document_prompt(bg_url):
    return PromptTemplate(input_variables=["agent_scratchpad", "input"], template=f"""
Create a MARP presentation using CommonMark based on the following document: {{input}}
Only output valid Markdown MARP text, do not add anything else.
Please be a bit verbose in the slides, include, tables, bulleted and ordered lists, links, emoticons and code samples where applicable.
Be creative with the graphics, stick with the content that is provided.
Content of the presentation shall be very effective and engaging, and shall follow a logical flow typical of world-class presentations.


MARP file must start with the following header, unchanged:

---
marp: true
theme: default
paginate: true
_class: lead
backgroundColor: #fff
backgroundImage: url('{bg_url}')
---

{{agent_scratchpad}}
""")


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Langchain Agent for MARP presentations.")
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-q", "--query", type=str,
                         help="The query to search and generate a MARP presentation for.")
input_group.add_argument("-u", "--url", type=str,
                         help="The URL to scrape and generate a MARP presentation for.")
input_group.add_argument("-d", "--document", type=str,
                         help="Local document (TXT, MD, PDF, DOCX) to extract content from for the presentation.")
parser.add_argument("-b", "--background", type=str,
                    help="Custom background image URL for the presentation (default: https://marp.app/assets/hero-background.svg)")
args = parser.parse_args()

# Use the query, url or document from the command-line arguments
query = args.query
url = args.url
document_path = args.document
background_url = args.background if args.background else DEFAULT_BACKGROUND_URL

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize DuckDuckGo Search
duckduckgo_search_tool = DuckDuckGoSearchRun()

def search_web(query: str) -> List[Dict]:
    """Search the web using DuckDuckGo Search API."""
    print(f"Searching the web for: {query}")
    search_results = duckduckgo_search_tool.run(query)
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
        #return doc.summary()
    except Exception as e:
        print(f"Error extracting markdown from {url}: {str(e)}")
        return None


def extract_text_from_document(file_path: str) -> Optional[str]:
    """
    Extract text content from various document formats.

    Args:
        file_path: Path to the document file

    Returns:
        Text content if successful, None otherwise
    """
    try:
        file_extension = pathlib.Path(file_path).suffix.lower()

        if file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension in ['.md', '.markdown']:
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        docs = loader.load()

        if not docs:
            return None

        # Combine all document pages/sections
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error extracting text from document {file_path}: {str(e)}")
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

# Define the tools
search_tool = Tool(
    name="Search",
    description="Useful for searching the web for information.",
    func=search_web,
)
extract_markdown_from_url_tool = Tool(
    name="MarkdownExtractor",
    description="Useful to extract markdown content from a webpage URL.",
    func=extract_markdown_from_url,
)
extract_text_from_document_tool = Tool(
    name="TextExtractor",
    description="Useful to extract text content from a local document.",
    func=extract_text_from_document,
)

# Process based on input type
if document_path:
    # Process document input
    prompt = create_document_prompt(background_url)
    tools = [extract_text_from_document_tool]
    input_value = document_path

    # Use document name for the filename
    doc_name = os.path.basename(document_path)
    safe_filename = sanitize_filename(os.path.splitext(doc_name)[0])
elif url:
    # Process URL input
    prompt = create_url_prompt(background_url)
    tools = [extract_markdown_from_url_tool]
    input_value = url

    # Use URL for the filename
    safe_filename = sanitize_filename(url)
else:
    # Process search query input
    prompt = create_search_prompt(background_url)
    tools = [search_tool, extract_markdown_from_url_tool]
    input_value = query

    # Use query for the filename
    safe_filename = sanitize_filename(query)

# Create and execute the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": input_value})
result_output = result['output']

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Save result to a file in the outputs/ folder
with open(f"outputs/{safe_filename}.md", "w") as f:
    f.write(result_output)
