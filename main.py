"""
Langchain Agent with DuckDuckGo Search & Markdown Extraction
This script implements a Langchain agent that:
1. Searches the web using DuckDuckGo Search API
2. Downloads web content from search results
3. Transforms HTML content to Markdown using MarkdownifyTransformer
4. Processes local documents (TXT, MD, PDF, DOCX)
"""

import os
import re
import argparse

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import AsyncHtmlLoader
from readability import Document
from langchain.chains.llm import LLMChain

import pyhtml2md

# Load environment variables
load_dotenv()

# Constants
DEFAULT_BACKGROUND_URL = 'https://marp.app/assets/hero-background.svg'


# Prompts
def create_prompts():
    """Create and return prompt templates used by the application."""

    generate_queries_prompt = PromptTemplate.from_template("""
    You are a research assistant. Based on the following text, suggest 3 to 5 precise follow-up search queries
    that could expand or deepen the understanding of the topic. These queries should be clear and relevant to the content,
    and ready to be used in a web search engine.
    Output only the text of the queries, one per line, without numbers or bullet points.

    Text:
    ---
    {content}
    ---

    Search Queries:
    """)

    summarize_prompt = PromptTemplate.from_template("""
    You are a summarization assistant. Read the following content and write a clear, concise summary
    highlighting the main points. Aim for a maximum of 2 paragraphs.

    Content:
    ---
    {content}
    ---

    Summary:
    """)

    marp_prompt = PromptTemplate(input_variables=["bg_url", "content"], template="""
    ## Persona:
    You are a Cloud Solution Architect that delivers top-notch presentations to your customers on technological-related topics.

    ## Tasks:
    Create a MARP presentation using CommonMark that summarizes all the researched content.
    Researched content is:
    ---
    {content}
    ---

    ## Rules:
    - There is no minimum or maximum number of slides: do all those that are needed to fully explain the topic
    - Only output valid Markdown MARP text, do not add anything else
    - Please be verbose in the text used in the slides
    - Include images from research material, tables, bulleted and ordered lists, links, emoticons and code samples where applicable.
    - Be creative with the graphics, stick with the content that is provided and remember that you can include images and icons full-page if this helps the creative flow.
    - Content of the presentation shall be very effective and engaging
    - Content shall follow a logical flow typical of the world-class presentations
    - Image URLs shall be taken directly from the researched content, do not make them up
    - In presentation first slide, always add a cool title (h1 header) and a subtitle (h2 header) related to the topic
    - In presentation first slide, always prepend the speaker name with "> Presented by:" (to quote it)
    - Do not add any dates for the presentation
    - In the slides, for titles use h2-level headings

    MARP file must start with the following header, unchanged:

    ---
    marp: true
    theme: default
    paginate: false
    _class: lead
    backgroundColor: #fff
    backgroundImage: url('{bg_url}')
    ---
    """)

    return generate_queries_prompt, summarize_prompt, marp_prompt


def initialize_llm():
    """Initialize and return the LLM based on available credentials."""
    # API Keys
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Azure OpenAI configuration
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    # Initialize the LLM
    if azure_openai_api_key and azure_openai_endpoint and azure_openai_deployment_name:
        return AzureChatOpenAI(
            azure_deployment=azure_openai_deployment_name,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
        )
    elif openai_api_key:
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=openai_api_key
        )
    else:
        raise ValueError(
            "No valid OpenAI or Azure OpenAI API key configuration found.")


def create_chains(llm):
    """Create and return all LLM chains used by the application."""
    generate_queries_prompt, summarize_prompt, marp_prompt = create_prompts()

    return {
        "generate_queries": generate_queries_prompt | llm,
        "summarize": summarize_prompt | llm,
        "marp": marp_prompt | llm
        # "generate_queries": LLMChain(llm=llm, prompt=generate_queries_prompt),
        # "summarize": LLMChain(llm=llm, prompt=summarize_prompt),
        # "marp": LLMChain(llm=llm, prompt=marp_prompt)
    }


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


@tool
def web_search_tool(query: str) -> str:
    """Performs a web search for the given query."""
    print(f"Searching the web for: {query}")

    # Initialize DuckDuckGo Search
    duckduckgo_search = DuckDuckGoSearchResults(
        output_format='list', num_results=5)

    search_results = duckduckgo_search.invoke(query)
    return search_results


@tool
def scrape_url_tool(url: str) -> str:
    """Scrapes a webpage and returns markdown text."""
    try:
        print(f"Extracting markdown from {url}")

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


class ResearchAssistant:
    """Handles web research and content generation for the presentation."""

    def __init__(self, chains):
        self.chains = chains
        self.seen_urls = set()
        self.research_material = ""

    def generate_queries(self, content: str) -> str:
        """Generates several follow-up search queries based on the input content."""
        result = self.chains["generate_queries"].invoke({"content": content})
        return result

    def summarize(self, content: str) -> str:
        """Summarizes the provided text content into 1-2 paragraphs."""
        result = self.chains["summarize"].invoke({"content": content})
        return result

    def research_topic(self, query: str) -> str:
        """Conduct research on a topic and collect material."""
        # Initial search
        res = web_search_tool.invoke(query)
        for r in res:
            content = scrape_url_tool.invoke(r['link'])
            if content:
                self.research_material += "\n" + content
                self.seen_urls.add(r['link'])

        # Follow-up searches based on generated queries
        additional_queries = self.generate_queries(self.research_material)

        for q in additional_queries.content.split("\n"):
            if q.strip():  # Skip empty lines
                print(q)
                res = web_search_tool.invoke(q)
                for r in res:
                    if r['link'] not in self.seen_urls:
                        content = scrape_url_tool.invoke(r['link'])
                        if content:
                            self.research_material += "\n" + content
                            self.seen_urls.add(r['link'])

        return self.research_material

    def create_presentation(self, background_url: str) -> str:
        """Generate a presentation from the researched material."""
        presentation = self.chains["marp"].invoke(
            {"bg_url": background_url, "content": self.research_material})
        return presentation.content


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Langchain Agent for MARP presentations.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-q", "--query", type=str,
                             help="The query to search and generate a MARP presentation for.")
    parser.add_argument("-b", "--background", type=str,
                        help=f"Custom background image URL for the presentation (default: {DEFAULT_BACKGROUND_URL})")
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    args = parse_arguments()
    query = args.query
    background_url = args.background if args.background else DEFAULT_BACKGROUND_URL

    # Initialize components
    llm = initialize_llm()
    chains = create_chains(llm)

    # Create research assistant
    assistant = ResearchAssistant(chains)

    # Research the topic
    print(f"Researching topic: {query}")
    assistant.research_topic(query)

    # Generate presentation
    print("Done researching material. Now creating presentation...")
    presentation_text = assistant.create_presentation(background_url)
    print(presentation_text)

    # Save the presentation
    safe_filename = sanitize_filename(query)
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{safe_filename}.md", "w") as f:
        f.write(presentation_text)

    print(f"Presentation saved to outputs/{safe_filename}.md")


if __name__ == "__main__":
    main()
