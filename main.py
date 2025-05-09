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
from colorama import init, Fore, Style

import pyhtml2md

# Load environment variables
load_dotenv()

# Constants
DEFAULT_BACKGROUND_URL = 'https://marp.app/assets/hero-background.svg'
DEBUG = False

# Prompts
def create_prompts():
    """Create and return prompt templates used by the application."""

    generate_queries_prompt = PromptTemplate.from_template("""
    You are a research assistant. Based on the following text, suggest 3 to 5 precise follow-up search queries
    that could expand or deepen the understanding of the topic. These queries should be clear and relevant to the content,
    keyword-based, and ready to be used in a web search engine.
    Output only the text of the queries, one per line, without numbers or bullet points.

    Text:
    ---
    {content}
    ---

    Search Queries:
    """)

    summarize_prompt = PromptTemplate.from_template("""
    You are a summarization assistant. Read the following content and write a clear, concise summary
    highlighting all the important points. Use bullet points for clarity and organization.

    Content:
    ---
    {content}
    ---

    Summary:
    """)

    # New agenda prompt for creating presentation structure
    presentation_agenda_prompt = PromptTemplate.from_template("""
    You are a presentation structure specialist. Create a detailed agenda for a presentation based on the following research content.

    Research Content:
    ---
    {content}
    ---

    Create an agenda with the following format:
    1. For each slide, use the format:
       ## [Slide Title]
       - Content: Brief summary of what should be on this slide
       - Type: What type of slide this is (title, content, graphics, quote, comparison, etc.)
       - Visual: Any specific visual elements to include (chart, image, diagram, etc.)

    Please include:
    - Title slide with main topic and subtopic
    - Introduction/Agenda slide
    - Main content slides (logically organized)
    - Summary or conclusion slide

    Be thorough and ensure the presentation flows logically, covering all important aspects of the topic.
    
    {tone}

    Today's date is {date}.

    Output format:
    Only output the markdown file with the agenda, without any other text. Stick to the format below:

    ## Slide 1: <title of slide1>
    - Content: <Brief summary of what should be on this slide>
    - Type: <What type of slide this is (title, content, graphics, quote, comparison, etc.)>
    - Visual: <Any specific visual elements to include (chart, image, diagram, etc.)>

    ## Slide 2: <title of slide2>
    - Content: <Brief summary of what should be on this slide>
    - Type: <What type of slide this is (title, content, graphics, quote, comparison, etc.)>
    - Visual: <Any specific visual elements to include (chart, image, diagram, etc.)>
                                                              
    etc.
    """)

    slide_content_prompt = PromptTemplate.from_template("""
    You are a slide content creator specialized in MARP (Markdown Presentation) format.
    Create the content for a single slide based on the provided details.

    ## SLIDE DETAILS:
    Title: {slide_title}
    Type: {slide_type}
    Content Summary: {slide_content}
    Visual Elements: {slide_visual}

    ## CONTEXT:
    This is slide {slide_number} of {total_slides} in a presentation about {topic}
    
    Research Content:
    ---
    {research_content}
    ---
                                                        
    Agenda of the presentation:
    ---
    {agenda}
    ---

    ## RULES:
    - Output valid MARP Markdown for just this one slide
    - Start with "---" to indicate a new slide
    - Use h2-level heading (##) for the slide title
    - Include appropriate Markdown formatting (bullets, numbering, etc.)
    - If this is a title slide, include a subtitle (h3) and presenter information
    - Be informative
    - Include any relevant URLs from the research content if appropriate
    - For code samples, use proper Markdown code blocks
    - Don't mention "slide [number]" in the content
    - Include images from research material, tables, bulleted and ordered lists, links, emoticons and code samples where applicable.
    - Content of the presentation shall be very effective and engaging
    - Content shall follow a logical flow typical of the world-class presentations
    - Image URLs shall be taken directly from the researched content, do not make them up
    - In presentation first slide, always add a cool title (h1 header) and a subtitle (h2 header) related to the topic
    - In presentation first slide, always prepend the speaker name with "> Presented by:" (to quote it)
    - Make sure that content of the slide is relevant to the topic and agenda
    - Make sure that content of the slide fits the size of the slide; create more slides if needed for longer content

    ## TONE:
    {tone}

    Today's date is {date}.

    FORMAT REMINDERS:
    - Start with "---" to indicate a new slide
    - Use standard Markdown formatting (##, -, *, etc.)
    - Do NOT include any additional text or explanations
    - Include image placeholders as ![description](url) if appropriate
    - Do NOT include ```markdown or ``` at the beginning or end of the slide, just the content (in Markdown format as described)
    """)

    marp_prompt = PromptTemplate(input_variables=["bg_url", "content", "tone", "date"], template="""
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
    
    ## Tone:
    {tone}

    Today's date is {date}.

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

    return generate_queries_prompt, summarize_prompt, presentation_agenda_prompt, slide_content_prompt, marp_prompt


def initialize_llms():
    """Initialize and return the LLM based on available credentials."""
    # API Keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_simple_model = os.getenv("OPENAI_SIMPLE_MODEL")
    openai_complex_model = os.getenv("OPENAI_COMPLEX_MODEL")

    # Azure OpenAI configuration
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_simple_deployment_name = os.getenv("AZURE_OPENAI_SIMPLE_DEPLOYMENT_NAME")
    azure_openai_complex_deployment_name = os.getenv("AZURE_OPENAI_COMPLEX_DEPLOYMENT_NAME")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    # Initialize the LLM
    if azure_openai_api_key and azure_openai_endpoint and azure_openai_simple_deployment_name and azure_openai_complex_deployment_name:
        print(Fore.MAGENTA + "Initializing Azure OpenAI API...")
        simple_llm = AzureChatOpenAI(
            azure_deployment=azure_openai_simple_deployment_name,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
        )
        complex_llm = AzureChatOpenAI(
            azure_deployment=azure_openai_complex_deployment_name,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
        )
        return simple_llm, complex_llm
    elif openai_api_key:
        print(Fore.MAGENTA + "Initializing OpenAI API...")
        simple_llm = ChatOpenAI(
            model=openai_simple_model,
            temperature=0,
            openai_api_key=openai_api_key
        )
        complex_llm = ChatOpenAI(
            model=openai_complex_model,
            temperature=0,
            openai_api_key=openai_api_key
        )
        return simple_llm, complex_llm
    else:
        raise ValueError(
            "No valid OpenAI or Azure OpenAI API key configuration found.")


def create_chains(simple_llm, complex_llm):
    """Create and return all LLM chains used by the application."""
    generate_queries_prompt, summarize_prompt, presentation_agenda_prompt, slide_content_prompt, marp_prompt = create_prompts()

    return {
        "generate_queries": generate_queries_prompt | simple_llm,
        "summarize": summarize_prompt | simple_llm,
        "presentation_agenda": presentation_agenda_prompt | simple_llm,
        "slide_content": slide_content_prompt | complex_llm,
        "marp": marp_prompt | complex_llm
    }


def read_tone(file_path):
    """
    Read the tone from a file.
    
    Args:
        file_path: Path to the tone file
        
    Returns:
        String containing the tone instructions, or empty string if file doesn't exist
    """
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(Fore.RED + f"Error reading tone file: {e}")
    return ""


def parse_agenda(agenda_text):
    """
    Parse the agenda text to extract slide details.
    
    Args:
        agenda_text: String containing the presentation agenda
        
    Returns:
        List of dictionaries, each containing details for a slide
    """
    slides = []
    current_slide = None
    
    # Split by slide headers (## ...)
    for line in agenda_text.split('\n'):
        line = line.strip()
        
        # New slide
        if line.startswith('## ') or line.startswith('### ') or line.startswith('Slide'):
            if current_slide:
                slides.append(current_slide)
            
            current_slide = {
                'title': line[3:].strip(),
                'content': '',
                'type': 'content',  # Default type
                'visual': ''
            }
        # Slide content details
        elif current_slide and (line.startswith('Content:') or line.startswith('- Content:') or line.startswith('- **Content:**')):
            current_slide['content'] = line[10:].strip()
        elif current_slide and (line.startswith('Type') or line.startswith('- Type:') or line.startswith('- **Type:**')):
            current_slide['type'] = line[7:].strip()
        elif current_slide and (line.startswith('Visual:') or line.startswith('- Visual:') or line.startswith('- **Visual:**')):
            current_slide['visual'] = line[9:].strip()
    
    # Add the last slide if there is one
    if current_slide:
        slides.append(current_slide)
    
    if DEBUG:
        print(Fore.CYAN + "\n=== PARSED AGENDA ===")
        for i, slide in enumerate(slides, 1):
            print(Fore.CYAN + f"\nSlide {i}:")
            print(Fore.CYAN + f"  Title: {slide['title']}")
            print(Fore.CYAN + f"  Content: {slide['content']}")
            print(Fore.CYAN + f"  Type: {slide['type']}")
            print(Fore.CYAN + f"  Visual: {slide['visual']}")
        print(Fore.CYAN + "\n=== END OF PARSED AGENDA ===\n")
        
    return slides


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


def file_is_text(file_path):
    """
    Validate that the given file exists and is a text file.

    Args:
        file_path: Path to the file

    Returns:
        The validated file path if valid

    Raises:
        argparse.ArgumentTypeError: If the file does not exist or is not a text file
    """
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError(f"The file {file_path} does not exist")

    if not file_path.lower().endswith(('.txt', '.md', '.markdown')):
        raise argparse.ArgumentTypeError(
            f"The file {file_path} is not a recognized text file format (.txt, .md, or .markdown)")

    return file_path


@tool
def web_search_tool(query: str) -> str:
    """Performs a web search for the given query."""
    print(Fore.YELLOW + f"Searching the web for: {query}")

    # Initialize DuckDuckGo Search
    duckduckgo_search = DuckDuckGoSearchResults(
        output_format='list', num_results=5)

    search_results = duckduckgo_search.invoke(query)
    
    if DEBUG:
        print(Fore.CYAN + f"\n=== SEARCH RESULTS FOR: {query} ===")
        for i, result in enumerate(search_results, 1):
            print(Fore.CYAN + f"\nResult {i}:")
            print(Fore.CYAN + f"  Title: {result.get('title', 'No title')}")
            print(Fore.CYAN + f"  Link: {result.get('link', 'No link')}")
            print(Fore.CYAN + f"  Snippet: {result.get('snippet', 'No snippet')}")
        print(Fore.CYAN + "\n=== END OF SEARCH RESULTS ===\n")
    
    return search_results


@tool
def scrape_url_tool(url: str) -> str:
    """Scrapes a webpage and returns markdown text."""
    try:
        print(Fore.YELLOW + f"Extracting markdown from {url}")

        # Load the webpage
        loader = AsyncHtmlLoader(url)
        docs = loader.load()

        if not docs:
            return None

        doc = Document(docs[0].page_content)
        markdown_content = pyhtml2md.convert(doc.summary())
        
        if DEBUG:
            print(Fore.CYAN + f"\n=== SCRAPED CONTENT FROM: {url} ===")
            print(Fore.CYAN + f"Content length: {len(markdown_content)} characters")
            # Print a preview of the content (first 200 characters)
            preview = markdown_content[:200] + "..." if len(markdown_content) > 200 else markdown_content
            print(Fore.CYAN + f"Content preview: {preview}")
            print(Fore.CYAN + "\n=== END OF SCRAPED CONTENT ===\n")
        
        return markdown_content
    except Exception as e:
        print(Fore.RED + f"Error extracting markdown from {url}: {str(e)}")
        return None


class ResearchAssistant:
    """Handles web research and content generation for the presentation."""

    def __init__(self, chains):
        self.chains = chains
        self.seen_urls = set()
        self.research_material = ""
        self.tone = ""
        
    def set_tone(self, tone_path):
        """Set the tone from a file path."""
        self.tone = read_tone(tone_path)
        if self.tone:
            print(Fore.BLUE + "Tone of voice loaded successfully.")
            if DEBUG:
                print(Fore.CYAN + "\n=== TONE OF VOICE ===")
                print(Fore.CYAN + self.tone)
                print(Fore.CYAN + "\n=== END OF TONE OF VOICE ===\n")

    def generate_queries(self, content: str) -> str:
        """Generates several follow-up search queries based on the input content."""
        print(Fore.CYAN + "Generating follow-up search queries...")
        result = self.chains["generate_queries"].invoke({"content": content})
        
        if DEBUG:
            print(Fore.CYAN + "\n=== GENERATED SEARCH QUERIES ===")
            print(Fore.CYAN + result.content)
            print(Fore.CYAN + "\n=== END OF GENERATED SEARCH QUERIES ===\n")
            
        return result

    def summarize(self, content: str) -> str:
        """Summarizes the provided text content into 1-2 paragraphs."""
        print(Fore.CYAN + "Summarizing content...")
        result = self.chains["summarize"].invoke({"content": content})
        
        if DEBUG:
            print(Fore.CYAN + "\n=== CONTENT SUMMARY ===")
            print(Fore.CYAN + result.content)
            print(Fore.CYAN + "\n=== END OF CONTENT SUMMARY ===\n")
            
        return result

    def research_topic(self, query: str) -> str:
        """Conduct research on a topic and collect material."""
        # Initial search
        print(Fore.BLUE + f"Beginning research on: {query}")
        res = web_search_tool.invoke(query)
        for r in res:
            content = scrape_url_tool.invoke(r['link'])
            summarized_content = self.summarize(content)

            if summarized_content.content:
                self.research_material += "\n" + summarized_content.content
                self.seen_urls.add(r['link'])

        # Follow-up searches based on generated queries
        additional_queries = self.generate_queries(self.research_material)

        for q in additional_queries.content.split("\n"):
            # TODO remove this line
            break
            if q.strip():  # Skip empty lines
                print(Fore.YELLOW + q)
                res = web_search_tool.invoke(q)
                for r in res:
                    if r['link'] not in self.seen_urls:
                        content = scrape_url_tool.invoke(r['link'])
                        summarized_content = self.summarize(content)

                        if summarized_content.content:
                            self.research_material += "\n" + summarized_content.content
                            self.seen_urls.add(r['link'])

        print(Fore.GREEN + f"Research complete. Collected {len(self.seen_urls)} sources.")
        
        if DEBUG:
            print(Fore.CYAN + "\n=== RESEARCH MATERIAL STATS ===")
            print(Fore.CYAN + f"Total length: {len(self.research_material)} characters")
            print(Fore.CYAN + f"Number of sources: {len(self.seen_urls)}")
            print(Fore.CYAN + "Sources:")
            for url in self.seen_urls:
                print(Fore.CYAN + f"  - {url}")
            print(Fore.CYAN + "\n=== END OF RESEARCH MATERIAL STATS ===\n")
            
        return self.research_material

    def create_presentation_agenda(self):
        """Generate agenda for the presentation based on researched material."""
        print(Fore.YELLOW + "Creating presentation agenda...")
        from datetime import datetime
        
        agenda_response = self.chains["presentation_agenda"].invoke({
            "content": self.research_material,
            "tone": self.tone,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        
        if DEBUG:
            print(Fore.CYAN + "\n=== PRESENTATION AGENDA ===")
            print(Fore.CYAN + agenda_response.content)
            print(Fore.CYAN + "\n=== END OF PRESENTATION AGENDA ===\n")
            
        print(Fore.GREEN + "Agenda created successfully.")
        return agenda_response.content

    def create_slide(self, slide_info, slide_number, total_slides, topic, agenda):
        """Generate content for a single slide."""
        print(Fore.YELLOW + f"Creating slide {slide_number}/{total_slides}: {slide_info['title']}")
        from datetime import datetime

        slide_response = self.chains["slide_content"].invoke({
            "slide_title": slide_info['title'],
            "slide_type": slide_info['type'],
            "slide_content": slide_info['content'],
            "slide_visual": slide_info['visual'],
            "slide_number": slide_number,
            "total_slides": total_slides,
            "topic": topic,
            "research_content": self.research_material,
            "agenda": agenda,
            "tone": self.tone,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        
        if DEBUG:
            print(Fore.CYAN + f"\n=== SLIDE {slide_number}/{total_slides} CONTENT ===")
            print(Fore.CYAN + f"Title: {slide_info['title']}")
            print(Fore.CYAN + f"Type: {slide_info['type']}")
            print(Fore.CYAN + f"Content: {slide_info['content']}")
            print(Fore.CYAN + f"Visual: {slide_info['visual']}")
            print(Fore.CYAN + "\nGenerated slide content:")
            print(Fore.CYAN + slide_response.content)
            print(Fore.CYAN + f"\n=== END OF SLIDE {slide_number}/{total_slides} CONTENT ===\n")
            
        print(Fore.GREEN + f"Completed slide {slide_number}/{total_slides}")
        return slide_response.content

    def create_presentation(self, background_url: str, topic: str) -> str:
        """Generate a presentation from the researched material."""
        agenda = self.create_presentation_agenda()
        slides_info = parse_agenda(agenda)
        
        print(Fore.BLUE + f"Creating {len(slides_info)} slides based on agenda...")
        from datetime import datetime
        
        # Start with MARP header
        presentation = f"""---
marp: true
theme: default
paginate: false
_class: lead
backgroundColor: #fff
backgroundImage: url('{background_url}')
---

"""
        
        # Generate each slide based on the agenda
        for i, slide_info in enumerate(slides_info, 1):
            slide_content = self.create_slide(slide_info, i, len(slides_info), topic, agenda)
            
            # Ensure slide content starts with separator if not the first slide
            if i > 1 and not slide_content.startswith('---'):
                slide_content = f"---\n{slide_content}"
            elif i == 1 and slide_content.startswith('---'):
                # Remove separator from first slide as we already added the header
                slide_content = slide_content[3:].lstrip()
                
            presentation += f"{slide_content}\n\n"

            import time
            print(Fore.YELLOW + "Sleeping for 10 seconds to cool down OpenAI usage metrics...")
            time.sleep(10)
            print(Fore.YELLOW + "Resuming slide generation...")

        if DEBUG:
            print(Fore.CYAN + "\n=== FINAL PRESENTATION SUMMARY ===")
            print(Fore.CYAN + f"Total slides: {len(slides_info)}")
            print(Fore.CYAN + f"Total content length: {len(presentation)} characters")
            print(Fore.CYAN + "\n=== END OF FINAL PRESENTATION SUMMARY ===\n")
            
        print(Style.BRIGHT + Fore.GREEN + "Presentation creation complete!")
        return presentation


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Langchain Agent for MARP presentations.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-q", "--query", type=str,
                             help="The query to search and generate a MARP presentation for.")
    parser.add_argument("-b", "--background", type=str,
                        help=f"Custom background image URL for the presentation (default: {DEFAULT_BACKGROUND_URL})")
    parser.add_argument("--voice-tone", type=file_is_text, 
                        help="Path to a file containing the tone of voice instructions")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode to show detailed information")
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Initialize colorama
    init(autoreset=True)
    
    # Parse command-line arguments
    args = parse_arguments()
    query = args.query
    background_url = args.background if args.background else DEFAULT_BACKGROUND_URL
    tone_path = args.voice_tone
    
    global DEBUG
    DEBUG = args.debug
    
    if DEBUG:
        print(Style.BRIGHT + Fore.CYAN + "\n=== DEBUG MODE ENABLED ===")

    # Print startup information
    print(Style.BRIGHT + Fore.BLUE + f"\nTopic: {query}")
    print(Style.BRIGHT + Fore.BLUE + f"Background URL: {background_url}")
    if tone_path:
        print(Style.BRIGHT + Fore.BLUE + f"Tone: {tone_path}\n")
    else:
        print(Style.BRIGHT + Fore.BLUE + "No tone file specified\n")

    # Initialize components
    simple_llm, complex_llm = initialize_llms()
    chains = create_chains(simple_llm, complex_llm)

    # Create research assistant
    assistant = ResearchAssistant(chains)
    if tone_path:
        assistant.set_tone(tone_path)

    # Research the topic
    print(Style.BRIGHT + Fore.YELLOW + f"Researching topic: {query}")
    assistant.research_topic(query)

    # Generate presentation
    print(Style.BRIGHT + Fore.YELLOW + "Done researching material. Now creating presentation...")
    presentation_text = assistant.create_presentation(background_url, query)

    # Save the presentation
    safe_filename = sanitize_filename(query)
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{safe_filename}.md"
    with open(output_path, "w") as f:
        f.write(presentation_text)

    print(Style.BRIGHT + Fore.GREEN + f"Presentation saved to {output_path}")
    
    if DEBUG:
        print(Style.BRIGHT + Fore.CYAN + "\n=== DEBUG MODE SUMMARY ===")
        print(Style.BRIGHT + Fore.CYAN + f"Query: {query}")
        print(Style.BRIGHT + Fore.CYAN + f"Background URL: {background_url}")
        print(Style.BRIGHT + Fore.CYAN + f"Output file: {output_path}")
        print(Style.BRIGHT + Fore.CYAN + f"Tone file: {tone_path if tone_path else 'None'}")
        print(Style.BRIGHT + Fore.CYAN + "=== END OF DEBUG MODE ===\n")


if __name__ == "__main__":
    main()
