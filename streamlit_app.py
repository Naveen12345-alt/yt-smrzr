# Import required libraries
import streamlit as st  # Web application framework
from langchain_ollama import ChatOllama  # Interface to Ollama language model
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # For breaking text into manageable chunks
from langchain.chains.summarize import (
    load_summarize_chain,
)  # LangChain's summarization pipeline
from langchain.prompts import PromptTemplate  # For creating structured prompts
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)  # For fetching YouTube video transcripts
from typing import Optional  # Type hinting
import re  # Regular expressions for URL parsing


class YouTubeSummarizer:
    """
    A class that handles YouTube video summarization using LangChain and Ollama.
    Processes video transcripts and generates structured summaries.
    """

    def __init__(self):
        """
        Initialize the summarizer with necessary components:
        - Ollama language model
        - Text splitter for chunking
        - Prompt templates for summarization
        - LangChain summarization chain
        """
        # Initialize Ollama model with temperature 0 for consistent outputs
        self.llm = ChatOllama(temperature=0, model="llama3.2")

        # Configure text splitter with large chunks for context preservation
        # chunk_size=10000: Each chunk contains ~10000 characters for comprehensive context
        # chunk_overlap=1000: Overlap ensures continuity between chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],  # Hierarchy of split points for natural text division
        )

        # Template for processing individual chunks of transcript
        # Focuses on extracting key points from each section
        self.map_prompt_template = """
        Summarize the following part of a YouTube video transcript:
        "{text}"
        
        KEY POINTS AND TAKEAWAYS:
        """

        # Template for combining chunk summaries into final output
        # Provides structured format for comprehensive summary
        self.combine_prompt_template = """
        Create a detailed summary of the YouTube video based on these transcript summaries:
        "{text}"
        
        Please structure the summary as follows:
        1. Main Topic/Theme
        2. Key Points
        3. Important Details
        4. Conclusions/Takeaways
        
        DETAILED SUMMARY:
        """

        # Create prompt templates for the chain
        self.map_prompt = PromptTemplate(
            template=self.map_prompt_template, input_variables=["text"]
        )

        self.combine_prompt = PromptTemplate(
            template=self.combine_prompt_template, input_variables=["text"]
        )

        # Initialize the map-reduce chain for efficient processing of long texts
        # map_reduce: Processes chunks independently then combines results
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=self.map_prompt,
            combine_prompt=self.combine_prompt,
            verbose=False,
        )

    def extract_video_id(self, youtube_url: str) -> Optional[str]:
        """
        Extract the video ID from various forms of YouTube URLs.
        Supports standard videos, shorts, and embedded URLs.

        Args:
            youtube_url (str): The YouTube URL to process

        Returns:
            Optional[str]: The video ID if found, None otherwise
        """
        # Patterns handle multiple YouTube URL formats
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?]*)",  # Standard and embed URLs
            r"(?:youtube\.com\/shorts\/)([^&\n?]*)",  # Shorts URLs
        ]

        # Try each pattern until a match is found
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        return None

    def get_transcript(self, video_id: str) -> str:
        """
        Retrieve and process the transcript for a YouTube video.

        Args:
            video_id (str): YouTube video identifier

        Returns:
            str: Concatenated transcript text

        Raises:
            Exception: If transcript cannot be retrieved
        """
        try:
            # Fetch transcript and join all text entries
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript_list])
        except Exception as e:
            raise Exception(f"Error getting transcript: {str(e)}")

    def summarize_video(self, youtube_url: str) -> dict:
        """
        Main pipeline for video summarization.
        Handles the entire process from URL to summary generation.

        Args:
            youtube_url (str): URL of the YouTube video

        Returns:
            dict: Contains status, summary, and video ID or error message
        """
        try:
            # Extract video ID and validate URL
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                return {"status": "error", "message": "Invalid YouTube URL"}

            # Get transcript and process through summarization pipeline
            transcript = self.get_transcript(video_id)
            texts = self.text_splitter.create_documents([transcript])
            summary = self.chain.invoke(texts)

            return {"status": "success", "summary": summary, "video_id": video_id}

        except Exception as e:
            return {"status": "error", "message": str(e)}


def main():
    """
    Main function for the Streamlit web application.
    Sets up the UI and handles user interactions.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="🎥",
        layout="wide",  # Use full width layout
    )

    # Define custom CSS for styling
    st.markdown(
        """
        <style>
        .big-font {
            font-size:24px !important;
            font-weight: bold;
        }
        .summary-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Application header
    st.markdown(
        '<p class="big-font">🎥 YouTube Video Summarizer</p>', unsafe_allow_html=True
    )
    st.markdown("Powered by LangChain and Ollama")

    # Sidebar information
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            """
        This app uses AI to create summaries of YouTube videos.
        
        **Features:**
        - Supports regular YouTube videos and shorts
        - Provides structured summaries
        - Uses local Ollama model
        
        **Note:** Videos must have closed captions/transcripts available.
        """
        )

        st.markdown("### Instructions")
        st.markdown(
            """
        1. Paste a YouTube URL
        2. Click 'Generate Summary'
        3. Wait for the AI to process the video
        """
        )

    # Create two-column layout
    col1, col2 = st.columns([2, 1])  # 2:1 ratio for main content vs video display

    with col1:
        # URL input field
        youtube_url = st.text_input(
            "Enter YouTube URL:", placeholder="https://youtube.com/watch?v=..."
        )

        # Summary generation button and logic
        if st.button("Generate Summary", type="primary"):
            if youtube_url:
                try:
                    # Show processing indicator
                    with st.spinner(
                        "Generating summary... This may take a few moments."
                    ):
                        # Create summarizer instance and process video
                        summarizer = YouTubeSummarizer()
                        result = summarizer.summarize_video(youtube_url)

                        if result["status"] == "success":
                            # Display video thumbnail
                            video_id = result["video_id"]
                            st.image(
                                f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                                use_column_width=True,
                            )

                            # Show formatted summary
                            st.markdown("### Summary")
                            st.markdown(
                                '<div class="summary-box">', unsafe_allow_html=True
                            )
                            st.markdown(result["summary"])
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Add copy functionality
                            st.markdown("### Actions")
                            if st.button("Copy Summary to Clipboard"):
                                st.write("Summary copied!")
                                # Store in session state for clipboard access
                                st.session_state.clipboard = result["summary"]
                        else:
                            # Display error message if processing failed
                            st.error(f"Error: {result['message']}")
                except Exception as e:
                    # Handle unexpected errors
                    st.error(f"An error occurred: {str(e)}")
            else:
                # Prompt user to enter URL if missing
                st.warning("Please enter a YouTube URL")

    # Video display column
    with col2:
        if youtube_url and "video_id" in locals():
            st.markdown("### Original Video")
            st.video(youtube_url)


# Entry point of the application
if __name__ == "__main__":
    main()
