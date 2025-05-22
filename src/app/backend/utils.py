from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
import re
import xml.etree.ElementTree as ET

def get_transcript(url):
    """Fetch transcript with robust error handling"""
    video_id = extract_video_id(url)
    try:
        # Attempt to fetch transcript with retries
        for _ in range(3):  # Retry up to 3 times
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                return " ".join([t['text'] for t in transcript_list])
            except ET.ParseError:
                continue  # Retry on XML parse error
        raise Exception("Failed to parse transcript after 3 attempts")
    except (NoTranscriptFound, TranscriptsDisabled):
        raise Exception("No English transcript available")
    except Exception as e:
        raise Exception(f"Transcript error: {str(e)}")

def extract_video_id(url):
    """Improved video ID extraction"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL")
