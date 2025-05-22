from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_transcript(url):
    """Fetch and concatenate YouTube video transcript."""
    video_id = extract_video_id(url)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript_list])
    except Exception as e:
        raise Exception(f"Transcript error: {str(e)}")

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    matches = re.search(regex, url)
    if not matches:
        raise ValueError("Invalid YouTube URL")
    return matches.group(1)
