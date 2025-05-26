import requests
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(url):
    # Handles both youtu.be and youtube.com URLs
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "youtube.com/watch?v=" in url:
        return url.split("v=")[-1].split("&")[0]
    return None

def get_transcript(url):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x['text'] for x in transcript])
    except Exception as e:
        raise RuntimeError(f"Transcript error: {e}")

def get_video_metadata(url):
    try:
        response = requests.get(
            f"https://www.youtube.com/oembed?url={url}&format=json"
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", "No title"),
                "channel": data.get("author_name", "Unknown channel"),
                "thumbnail_url": data.get("thumbnail_url", "")
            }
        return None
    except Exception as e:
        print(f"Metadata error: {e}")
        return None
