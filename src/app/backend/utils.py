# backend/utils.py (updated)
from youtube_transcript_api import YouTubeTranscriptApi
from langdetect import detect, DetectorFactory
import re
from pytube import YouTube
import whisper
import tempfile
import os

# For consistent results
DetectorFactory.seed = 0

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_transcript_with_lang(url):
    video_id = extract_video_id(url)
    try:
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find manual transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript()
        except:
            # Fallback to auto-generated transcript
            transcript = transcript_list.find_generated_transcript([transcript.language_code for transcript in transcript_list])
        
        # Get text and language
        transcript_text = " ".join([entry['text'] for entry in transcript.fetch()])
        return transcript_text, transcript.language_code
    
    except Exception as e:
        # Fallback to language detection from text
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text, detect(transcript_text)

def get_transcript(url):
    try:
        # First try YouTube captions
        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except:
        # Fallback to audio transcription
        try:
            audio_path = download_audio(url)
            return transcribe_audio(audio_path)
        except Exception as e:
            raise ValueError(f"Both methods failed: {str(e)}")

def download_audio(url):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        audio_stream.download(filename=temp_file.name)
        return temp_file.name
    except Exception as e:
        raise ValueError(f"Audio download failed: {str(e)}")

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")  # Use 'small' or 'medium' for better accuracy
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        raise ValueError(f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
