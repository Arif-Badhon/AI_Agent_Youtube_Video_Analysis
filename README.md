# YouTube AI Analyzer

A web application that analyzes YouTube and other video links using AI. It generates summaries, mind maps, translations, and provides an interactive Q&A chatbot for any video content.

## Features

- 🎥 **Video Analysis:** Paste a video link to analyze its content.
- 📝 **AI Summarization:** Get concise, medium, or detailed summaries.
- 🧠 **Mind Map Generation:** Visualize concepts and relationships from the video.
- 🌍 **Translation:** Translate summaries into 8 major languages.
- 💬 **AI-Powered Q&A:** Ask questions about the video and get instant answers.
- 📋 **Copy to Clipboard:** Easily copy summaries and translations.

## Tech Stack

- **Frontend:** [Gradio](https://gradio.app/) (Python)
- **Backend:** Python (custom AI agents, transcript processing)
- **Transcription:** YouTube API, Speech-to-Text (for other platforms)
- **Visualization:** Mind map image generation

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Youtube_AI_Agent.git
cd Youtube_AI_Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python src/app/frontend/gradio_app.py
```

The app will launch in your browser.

### 4. (Optional) Run Tests

```bash
bash run_test.sh
```

## Usage

1. Paste a YouTube or other video link in the input box.
2. Select the summary mode (Short, Medium, Detailed).
3. Click **Analyze Video**.
4. View the summary, mind map, and use translation or Q&A features.

## Project Structure

```
src/
  app/
    backend/
      agent.py         # AI logic for summary, mind map, Q&A
      utils.py         # Transcript extraction and helpers
      visualizer.py    # Mind map visualization
    frontend/
      gradio_app.py    # Gradio web interface
requirements.txt
run.sh                # Script for linting, formatting, and testing
run_test.sh           # Script to run tests
README.md
```

## Customization

- To support more video platforms, extend `get_transcript` in utils.py.
- To use a different AI model, update `agent.py`.

## License

MIT License

---