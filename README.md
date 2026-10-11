# Thaif-Embeddings

A Python desktop application that compares the semantic similarity between two sentences using NVIDIA's Nemotron embedding model.

## Features
- Generates text embeddings using NVIDIA's API
- Measures sentence similarity using cosine similarity
- Displays similarity scores
- Simple desktop GUI built with Tkinter

## Technologies
- Python
- NVIDIA Nemotron Embeddings
- OpenAI Python SDK
- NumPy
- Tkinter

## Setup
1. Install dependencies:
   `pip install -r requirements.txt`
2. Create a `.env` file and add your NVIDIA API key:
   `NVIDIA_API_KEY=your_api_key_here`
3. Run the application:
   `python main.py`

**Security:** Never upload your `.env` file or expose your API key.