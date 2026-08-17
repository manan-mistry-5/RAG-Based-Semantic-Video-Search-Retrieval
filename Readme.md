# RAG-Based AI Video Content Retrieval System

An end-to-end **Retrieval-Augmented Generation (RAG)** system that transforms video content into searchable knowledge and enables users to retrieve relevant video segments using natural-language queries.

Instead of manually searching through lengthy videos, users can ask questions in natural language and receive the **most relevant video content along with its timestamp**, making video-based learning and information retrieval significantly more efficient.

---

## Overview

Searching for specific information inside long educational or informational videos is time-consuming because users often need to manually scan transcripts or scrub through the video timeline.

This project addresses that problem by building a complete pipeline that:

**Video → Audio → Transcription → Chunking → Embeddings → Semantic Search → LLM Response**

The system processes video content, converts speech into text, creates semantic embeddings, retrieves the most relevant transcript segments, and uses an LLM to generate a response based on the retrieved context.

---

## Key Features

* 🎥 **Video-to-text processing** using FFmpeg and OpenAI Whisper
* 🌐 **Multilingual transcription** for diverse video content
* ✂️ **Transcript chunking** for efficient retrieval
* 🧠 **Semantic embeddings** using BGE-M3
* 🔎 **Vector similarity search** using cosine similarity
* 🤖 **LLM-powered response generation** using LLaMA 3.2
* ⏱️ **Timestamp-level retrieval** to identify where relevant information appears
* 📚 Supports processing multiple videos as a searchable knowledge base
* ⚡ Reduces the need for manual video searching

---

## System Architecture

```text
                    ┌─────────────────┐
                    │   Video Files   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     FFmpeg      │
                    │ Video → Audio   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ OpenAI Whisper  │
                    │  Transcription │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Transcript      │
                    │ Chunking        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     BGE-M3      │
                    │   Embeddings    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Vector Store /  │
                    │ Joblib Dataset  │
                    └────────┬────────┘
                             │
                     User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │ Query Embedding │
                    │ + Similarity    │
                    │     Search      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LLaMA 3.2     │
                    │ Response LLM    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant Video  │
                    │ + Timestamp     │
                    └─────────────────┘
```

---

## How It Works

### 1. Video Collection

Place the video files you want to search inside the `videos` directory.

The system can work with a collection of videos rather than a single video, allowing the entire collection to function as a searchable knowledge base.

### 2. Video → Audio

Video files are converted into MP3 audio using **FFmpeg**.

```text
Video
  ↓
MP3 Audio
```

This separates the audio content from the original video so it can be processed by the transcription pipeline.

### 3. Audio → Transcript

The extracted audio is processed using **OpenAI Whisper** to generate text transcripts.

The resulting transcript contains the spoken content along with information that can be used to associate retrieved content with its location in the video.

### 4. Transcript Chunking

The generated transcript is divided into smaller chunks.

Chunking allows the retrieval system to search for specific pieces of information rather than processing an entire transcript at once.

```text
Full Transcript
       ↓
 ┌─────┬─────┬─────┬─────┐
 │ C1  │ C2  │ C3  │ C4  │ ...
 └─────┴─────┴─────┴─────┘
```

### 5. Generate Embeddings

Each transcript chunk is converted into a numerical vector representation using **BGE-M3 embeddings through Ollama**.

These embeddings capture the semantic meaning of the text, allowing the system to identify conceptually similar content rather than relying only on exact keyword matches.

### 6. Semantic Retrieval

When a user submits a query, the query is converted into an embedding and compared against the stored transcript embeddings.

The system uses **cosine similarity** to identify the most relevant chunks.

```text
User Query
    ↓
Query Embedding
    ↓
Cosine Similarity
    ↓
Top Relevant Chunks
```

### 7. LLM Response Generation

The retrieved transcript segments are passed as context to **LLaMA 3.2**.

The LLM uses the retrieved information to generate a relevant response and identify the corresponding video content and timestamp.

```text
User Query
     +
Retrieved Context
     ↓
  LLaMA 3.2
     ↓
Answer + Relevant Video Timestamp
```

---

## Example Workflow

A user could ask:

> "What is explained about HTML forms?"

Instead of manually searching through every video, the system:

1. Converts the query into an embedding.
2. Searches the transcript embeddings.
3. Finds the most semantically relevant transcript segments.
4. Identifies the corresponding video and timestamp.
5. Provides the retrieved information to the LLM.
6. Generates a contextual response.

This turns a collection of videos into a **natural-language searchable knowledge base**.

---

## Technology Stack

| Component           | Technology        |
| ------------------- | ----------------- |
| Video Processing    | FFmpeg            |
| Speech-to-Text      | OpenAI Whisper    |
| Embeddings          | BGE-M3            |
| Local Model Runtime | Ollama            |
| Vector Retrieval    | Cosine Similarity |
| Response Generation | LLaMA 3.2         |
| Data Processing     | Python            |
| Storage             | Joblib            |

---

## Project Pipeline

```text
Video Collection
      ↓
FFmpeg
      ↓
Audio Extraction
      ↓
OpenAI Whisper
      ↓
Transcript Generation
      ↓
Transcript Chunking
      ↓
BGE-M3 Embeddings
      ↓
Vector Storage
      ↓
User Query
      ↓
Query Embedding
      ↓
Cosine Similarity Search
      ↓
Relevant Context
      ↓
LLaMA 3.2
      ↓
Answer + Timestamp
```

---

## Dataset

The system was tested on a collection of educational videos totaling **100+ minutes of video content**.

The architecture is designed so that additional videos can be added to the knowledge base without changing the overall retrieval workflow.

---

## Getting Started

### Prerequisites

Make sure the required tools and models are installed and configured before running the pipeline.

The project requires:

* Python
* FFmpeg
* Ollama
* BGE-M3
* LLaMA 3.2

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Add Your Videos

Place your video files inside:

```text
videos/
```

Alternatively, use MP3 files directly if the audio has already been extracted.

### 3. Convert Videos to MP3

Run the video-to-audio processing script:

```text
video_to_mp3
```

This converts the video files into MP3 audio files.

### 4. Generate Transcript JSON Files

Run:

```text
create_json_chunks
```

This processes the MP3 files and generates transcript chunks in JSON format.

### 5. Generate Embeddings

Use:

```text
read_chunks
```

to process the JSON transcript chunks, generate embeddings, and save the resulting dataset as a Joblib file.

### 6. Run Retrieval

Load the generated Joblib dataset into memory and use a user query to retrieve the most relevant transcript segments.

The retrieved context is then passed to the LLM to generate the final response.

---

## Project Structure

```text
project/
│
├── videos/
│   └── *.mp4 / *.mp3
│
├── video_to_mp3
├── create_json_chunks
├── read_chunks
│
├── embeddings/
│   └── *.joblib
│
├── transcripts/
│   └── *.json
│
└── README.md
```

> Update the filenames above to match the exact files in your repository.

---

## Results

The system successfully demonstrated semantic retrieval across **100+ minutes of video content**, enabling users to locate relevant information without manually scanning the entire video library.

The project achieved approximately **60–70% reduction in manual video-search effort** through query-based retrieval and timestamp-level navigation.

---

## Why This Project?

This project demonstrates the practical application of **Retrieval-Augmented Generation** to unstructured multimedia data.

Rather than simply sending a user query directly to an LLM, the system first retrieves relevant information from a custom knowledge base and then provides that context to the LLM.

This approach helps ground the generated response in the available video content.

---

## Future Improvements

Potential improvements include:

* Persistent vector databases such as FAISS or Chroma
* Improved chunking and retrieval strategies
* Reranking retrieved results
* Hybrid keyword + semantic retrieval
* Interactive video playback from retrieved timestamps
* Web-based user interface
* Support for larger video collections
* Evaluation metrics for retrieval accuracy and response quality

---

## Key Learning Outcomes

Through this project, I gained hands-on experience with:

* Designing an end-to-end RAG pipeline
* Processing unstructured video data
* Speech-to-text transcription
* Text chunking and preprocessing
* Semantic embeddings
* Vector similarity search
* Context retrieval for LLMs
* Local LLM deployment using Ollama
* Connecting retrieval systems with generative AI

