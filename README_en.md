# MagicExamUtilities

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
English | [简体中文](README.md)

---

## Project Overview

An AI-powered tool for final exam preparation that automatically generates study notes from course materials.

**Note: This tool is for study assistance only and does not contain any cheating functionality.**

### Key Features

| Feature | Description |
|---------|-------------|
| **PDF Processing** | Generate structured study notes from lecture slides |
| **Audio Processing** | Generate study notes from review session recordings |
| **Ask AI** | Answer questions based on processed content with knowledge base isolation |
| **Knowledge Base Management** | Organize PDFs by subject with independent indexing and search |
| **Smart Caching** | File hash-based caching to avoid redundant processing |
| **Chunked Summarization** | Handle ultra-long texts beyond model context window limits |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended Python package manager)
- poppler-utils (dependency for PDF processing)

### One-Click Installation (Recommended)

```bash
# Clone the project
git clone https://github.com/yourusername/MagicExamUtilities.git
cd MagicExamUtilities

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate.sh  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### Install poppler

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download and install [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

---

## Configuration Guide

### 1. Get API Key

This tool requires an LLM service with OpenAI-compatible API. Recommended providers:

| Provider | Website | Recommended Model |
|----------|---------|-------------------|
| SiliconFlow | https://siliconflow.cn | Qwen3-Next-80B-A3B-Instruct |
| DeepSeek | https://platform.deepseek.com | DeepSeek-V3 |

### 2. Configure Environment Variables

Copy the example configuration file and fill in your API information:

```bash
cp .env.example .env
```

Edit the `.env` file with the following required configurations:

```bash
# === OCR Configuration (PDF Processing) ===
OCR_API_URL=https://api.siliconflow.cn/v1
OCR_API_KEY=your_api_key_here
OCR_MODEL=deepseek-ai/DeepSeek-OCR

# === Summarization Configuration (Note Generation) ===
SUMMARIZATION_API_URL=https://api.siliconflow.cn/v1
SUMMARIZATION_API_KEY=your_api_key_here
SUMMARIZATION_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct

# === ASR Configuration (Audio Processing, Optional) ===
ASR_API_URL=https://api.siliconflow.cn/v1/audio/transcriptions
ASR_API_KEY=your_api_key_here
ASR_MODEL=TeleAI/TeleSpeechASR

# === Ask AI Configuration (Q&A Feature) ===
ASK_AI_API_URL=https://api.siliconflow.cn/v1
ASK_AI_API_KEY=your_api_key_here
ASK_AI_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
```

### 3. Verify Configuration

Run the following command to verify your configuration:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OCR Model:', os.getenv('OCR_MODEL'))"
```

---

## Usage

### GUI Mode (Recommended)

```bash
python main.py --gui
```

A web interface will open in your browser with the following features:

#### 1. PDF Processing
- Upload PDF files (multiple files supported)
- Optional output directory selection
- Enable caching to speed up reprocessing
- Advanced option: Enable chunked summarization for ultra-long documents

#### 2. Audio Processing
- Upload MP3 recording files
- Automatic transcription and note generation

#### 3. Ask AI
- Enter questions, AI retrieves and answers from processed content
- Optional knowledge base selection to limit search scope

#### 4. Knowledge Base Management
- Create knowledge bases (organized by subject)
- Add/remove PDF documents
- Rebuild search indices

### CLI Mode

```bash
# Process PDFs
python main.py --type pdf --input slides/*.pdf --output ./notes

# Process audio
python main.py --type audio --input recording.mp3 --output ./notes

# Specify cache directory
python main.py --type pdf --input file.pdf --output ./notes --dump-dir ./cache
```

---

## FAQ

### Q: Why is the generated note quality poor?

**A:** This usually depends on:
1. **Model Selection** - Recommended models: `Qwen3-Next-80B-A3B-Instruct` or `DeepSeek-V3`
2. **Audio Quality** - Noisy recording environments lead to fragmented STT results
3. **PDF Quality** - Scanned documents or poor image quality affect OCR accuracy

### Q: What if STT request fails?

**A:** This tool currently only supports SiliconFlow's ASR API. If you need to use other services, please verify the `ASR_API_URL` in your `.env` file is correct.

### Q: How to process ultra-long documents?

**A:** In the GUI, expand "Advanced Options" and check "Enable Chunked Summarization". Documents will be automatically chunked to process beyond the model's context window limit.

### Q: Where are the cache files stored?

**A:** By default in the `./cache/` directory, containing:
- `ocr/` - OCR result cache
- `stt/` - STT result cache
- `knowledge_bases.json` - Knowledge base configuration
- `kb_indices/` - Knowledge base indices

---

## Feature Details

### Knowledge Base Management

Create knowledge bases to organize PDFs by subject for independent searching:

```
1. Go to "Knowledge Base Management" tab
2. Create a knowledge base
3. Select relevant PDF documents to add
4. System automatically builds semantic index
5. In Ask AI, select the knowledge base for Q&A
```

### Chunked Summarization

When a document exceeds the model's context window:

1. Document is automatically split by paragraphs/sentences
2. Each chunk is summarized independently
3. All summaries are merged
4. If merged result is still too long, a second summarization pass is performed

---

## Development Roadmap

- [ ] Past Exam & Problem Bank: Generate review questions and answer explanations from past assignments
- [ ] More ASR service adapters
- [ ] Export to Anki card format

---

## License

This project is open source under [GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

**Don't be an asshole!**

---

## Changelog

See [Changelog](#changelog) section below.

### [0.7.1] - 2024-12-30

**Fixed: Qwen Model Markdown Code Block Wrapper Issue**
- Fixed issue where Qwen series models automatically wrap output in ```markdown code blocks
- Added `_strip_outer_markdown_block()` function to automatically detect and remove wrapper
- Preserves legitimate code blocks within documents

**Fixed: VLM OCR Hallucinating Image URLs**
- Fixed issue where VLM generates fake URLs when processing documents containing images
- Updated OCR prompts to explicitly prohibit generating any URLs or file paths
- Added image processing rules to guide models to describe images in text

**Improved: OCR and Summarization Prompt Refactoring**
- Rewrote Generic VLM OCR prompt with clearer structure
- Rewrote OCR summarization prompt (PDF-specific) emphasizing structured organization
- Rewrote STT summarization prompt (audio-specific) emphasizing removing speech redundancies
- Added detailed special content handling table
- Optimized user prompts for conciseness and clarity

### [0.7.0] - 2024-12-30

**Added: Knowledge Base Management System**
- Added `utilities/knowledge_base/` module
- Support for creating knowledge bases by subject
- Independent PDF document management per knowledge base
- Independent semantic search indexing per knowledge base
- GUI "Knowledge Base Management" tab

**Added: Ask AI Feature**
- Added `utilities/workers/ask_ai_worker.py`: LangChain-based AI Q&A worker
- Integrated retrieval tools for fetching relevant content from cached PDF/audio
- Support for knowledge base isolated search
- GUI "Ask AI" tab with Q&A interface

**Added: Chunked Summarization**
- SummarizationWorker now supports chunked summarization mode for ultra-long texts
- Smart chunking: Split by paragraphs and sentences while maintaining semantic integrity
- Two-stage summarization: Chunk summarization followed by merged summary
- GUI "Advanced Options" with configurable chunk threshold

**Improved: GUI Layout Refactor**
- Refactored to vertically-flowing centered layout
- Accordion components for collapsing advanced options
- Clear buttons for easy form reset
- Improved visual hierarchy and spacing

**Architecture Changes:**
- `utilities/retrieval/`: New retrieval module
  - `retriever.py`: CacheRetriever with keyword and semantic search
  - `tools.py`: LangChain tool definitions
  - `cache_loader.py`: Cache loader
- `utilities/workers/ask_ai_worker.py`: Ask AI Worker

### [0.6.0] - 2024-12-30

**Added: Gradio Web GUI**
- Added `gui.py`: Gradio-based web interface
- `main.py`: Added `--gui` parameter for GUI mode
- GUI provides friendly interface for PDF and audio processing
- Supports drag-and-drop file upload, real-time progress display, result preview and download

**Architecture Changes:**
- `gui.py`: GradioApp class managing GUI lifecycle and Worker initialization
- `main.py`: Supports GUI/CLI dual mode startup

### [0.5.0] - 2024-12-30

**Added: OCR Cache and Structured Data Storage**
- Added `utilities/models.py`: Structured data models (OCRResult, OCRSection, PDFCache)
- Added `utilities/cache.py`: File hash-based OCR cache manager
- OCRWorker: Added `process_images_structured()` method returning structured OCRResult
- main.py PDF processing pipeline now supports caching to avoid redundant OCR
- Cache based on PDF file MD5 hash with validity validation (modification time check)

### [0.4.0] - 2024-12-30

**Refactor: Simplified Architecture, Merged PDF Processing into main.py**
- Merged DumpPDF's PDF-to-image logic into main.py
- Removed `utilities/DumpPDF.py`, all PDF processing now in main.py's `convert_pdf_to_images` function

### [0.3.0] - 2024-12-30

**Refactor: Simplified Architecture, Merged All Logic into Workers**
- Merged DeepSeekOCR, GenericVL, STT, and Summarization logic directly into their respective Workers
- Removed redundant intermediate classes, simplified code structure

### [0.2.0] - 2024-12-30

**Refactor: Worker Architecture and Rate Limiting**
- Added Worker architecture with all AI services processed through independent daemon threads
- Added token bucket algorithm-based RPM/TPM rate limiting
- Independent API configuration (URL, Key, Model) for each service
- Implemented graceful shutdown ensuring queued tasks complete before exit

### [0.1.0] - 2024-12-21

**Initial Release**
- PDF to OCR with summarized note generation
- Audio to STT with summarized note generation
- OpenAI Like API support
