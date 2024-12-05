# Llama3 API

## Overview

Llama3 API provides an interface to interact with the Meta-Llama-3-8B-Instruct model for generating text-based responses from a given context and prompt. The API is built using FastAPI and allows context splitting for efficient model processing.

## Features

- **Text Generation**: Generate responses based on user input context and prompt.
- **Context Splitting**: Large context text is split into manageable chunks.
- **Custom System Prompt**: Optionally provide a system prompt to guide model behavior.

## Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install fastapi uvicorn transformers torch huggingface_hub langchain PyPDF2 llama_cpp

## Postman
{
  "system_prompt": "Optional custom system prompt",
  "context": "The context to analyze.",
  "prompt": "Specific instruction or question."
}


