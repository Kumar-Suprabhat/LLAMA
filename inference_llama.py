import torch
from fastapi import FastAPI, HTTPException, Body, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder, whoami
from transformers import pipeline
import uvicorn
import time
import transformers
import io
import sys
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import File, UploadFile
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.responses import FileResponse
from llama_cpp import Llama
import os
from pydantic import BaseModel
from fastapi import Body

# API endpoint for text generation
from fastapi.responses import JSONResponse

# Save the Hugging Face token and confirm login
hf_token = '<your access code>'
HfFolder.save_token(hf_token)
user = whoami()
print(f"Logged in as {user['name']}")

app = FastAPI()
cache_dir='llama-3b/models--meta-llama--Meta-Llama-3-8B-Instruct'
# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    ).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

#model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", quantization_config=quantization_config)
model_device=next(model.parameters()).device
memory = model.get_memory_footprint()
memory = memory / (1024 ** 3)
print(f"Model loaded successfully on {device} and it is consuming {memory} GB.")
text_generation_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
    return_full_text=False
)
text_generation_pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

DEFAULT_SYSTEM_PROMPT = """\
You are a precise and reliable AI assistant for Technical query analysis. Your task is to provide the precise values for the attributes such as criticality score(an integer), topics(string with singular/multiple tokens) and complaince(string with singular/multiple tokens) based on the context. Keep your answer short and include only the relevant parts. Strive for accuracy, clarity in your answers, and only report the values if it is present in the context."""

PROMPT_FOR_GENERATION_FORMAT="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""



# Function to split text into chunks using token counts with 10-token overlap
def chunk_text_with_tokenizer(text, model_name="meta-llama/Meta-Llama-3-8B-Instruct", chunk_size=1024, chunk_overlap=0):
    """
    Splits text into chunks based on token counts using the specified model's tokenizer.

    Args:
        text: The input text to split.
        model_name: The model name to use for tokenization.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: The number of overlapping tokens between chunks (10 tokens overlap by default).

    Returns:
        List of text chunks.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=False).squeeze()
    
    # Check if the number of tokens is less than or equal to chunk_size
    if len(tokens) <= chunk_size:
        # If the text is small enough, return it as a single chunk
        return [tokenizer.decode(tokens, skip_special_tokens=True)]
    
    # Split the tokens into chunks
    chunks = []
    start = 0
    while start + chunk_overlap < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        curr_context=tokenizer.decode(chunk, skip_special_tokens=True)
        #print(curr_context)
        chunks.append(curr_context)
        start = end - chunk_overlap  # Adjust start for overlap (10 tokens overlap)

    return chunks

class Llama3Request(BaseModel):
    system_prompt: str | None = None  # Optional field
    context: str  # Required field
    prompt: str  # Required field
    
class BadRequestException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

class ForcedException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors



# Utility functions for generating messages and statuses
def get_success_msg(prefix, status_code):
    if status_code == 202:
        _message = "{} ACCEPTED".format(prefix)
    elif status_code == 201:
        _message = "{} CREATED".format(prefix)
    else:
        _message = "{} DONE".format(prefix)
    return _message, "success", status_code

def get_bad_request_msg(prefix, error, status_code=400):
    return "{} BAD REQUEST | {}".format(prefix, error), "bad_request", status_code

def get_forced_error_msg(prefix, error, status_code):
    return "{} FAILED | Error = {}".format(prefix, error), "failed", status_code

def get_internal_server_error_msg(prefix, error, status_code=500):
    return "{} INTERNAL SERVER ERROR | {}".format(prefix, error), "internal_error", status_code

# API endpoint for text generation
@app.post("/api/llama3/query")
async def llama3_endpoint(request: Llama3Request):
    message = "LLAMA3-Query POST: DONE"
    try:
        system_prompt = request.system_prompt
        context = request.context
        prompt = request.prompt

        # Validate input: context and prompt
        # if 'topic' in prompt:
        #     raise ForcedException("Bad request successfully rasied")
        if not context:
            raise BadRequestException("No context provided. Please enter a valid context.")
        if not prompt:
            raise BadRequestException("No prompt provided. Please enter a valid prompt.")

        print(context)
        print(prompt)
        print(system_prompt)
        start_time = time.time()

        final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

        # Split the context into chunks
        context_chunks = chunk_text_with_tokenizer(context, chunk_size=2000, chunk_overlap=50)
        print(f"context_chunks: {context_chunks}")
        print(f"Number of chunks: {len(context_chunks)}")

        complete_response = []
        for chunk in context_chunks:
            instruction = f"Context: '''{chunk}'''\nPrompt: '{prompt}'"
            full_prompt = PROMPT_FOR_GENERATION_FORMAT.format(system_prompt=final_system_prompt, instruction=instruction)
            outputs = text_generation_pipeline([full_prompt], max_new_tokens=512)

            print(f"full prompt: {full_prompt}")
            print(f"Type of outputs: {type(outputs)}")
            print(outputs)

            outputs = outputs[0][0]['generated_text']
            outputs = outputs.replace("\n\n", "")
            complete_response.append(outputs.strip())

        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time} seconds")
        print(complete_response)# TODO MOVE above
        _message, _status, status_code = get_success_msg(message, 200)
        return JSONResponse(content={
            "message": _message,
            "status": _status,
            "output": {
                "generated_response": complete_response,
                "chunks_count": len(context_chunks),
                "processing_time_seconds": elapsed_time
            }
        }, status_code=status_code)

    except BadRequestException as err:
        _message, _status, status_code = get_bad_request_msg(message, str(err), 400)
        return JSONResponse(content={"message": _message, "status": _status}, status_code=status_code)
    except ForcedException as err:
        _message, _status, status_code = get_forced_error_msg(message, str(err), 500)
        return JSONResponse(content={"message": _message, "status": _status}, status_code=status_code)
    except Exception as err:
        _message, _status, status_code = get_internal_server_error_msg(message, str(err), 500)
        return JSONResponse(content={"message": _message, "status": _status}, status_code=status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
