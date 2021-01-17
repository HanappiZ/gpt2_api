from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import argparse
import logging

import numpy as np
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)
tokenizer = tokenizer_class.from_pretrained('gpt2')
model = model_class.from_pretrained('gpt2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationArgs(BaseModel):
    prompt: str
    temperature: float = 1.0
    k: int = 0
    p: float = 0.9
    repetition_penalty: float = 10.0
    min_length: int = 50
    max_length: int = 100
    stop_token: Optional[str] = None


@app.get("/")
async def root():
    return { 'message' : 'Welcome to GPT-2 text generation API'}


@app.post("/generate")
async def generate(args: GenerationArgs):
    prompt = args.prompt
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        temperature=args.temperature,
        min_length=args.min_length,
        max_length=args.max_length,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        output_scores=True,
        top_k=args.k,
        top_p=args.p
    )

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences.sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )
        generated_sequences.append(total_sequence)

    return { 'result': generated_sequences}
