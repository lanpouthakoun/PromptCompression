import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional
import os
import tiktoken


def load_models(
    model_name: str,
    frozen_model_name: str
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoTokenizer]:

    print(f"Loading trainable model: {model_name}")
   
    tokenizer = AutoTokenizer.from_pretrained(
       model_name
    )
    frozen_tokenizer = AutoTokenizer.from_pretrained(
        frozen_model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if frozen_tokenizer.pad_token is None:
        frozen_tokenizer.pad_token = frozen_tokenizer.eos_token
        frozen_tokenizer.pad_token_id = frozen_tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    
    print(f"Loading frozen model: {frozen_model_name}")
    
    frozen_model = AutoModelForCausalLM.from_pretrained(
         frozen_model_name
    )
    
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False
    
    return model, frozen_model, tokenizer, frozen_tokenizer





