import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_code_submission

def main():
    templates = [
        ("square", "def square(x):\n", "Write a python function to compute the square of x:"),
        ("is_even", "def is_even(x):\n", "Write a python function to check if a number x is even:"),
        ("get_length", "def get_length(lst):\n", "Write a python function to calculate the length of list lst:"),
        ("reverse_string", "def reverse_string(s):\n", "Write a python function to reverse a string s:"),
        ("find_max", "def find_max(a, b):\n", "Write a python function to find maximum of a and b:"),
        ("double_nums", "def double_nums(nums):\n", "Write a python function to double all numbers in list nums:"),
        ("first_elem", "def first_elem(lst):\n", "Write a python function to return the first element of lst:"),
        ("is_positive", "def is_positive(x):\n", "Write a python function to check if x is positive:"),
        ("cube", "def cube(n):\n", "Write a python function to calculate cube of n:"),
        ("concat", "def concat(a, b):\n", "Write a python function to join strings a and b:"),
    ]

    tok = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained('my_llm_folder', torch_dtype=torch.bfloat16).to('cuda')
    
    # We supply def header in prompt, and append generation
    prompts = [f"{desc}\n```python\n{hdr}" for name, hdr, desc in templates]
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')

    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
    decoded = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

    passed = 0
    print("\n--- Testing with Function Header in Prompt @ 160 tokens ---")
    for (name, hdr, _), gen in zip(templates, decoded):
        full_code = hdr + gen
        p, r = eval_code_submission(full_code, name)
        if p:
            passed += 1
        print(f"  {name:<15}: {'PASS' if p else 'FAIL':<6} | {r}")
        if not p:
            print(f"     RAW (first 100 chars): {repr(full_code.strip()[:100])}")
    print(f"\nFinal Score: {passed}/{len(templates)} ({passed*10}%)")

if __name__ == "__main__":
    main()
