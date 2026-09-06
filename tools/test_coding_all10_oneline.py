import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_code_submission

def main():
    templates = [
        ("Write a python function `square(x)` to compute square of x (one line):", "square"),
        ("Write a python function `is_even(x)` to check if x is even (one line):", "is_even"),
        ("Write a python function `get_length(lst)` to calculate length of lst (one line):", "get_length"),
        ("Write a python function `reverse_string(s)` to reverse string s (one line):", "reverse_string"),
        ("Write a python function `find_max(a, b)` to find maximum of a and b (one line):", "find_max"),
        ("Write a python function `double_nums(nums)` to double all numbers in nums (one line):", "double_nums"),
        ("Write a python function `first_elem(lst)` to return first element of lst (one line):", "first_elem"),
        ("Write a python function `is_positive(x)` to check if x is positive (one line):", "is_positive"),
        ("Write a python function `cube(n)` to calculate cube of n (one line):", "cube"),
        ("Write a python function `concat(a, b)` to join strings a and b (one line):", "concat"),
    ]

    tok = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained('my_llm_folder', torch_dtype=torch.bfloat16).to('cuda')
    prompts = [f"{desc}\n```python\n" for desc, name in templates]
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')

    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
    decoded = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

    print("\n=== One-Line Coding Evaluation (my_llm_folder @ 48 tokens) ===")
    passed = 0
    for (_, name), gen in zip(templates, decoded):
        p, r = eval_code_submission(gen, name)
        if p:
            passed += 1
        print(f"  {name:<15}: {'PASS' if p else 'FAIL':<6} | {repr(gen.strip()[:60])} | {r}")
    print(f"\nTotal Score: {passed}/{len(templates)} ({passed*10}%)")

if __name__ == "__main__":
    main()
