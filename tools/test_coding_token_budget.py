import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_code_submission, generate_coding_subset

def main():
    templates = generate_coding_subset(10)
    tok = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained('my_llm_folder', torch_dtype=torch.bfloat16).to('cuda')
    prompts = [t['prompt'] for t in templates]
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')

    print("--- Testing max_new_tokens = 36 (Benchmark setting) ---")
    with torch.no_grad():
        out36 = m.generate(**enc, max_new_tokens=36, do_sample=False, pad_token_id=tok.eos_token_id)
    dec36 = tok.batch_decode(out36[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    p36 = sum(1 for t, g in zip(templates, dec36) if eval_code_submission(g, t['template_name'])[0])
    print(f"Passed at 36 tokens: {p36}/10 ({p36*10}%)")

    print("\n--- Testing max_new_tokens = 96 (Sufficient for docstring + body) ---")
    with torch.no_grad():
        out96 = m.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
    dec96 = tok.batch_decode(out96[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    for t, g in zip(templates, dec96):
        p, r = eval_code_submission(g, t['template_name'])
        status = "PASS" if p else "FAIL"
        print(f"  {t['template_name']:<15}: {status:<6} | {r}")
    p96 = sum(1 for t, g in zip(templates, dec96) if eval_code_submission(g, t['template_name'])[0])
    print(f"\nPassed at 96 tokens: {p96}/10 ({p96*10}%)")

if __name__ == "__main__":
    main()
