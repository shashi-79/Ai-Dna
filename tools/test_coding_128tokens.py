import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import (
    generate_coding_subset,
    eval_code_submission,
)

def main():
    tok = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained('my_llm_folder', torch_dtype=torch.bfloat16).to('cuda')
    q_list = generate_coding_subset(10)
    prompts = [q['prompt'] for q in q_list]
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')

    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    decoded = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)

    print("\n--- Detailed Audit at 128 tokens ---")
    passed = 0
    for q, dec in zip(q_list, decoded):
        p, r = eval_code_submission(dec, q['template_name'])
        if p:
            passed += 1
        print(f"  {q['template_name']:<15}: {'PASS' if p else 'FAIL':<6} | {r}")
        if not p:
            print(f"     RAW (first 120 chars): {repr(dec.strip()[:120])}")
    print(f"\nScore at 128 tokens: {passed}/10 ({passed*10}%)")

if __name__ == "__main__":
    main()
