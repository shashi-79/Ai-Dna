import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import (
    generate_coding_subset,
    eval_code_submission,
)

def test_coding_batch(model_path, num_q=100, max_tokens=96):
    tok = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
    q_list = generate_coding_subset(num_q)
    
    passed = 0
    batch_size = 64
    for b_start in range(0, len(q_list), batch_size):
        b_end = min(b_start + batch_size, len(q_list))
        batch = q_list[b_start:b_end]
        prompts = [q['prompt'] for q in batch]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        decoded = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for q, dec in zip(batch, decoded):
            p, r = eval_code_submission(dec, q['template_name'])
            if p:
                passed += 1

    print(f"Results for {model_path} ({num_q} Qs @ max_tokens={max_tokens}): {passed}/{num_q} ({passed/num_q*100:.1f}%)")

if __name__ == "__main__":
    test_coding_batch('my_llm_folder', num_q=100, max_tokens=96)
