import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_fact_submission

def main():
    tok = AutoTokenizer.from_pretrained('modal/text_models/smollm2-360m')
    m = AutoModelForCausalLM.from_pretrained('modal/text_models/smollm2-360m', torch_dtype=torch.bfloat16).to('cuda')
    
    questions = [
        ("What is the atomic number of Hydrogen?", "1"),
        ("What is the atomic number of Helium?", "2"),
        ("What is the atomic number of Carbon?", "6"),
        ("What is the atomic number of Oxygen?", "8"),
        ("What is the atomic number of Gold?", "79"),
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the capital of Australia?", "Canberra"),
    ]

    print("--- 1. Raw Prompt (Current Benchmark setting) ---")
    raw_prompts = [f"{q}\nAnswer:" for q, _ in questions]
    enc_raw = tok(raw_prompts, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        out_raw = m.generate(**enc_raw, max_new_tokens=36, do_sample=False, pad_token_id=tok.eos_token_id)
    dec_raw = tok.batch_decode(out_raw[:, enc_raw['input_ids'].shape[1]:], skip_special_tokens=False)
    for (q, a), gen in zip(questions, dec_raw):
        p, r = eval_fact_submission(gen, a)
        print(f"  {q:<36} | Gen: {repr(gen[:30]):<32} | {'PASS' if p else 'FAIL'}")

    print("\n--- 2. ChatML Prompt (Native SmolLM2 format) ---")
    chat_prompts = [
        f"<|im_start|>user\n{q} Give only the direct short answer without explanation.<|im_end|>\n<|im_start|>assistant\n"
        for q, _ in questions
    ]
    enc_chat = tok(chat_prompts, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        out_chat = m.generate(**enc_chat, max_new_tokens=36, do_sample=False, pad_token_id=tok.eos_token_id)
    dec_chat = tok.batch_decode(out_chat[:, enc_chat['input_ids'].shape[1]:], skip_special_tokens=True)
    passed_chat = 0
    for (q, a), gen in zip(questions, dec_chat):
        p, r = eval_fact_submission(gen, a)
        if p:
            passed_chat += 1
        print(f"  {q:<36} | Gen: {repr(gen.strip()[:30]):<32} | {'PASS' if p else 'FAIL'}")
    print(f"\nChatML Score: {passed_chat}/{len(questions)} ({passed_chat*10}%)")

if __name__ == "__main__":
    main()
