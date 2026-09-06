import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_fact_submission, generate_history_geo_subset

def main():
    facts = generate_history_geo_subset(21)
    tok_qwen = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok_qwen.padding_side = 'left'
    if tok_qwen.pad_token is None:
        tok_qwen.pad_token = tok_qwen.eos_token

    tok_tiny = AutoTokenizer.from_pretrained('modal/text_models/tinyllama-1.1b')
    tok_tiny.padding_side = 'left'
    if tok_tiny.pad_token is None:
        tok_tiny.pad_token = tok_tiny.eos_token

    models = [
        ('Parent 2: Qwen2.5-0.5B', 'modal/text_models/qwen2.5-0.5b', tok_qwen),
        ('Parent 3: TinyLlama-1.1B', 'modal/text_models/tinyllama-1.1b', tok_tiny),
        ('Tri-Parent LoRA (my_llm_folder)', 'my_llm_folder', tok_qwen),
    ]

    results = {}
    for name, path, tokenizer in models:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda')
        prompts = [f['prompt'] for f in facts]
        enc = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=36, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        results[name] = []
        for f, gen in zip(facts, decoded):
            passed, rat = eval_fact_submission(gen, f['answer'])
            country = f['prompt'].split('of ')[1].split('?')[0]
            results[name].append((country, f['answer'], passed, gen.strip()))
        del m
        torch.cuda.empty_cache()

    print(f"\n{'Country':<16} | {'Target':<14} | {'Qwen2.5-0.5B':<14} | {'TinyLlama-1.1B':<14} | {'Tri-Parent LoRA':<14}")
    print("-" * 80)
    for i in range(len(facts)):
        c = facts[i]['prompt'].split('of ')[1].split('?')[0]
        tgt = facts[i]['answer']
        q_pass = "PASS" if results['Parent 2: Qwen2.5-0.5B'][i][2] else "FAIL"
        t_pass = "PASS" if results['Parent 3: TinyLlama-1.1B'][i][2] else "FAIL"
        tri_pass = "PASS" if results['Tri-Parent LoRA (my_llm_folder)'][i][2] else "FAIL"
        print(f"{c:<16} | {tgt:<14} | {q_pass:<14} | {t_pass:<14} | {tri_pass:<14}")
        if q_pass != tri_pass or t_pass != tri_pass:
            print(f"   [DIFF] Target={tgt}")
            print(f"      Qwen:     {repr(results['Parent 2: Qwen2.5-0.5B'][i][3][:60])}")
            print(f"      Tiny:     {repr(results['Parent 3: TinyLlama-1.1B'][i][3][:60])}")
            print(f"      TriLoRA:  {repr(results['Tri-Parent LoRA (my_llm_folder)'][i][3][:60])}")

if __name__ == "__main__":
    main()
