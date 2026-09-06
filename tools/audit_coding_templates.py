import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import eval_code_submission, generate_coding_subset

def main():
    templates = generate_coding_subset(10)
    tok_qwen = AutoTokenizer.from_pretrained('modal/text_models/qwen2.5-0.5b')
    tok_qwen.padding_side = 'left'
    if tok_qwen.pad_token is None:
        tok_qwen.pad_token = tok_qwen.eos_token

    tok_smol = AutoTokenizer.from_pretrained('modal/text_models/smollm2-360m')
    tok_smol.padding_side = 'left'
    if tok_smol.pad_token is None:
        tok_smol.pad_token = tok_smol.eos_token

    models = [
        ('Parent 1: SmolLM2-360M', 'modal/text_models/smollm2-360m', tok_smol),
        ('Parent 2: Qwen2.5-0.5B', 'modal/text_models/qwen2.5-0.5b', tok_qwen),
        ('Tri-Parent LoRA (my_llm_folder)', 'my_llm_folder', tok_qwen),
    ]

    results = {}
    for name, path, tokenizer in models:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda')
        prompts = [t['prompt'] for t in templates]
        enc = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        results[name] = []
        for t, gen in zip(templates, decoded):
            passed, rat = eval_code_submission(gen, t['template_name'])
            results[name].append((t['template_name'], passed, rat, gen.strip()))
        del m
        torch.cuda.empty_cache()

    print(f"\n{'Template':<16} | {'SmolLM2-360M':<14} | {'Qwen2.5-0.5B':<14} | {'Tri-Parent LoRA':<14}")
    print("-" * 65)
    for i in range(len(templates)):
        tpl = templates[i]['template_name']
        s_pass = "PASS" if results['Parent 1: SmolLM2-360M'][i][1] else "FAIL"
        q_pass = "PASS" if results['Parent 2: Qwen2.5-0.5B'][i][1] else "FAIL"
        tri_pass = "PASS" if results['Tri-Parent LoRA (my_llm_folder)'][i][1] else "FAIL"
        print(f"{tpl:<16} | {s_pass:<14} | {q_pass:<14} | {tri_pass:<14}")
        if s_pass != tri_pass or q_pass != tri_pass:
            print(f"   [DIFF] Template={tpl}")
            print(f"      Smol:     {repr(results['Parent 1: SmolLM2-360M'][i][3][:60])}")
            print(f"      Qwen:     {repr(results['Parent 2: Qwen2.5-0.5B'][i][3][:60])}")
            print(f"      TriLoRA:  {repr(results['Tri-Parent LoRA (my_llm_folder)'][i][3][:60])}")

if __name__ == "__main__":
    main()
