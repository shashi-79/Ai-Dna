import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import (
    eval_code_submission,
    eval_math_submission,
    eval_fact_submission,
    load_gsm8k_math_subset,
    generate_coding_subset,
    generate_science_subset,
    generate_history_geo_subset,
    generate_language_logic_subset
)

def main():
    model_path = "modal/fused_homogeneous_smollm2"
    tok_path = "modal/text_models/smollm2-360m"
    
    print("=" * 80)
    print("  MANUAL AUDIT: METHOD 4 HOMOGENEOUS LINEAGE (SMOLLM2 135M + 360M)")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    
    # Collect 30 questions across all 5 domains
    math_qs = load_gsm8k_math_subset(5)
    code_qs = generate_coding_subset(10)
    science_qs = generate_science_subset(5)
    hist_qs = generate_history_geo_subset(5)
    logic_qs = generate_language_logic_subset(5)
    
    test_suite = [
        ("Coding", code_qs),
        ("Math", math_qs),
        ("Science", science_qs),
        ("History/Geo", hist_qs),
        ("Language/Logic", logic_qs)
    ]
    
    audit_records = []
    q_num = 1
    
    for cat_name, qs in test_suite:
        print(f"\n>>> Running Category: {cat_name} ({len(qs)} questions)")
        prompts = [q["prompt"] for q in qs]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)
        decoded_clean = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for idx, (item, raw_str, clean_str) in enumerate(zip(qs, decoded, decoded_clean)):
            target = item["answer"]
            q_type = item.get("type", "fact")
            
            if q_type == "code":
                tpl = item.get("template_name", "square")
                passed, rationale = eval_code_submission(clean_str, tpl)
            elif q_type == "math":
                passed, rationale = eval_math_submission(clean_str, target)
            else:
                passed, rationale = eval_fact_submission(clean_str, target)
                
            record = {
                "question_num": q_num,
                "category": cat_name,
                "prompt": item["prompt"],
                "target": target,
                "raw_output_with_special": raw_str,
                "clean_output": clean_str,
                "passed": passed,
                "rationale": rationale,
                "token_ids": gen_tokens[idx].cpu().tolist()[:10]
            }
            audit_records.append(record)
            q_num += 1
            
    # Save full audit JSON
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/homogeneous_lineage_30q_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2)
        
    print(f"\nAudit complete. Saved to {out_file}")
    
    # Print formatted manual audit log
    print("\n" + "=" * 95)
    print(f"{'#':<3} | {'Category':<14} | {'Target':<12} | {'Verdict':<7} | {'Raw Output / Rationale'}")
    print("-" * 95)
    for r in audit_records:
        v_str = "PASS" if r["passed"] else "FAIL"
        clean_rep = repr(r["clean_output"][:40]) if r["clean_output"].strip() else repr(r["raw_output_with_special"])
        print(f"{r['question_num']:<3} | {r['category']:<14} | {r['target']:<12} | {v_str:<7} | {clean_rep}")
        print(f"    Prompt: {repr(r['prompt'][:70])}")
        print(f"    Rationale: {r['rationale']}")
        print(f"    First 5 Token IDs: {r['token_ids'][:5]}")
        print("-" * 95)

if __name__ == "__main__":
    main()
