import json

def main():
    report = json.load(open('outputs/all_methods_high_vram_report.json', encoding='utf-8'))
    for m in report:
        print("=" * 85)
        print(f"MODEL: {m['model_label']} | OVERALL: {m['total_accuracy_pct']}% ({m['total_passed']}/{m['total_evaluated']})")
        print("=" * 85)
        for cat, stats in m['categories'].items():
            print(f"  - {cat:<15}: {stats['passed']:>5}/{stats['total']:<5} ({stats['accuracy_pct']:>5.1f}%)")
        print("  Sample Audits:")
        for s in m.get('sample_audits', []):
            prompt = s['prompt'].replace('\n', ' ')[:45]
            out = repr(s['raw_output'][:55])
            p_str = "PASS" if s['passed'] else "FAIL"
            print(f"    [{p_str}] {s['category']:<12} | Prompt: {prompt:<45} | Exp: {s['expected']:<10} | Out: {out}")

if __name__ == "__main__":
    main()
