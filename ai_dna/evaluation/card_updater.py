"""
Model Card & Leaderboard Markdown Table Generator for AI-DNA.
Formats benchmark scores into readable GitHub markdown tables and injects
the official Hugging Face `model-index:` metadata schema into README.md YAML frontmatter.
"""

import os
import re
from typing import Dict, Any, List, Optional


def format_benchmark_markdown_table(summary: Dict[str, Any]) -> str:
    """Generates a clean GitHub Flavored Markdown table of benchmark results."""
    lines = [
        "| Benchmark Task | Questions Evaluated | Accuracy (%) | Performance |",
        "| :--- | :--- | :--- | :--- |",
    ]
    tasks = summary.get("tasks", {})
    for task_key, info in tasks.items():
        name = info.get("name", task_key.upper())
        total = info.get("total", 0)
        acc = info.get("accuracy", 0.0)
        qps = info.get("qps")
        perf_str = f"{qps:.1f} q/s" if qps else "Standard"
        lines.append(f"| **{name}** | {total} | **{acc:.1f}%** | {perf_str} |")

    avg = summary.get("summary", {}).get("average_accuracy", summary.get("overall_accuracy", 0.0))
    lines.append(f"| **Overall Average** | - | **{avg:.1f}%** | - |")
    return "\n".join(lines)


def update_readme_model_index(
    readme_path: str,
    model_name: str,
    results_summary: Dict[str, Any],
) -> bool:
    """
    Injects the official Hugging Face `model-index:` metadata schema
    into the YAML frontmatter of README.md so scores appear on the Hub page widget.
    """
    if not os.path.exists(readme_path):
        return False

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    task_results = []
    tasks = results_summary.get("tasks", {})
    for task_key, task_info in tasks.items():
        name = task_info.get("name", task_key.upper())
        acc = task_info.get("accuracy", 0.0)
        task_results.append({
            "task": {"type": "text-generation", "name": "Text Generation"},
            "dataset": {"name": name, "type": task_key.lower()},
            "metrics": [{"name": "Accuracy", "type": "accuracy", "value": acc}],
        })

    model_index_yaml = "model-index:\n"
    model_index_yaml += f"- name: {model_name}\n"
    model_index_yaml += "  results:\n"
    for tr in task_results:
        model_index_yaml += f"  - task:\n"
        model_index_yaml += f"      type: {tr['task']['type']}\n"
        model_index_yaml += f"      name: {tr['task']['name']}\n"
        model_index_yaml += f"    dataset:\n"
        model_index_yaml += f"      name: {tr['dataset']['name']}\n"
        model_index_yaml += f"      type: {tr['dataset']['type']}\n"
        model_index_yaml += f"    metrics:\n"
        for m in tr["metrics"]:
            model_index_yaml += f"    - name: {m['name']}\n"
            model_index_yaml += f"      type: {m['type']}\n"
            model_index_yaml += f"      value: {m['value']}\n"

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            existing_frontmatter = parts[1]
            clean_frontmatter = re.sub(r"model-index:.*?(?=\n[a-zA-Z_-]+:|\Z)", "", existing_frontmatter, flags=re.DOTALL)
            new_frontmatter = clean_frontmatter.strip() + "\n" + model_index_yaml
            new_content = f"---\n{new_frontmatter}\n---" + parts[2]
        else:
            new_content = f"---\n{model_index_yaml}---\n\n" + content
    else:
        new_content = f"---\n{model_index_yaml}---\n\n" + content

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return True
