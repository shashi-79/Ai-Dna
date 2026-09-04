"""
Unified Benchmark Task Definitions, Loaders, and Scorers for AI-DNA.
Supports MMLU, GSM8K, ARC-Challenge, and IFEval with:
1. Streaming / online datasets via Hugging Face `datasets`
2. High-quality offline fallback questions when network or datasets library is unavailable
3. Robust regex-based and constraint-based answer extractors
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Callable

# =====================================================================
# Offline Fallback Question Bank (Used when offline or for quick tests)
# =====================================================================
FALLBACK_QUESTIONS = {
    "mmlu": [
        {
            "prompt": "Question: Which of the following elements has the highest electronegativity?\nA) Sodium\nB) Chlorine\nC) Fluorine\nD) Oxygen\nAnswer (one letter only):",
            "answer": "C",
            "task": "mmlu",
            "subject": "chemistry",
        },
        {
            "prompt": "Question: In group theory, what is the order of the identity element in any group?\nA) 0\nB) 1\nC) Infinity\nD) Undefined\nAnswer (one letter only):",
            "answer": "B",
            "task": "mmlu",
            "subject": "mathematics",
        },
        {
            "prompt": "Question: What primary function does ATP synthase perform in cellular respiration?\nA) Hydrolyzing glucose\nB) Phosphorylating ADP to ATP\nC) Pumping protons out of mitochondria\nD) Oxidizing NADH\nAnswer (one letter only):",
            "answer": "B",
            "task": "mmlu",
            "subject": "biology",
        },
        {
            "prompt": "Question: Which law states that the total entropy of an isolated system always increases over time?\nA) First Law of Thermodynamics\nB) Second Law of Thermodynamics\nC) Third Law of Thermodynamics\nD) Zeroth Law of Thermodynamics\nAnswer (one letter only):",
            "answer": "B",
            "task": "mmlu",
            "subject": "physics",
        },
        {
            "prompt": "Question: In computer architecture, what does ALU stand for?\nA) Arithmetic Logic Unit\nB) Asynchronous Linear Utility\nC) Algorithmic Logic Utility\nD) Array Linear Unit\nAnswer (one letter only):",
            "answer": "A",
            "task": "mmlu",
            "subject": "computer_science",
        },
    ],
    "gsm8k": [
        {
            "prompt": "Problem: A bakery sells boxes of donuts for $12 each. If a customer buys 4 boxes and pays with a $100 bill, how much change should they receive?\nSolve step by step. Final Answer:",
            "expected": "52",
            "task": "gsm8k",
        },
        {
            "prompt": "Problem: A train travels at a constant speed of 60 miles per hour for 2.5 hours. How many miles does it travel?\nSolve step by step. Final Answer:",
            "expected": "150",
            "task": "gsm8k",
        },
        {
            "prompt": "Problem: If a rectangle has length 15 cm and width 8 cm, what is its perimeter in cm?\nSolve step by step. Final Answer:",
            "expected": "46",
            "task": "gsm8k",
        },
        {
            "prompt": "Problem: A store offers a 20% discount on an item originally priced at $80. What is the discounted price?\nSolve step by step. Final Answer:",
            "expected": "64",
            "task": "gsm8k",
        },
        {
            "prompt": "Problem: If 3x + 9 = 30, what is the value of x?\nSolve step by step. Final Answer:",
            "expected": "7",
            "task": "gsm8k",
        },
    ],
    "arc": [
        {
            "prompt": "Question: Which property is common to all metals at room temperature?\nA) High electrical conductivity\nB) Liquid state\nC) Low density\nD) Non-magnetic\nAnswer (one letter only):",
            "answer": "A",
            "task": "arc",
        },
        {
            "prompt": "Question: Which process is directly responsible for cloud formation in the atmosphere?\nA) Precipitation\nB) Condensation of water vapor\nC) Sublimation of ice\nD) Evaporation of groundwater\nAnswer (one letter only):",
            "answer": "B",
            "task": "arc",
        },
        {
            "prompt": "Question: What happens when an acid reacts with a base in an aqueous solution?\nA) Formation of salt and water\nB) Decomposition of base into hydrogen gas\nC) Deposition of pure oxygen\nD) Rapid crystallization of nitric acid\nAnswer (one letter only):",
            "answer": "A",
            "task": "arc",
        },
        {
            "prompt": "Question: Which of the following wavelengths has the highest frequency in the electromagnetic spectrum?\nA) Radio waves\nB) Infrared\nC) Visible light\nD) Gamma rays\nAnswer (one letter only):",
            "answer": "D",
            "task": "arc",
        },
    ],
    "ifeval": [
        {
            "prompt": "Write a three-item grocery shopping list using bullet points starting with - . Do not include any other text.",
            "instructions": ["bullet", "list"],
            "task": "ifeval",
        },
        {
            "prompt": "Provide a valid JSON object with keys 'name' and 'role' describing a scientist.",
            "instructions": ["json"],
            "task": "ifeval",
        },
        {
            "prompt": "Write the phrase HELLO WORLD completely in uppercase letters.",
            "instructions": ["uppercase"],
            "task": "ifeval",
        },
    ],
}

MMLU_SUBJECTS_SAMPLE = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics",
    "elementary_mathematics", "formal_logic", "global_facts",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "machine_learning",
]


# =====================================================================
# Answer Scoring Functions
# =====================================================================
_NUM_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def check_mc(response: str, expected: str) -> bool:
    """Evaluates multiple-choice response against expected letter A/B/C/D."""
    exp = _NUM_MAP.get(expected.upper(), expected.upper())
    r = response.strip().upper()
    for ch in r:
        if ch in "ABCD":
            return ch == exp
        if ch in "1234":
            return _NUM_MAP.get(ch, ch) == exp
        if ch.isalnum():
            break
    m = re.search(r"(?:answer|ans)[:\s]+([ABCD1-4])", r)
    if m:
        return _NUM_MAP.get(m.group(1), m.group(1)) == exp
    return False


def check_gsm8k(response: str, expected: str) -> bool:
    """Evaluates mathematical response against target numerical answer."""
    if not expected:
        return False
    exp_clean = expected.replace(",", "").strip()
    nums = re.findall(r"-?[\d]+\.?\d*", response.replace(",", ""))
    return any(n.strip() == exp_clean for n in nums)


def check_ifeval(response: str, instructions: List[str]) -> bool:
    """Evaluates instruction compliance constraints."""
    if not instructions:
        return len(response.strip()) > 5
    passed = 0
    for instr in instructions:
        low = instr.lower()
        if "uppercase" in low:
            passed += int(response == response.upper() and bool(response.strip()))
        elif "lowercase" in low:
            passed += int(response == response.lower() and bool(response.strip()))
        elif "bullet" in low or "list" in low:
            passed += int(any(l.lstrip().startswith(("- ", "* ", "• ", "1.")) for l in response.splitlines()))
        elif "json" in low:
            try:
                json.loads(response)
                passed += 1
            except Exception:
                pass
        else:
            passed += int(len(response.strip()) > 5)
    return passed > 0


# =====================================================================
# Dataset Loaders with Fallback
# =====================================================================
def _is_offline() -> bool:
    return any(
        os.environ.get(k) in ["1", "true", "True"]
        for k in ["HF_HUB_OFFLINE", "AI_DNA_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    )


def load_mmlu(limit: Optional[int] = None, subjects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Loads MMLU questions, falling back to built-in questions if offline."""
    if _is_offline():
        fallback = FALLBACK_QUESTIONS["mmlu"]
        return fallback[:limit] if limit else fallback
    try:
        from datasets import load_dataset
        target_subjs = subjects or MMLU_SUBJECTS_SAMPLE
        all_q = []
        labels = ["A", "B", "C", "D"]
        for subj in target_subjs:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                for item in ds:
                    choices = "\n".join(f"{labels[i]}) {item['choices'][i]}" for i in range(min(4, len(item["choices"]))))
                    all_q.append({
                        "prompt": f"Question: {item['question']}\n{choices}\nAnswer (one letter only):",
                        "answer": labels[int(item["answer"])],
                        "subject": subj,
                        "task": "mmlu",
                    })
                    if limit and len(all_q) >= limit:
                        return all_q
            except Exception:
                continue
        if all_q:
            return all_q
    except Exception:
        pass
    fallback = FALLBACK_QUESTIONS["mmlu"]
    return fallback[:limit] if limit else fallback


def load_gsm8k(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads GSM8K math questions, falling back to built-in questions if offline."""
    if _is_offline():
        fallback = FALLBACK_QUESTIONS["gsm8k"]
        return fallback[:limit] if limit else fallback
    try:
        from datasets import load_dataset
        for repo in ["openai/gsm8k", "gsm8k"]:
            try:
                ds = load_dataset(repo, "main", split="test", trust_remote_code=True)
                qs = []
                for item in ds:
                    m = re.search(r"####\s*([\d,\.\-]+)", item.get("answer", ""))
                    qs.append({
                        "prompt": f"Problem: {item['question']}\nSolve step by step. Final Answer:",
                        "expected": m.group(1).replace(",", "").strip() if m else "",
                        "task": "gsm8k",
                    })
                    if limit and len(qs) >= limit:
                        return qs
                if qs:
                    return qs
            except Exception:
                continue
    except Exception:
        pass
    fallback = FALLBACK_QUESTIONS["gsm8k"]
    return fallback[:limit] if limit else fallback


def load_arc(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads ARC-Challenge science questions, falling back to built-in questions if offline."""
    if _is_offline():
        fallback = FALLBACK_QUESTIONS["arc"]
        return fallback[:limit] if limit else fallback
    try:
        from datasets import load_dataset
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        qs = []
        for item in ds:
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            choices = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
            qs.append({
                "prompt": f"Question: {item['question']}\n{choices}\nAnswer (one letter only):",
                "answer": item["answerKey"],
                "task": "arc",
            })
            if limit and len(qs) >= limit:
                return qs
        if qs:
            return qs
    except Exception:
        pass
    fallback = FALLBACK_QUESTIONS["arc"]
    return fallback[:limit] if limit else fallback


def load_ifeval(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads IFEval instruction prompts, falling back to built-in questions if offline."""
    if _is_offline():
        fallback = FALLBACK_QUESTIONS["ifeval"]
        return fallback[:limit] if limit else fallback
    try:
        from datasets import load_dataset
        for repo, split in [("HuggingFaceH4/ifeval", "train"), ("google/IFEval", "train")]:
            try:
                ds = load_dataset(repo, split=split, trust_remote_code=True)
                qs = []
                for item in ds:
                    if not item.get("prompt"):
                        continue
                    qs.append({
                        "prompt": item["prompt"],
                        "instructions": item.get("instruction_id_list", []),
                        "task": "ifeval",
                    })
                    if limit and len(qs) >= limit:
                        return qs
                if qs:
                    return qs
            except Exception:
                continue
    except Exception:
        pass
    fallback = FALLBACK_QUESTIONS["ifeval"]
    return fallback[:limit] if limit else fallback


TASK_LOADERS = {
    "mmlu": load_mmlu,
    "gsm8k": load_gsm8k,
    "arc": load_arc,
    "ifeval": load_ifeval,
}

TASK_SCORERS = {
    "mmlu": lambda resp, q: check_mc(resp, q["answer"]),
    "arc": lambda resp, q: check_mc(resp, q["answer"]),
    "gsm8k": lambda resp, q: check_gsm8k(resp, q["expected"]),
    "ifeval": lambda resp, q: check_ifeval(resp, q.get("instructions", [])),
}
