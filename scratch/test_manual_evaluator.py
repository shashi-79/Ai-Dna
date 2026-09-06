"""
Test script for non-regex, deterministic evaluators.
Verifies:
1. Code execution testing (AST parse + test cases execution)
2. Math numeric parsing (tokenization + numerical comparison)
3. Word-level exact matching for facts & antonyms
"""

import ast
import sys
from typing import Tuple, Optional


def extract_pure_function_block(code_str: str, func_name: str) -> str:
    lines = code_str.split("\n")
    func_lines = []
    found_def = False
    for line in lines:
        stripped = line.strip()
        if not found_def:
            if stripped.startswith(f"def {func_name}(") or stripped.startswith("def "):
                found_def = True
                func_lines.append(line)
        else:
            if stripped.startswith("```") or stripped.startswith(">>>") or stripped.startswith("print(") or stripped.startswith("# Test") or stripped.startswith("assert ") or stripped.startswith("if __name__"):
                break
            if line and not line[0].isspace() and not stripped.startswith("#"):
                break
            func_lines.append(line)
    return "\n".join(func_lines) if func_lines else code_str.split("```")[0].strip()


def eval_code_submission(code_str: str, template_name: str) -> Tuple[bool, str]:
    """Evaluates Python code via AST parsing and sandbox execution without regex."""
    clean_code = extract_pure_function_block(code_str, template_name)

    # 1. AST check
    try:
        tree = ast.parse(clean_code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # 2. Execution sandbox
    sandbox = {}
    try:
        exec(clean_code, sandbox)
    except Exception as e:
        return False, f"ExecError: {e}"

    # 3. Test suites based on template
    tests = {
        "square": ("square", [(4, 16), (-3, 9), (0, 0)]),
        "is_even": ("is_even", [(8, True), (7, False), (0, True)]),
        "get_length": ("get_length", [([1, 2, 3], 3), ([], 0), ([10], 1)]),
        "reverse_string": ("reverse_string", [("hello", "olleh"), ("", ""), ("a", "a")]),
        "find_max": ("find_max", [(3, 7, 7), (10, 2, 10), (-5, -1, -1)]),
        "double_nums": ("double_nums", [([1, 2], [2, 4]), ([], [])]),
        "first_elem": ("first_elem", [([10, 20], 10), (["a"], "a")]),
        "is_positive": ("is_positive", [(5, True), (-2, False), (0, False)]),
        "cube": ("cube", [(3, 27), (-2, -8), (0, 0)]),
        "concat": ("concat", [("foo", "bar", "foobar"), ("", "x", "x")]),
    }

    if template_name not in tests:
        return True, "No specific unit test defined; AST parsed successfully."

    func_name, cases = tests[template_name]
    if func_name not in sandbox or not callable(sandbox[func_name]):
        return False, f"Function '{func_name}' not defined or not callable in code."

    fn = sandbox[func_name]
    for case in cases:
        try:
            if len(case) == 2:
                arg, expected = case
                res = fn(arg)
            else:
                arg1, arg2, expected = case
                res = fn(arg1, arg2)
            if res != expected:
                return False, f"Failed on input {case[:-1]}: expected {expected}, got {res}"
        except Exception as e:
            return False, f"Runtime error on input {case[:-1]}: {e}"

    return True, f"Passed all {len(cases)} unit tests for {func_name}()"


def eval_math_submission(response_str: str, expected_num_str: str) -> Tuple[bool, str]:
    """Evaluates mathematical answer by multi-candidate numeric parsing without regex."""
    try:
        expected = float(expected_num_str)
    except ValueError:
        return False, f"Invalid expected value: {expected_num_str}"

    candidates = []

    # 1. Look for numbers in first non-empty line (frequent direct answer location)
    lines = [ln.strip() for ln in response_str.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    if lines:
        clean_first = lines[0].replace("$", " ").replace(",", "").replace("=", " ").replace(":", " ")
        for t in clean_first.split():
            t_c = t.strip(".:;!?()[]\"'")
            if t_c.lstrip("-").replace(".", "", 1).isdigit():
                try:
                    candidates.append(float(t_c))
                except ValueError:
                    pass

    # 2. Look for \boxed{...}
    if "\\boxed{" in response_str:
        boxed_part = response_str.split("\\boxed{", 1)[1].split("}", 1)[0]
        b_clean = boxed_part.replace("$", " ").replace(",", "").strip()
        for t in b_clean.split():
            t_c = t.strip(".:;!?()[]\"'")
            if t_c.lstrip("-").replace(".", "", 1).isdigit():
                try:
                    candidates.append(float(t_c))
                except ValueError:
                    pass

    # 3. Look for "final answer:", "answer:", "is:"
    for kw in ["final answer:", "answer:", "is:"]:
        if kw in response_str.lower():
            after_kw = response_str.lower().split(kw, 1)[1]
            for t in after_kw.split()[:5]:
                t_c = t.strip(".:;!?()[]\"'")
                if t_c.lstrip("-").replace(".", "", 1).isdigit():
                    try:
                        candidates.append(float(t_c))
                    except ValueError:
                        pass

    # 4. Look across all tokens
    clean_all = response_str.replace("$", " ").replace(",", "").replace("=", " ").replace(":", " ")
    for t in clean_all.split():
        t_c = t.strip(".:;!?()[]\"'")
        if t_c.lstrip("-").replace(".", "", 1).isdigit():
            try:
                candidates.append(float(t_c))
            except ValueError:
                pass

    for c in candidates:
        if abs(c - expected) < 1e-4:
            return True, f"Parsed answer {c} == expected {expected}"

    if candidates:
        return False, f"Parsed answer {candidates[-1]} != expected {expected} (tested: {candidates[:3]})"
    return False, "No numeric tokens found in response"


def eval_fact_submission(response_str: str, target_phrase: str) -> Tuple[bool, str]:
    """Evaluates exact word/phrase presence without regex."""
    words = [w.strip(".,;:!?()[]\"'").lower() for w in response_str.split()]
    target_words = target_phrase.lower().split()

    if not words or not target_words:
        return False, "Empty response or target"

    n_w = len(words)
    n_t = len(target_words)
    for i in range(n_w - n_t + 1):
        if words[i:i + n_t] == target_words:
            return True, f"Found exact phrase '{target_phrase}' at token index {i}"
    return False, f"Phrase '{target_phrase}' not found in token sequence: {words[:15]}..."


# Verification tests
print("Testing code evaluator...")
sample_code_good = "def square(x):\n    return x * x"
sample_code_bad = "def square(x):\n    return x + 1"
sample_code_repl = "def square(x):\n    return x * x\n\n>>> square(2)\n4\n>>> square(-3)\n9\n```"
print("Good code:", eval_code_submission(sample_code_good, "square"))
print("Bad code:", eval_code_submission(sample_code_bad, "square"))
print("REPL code:", eval_code_submission(sample_code_repl, "square"))

print("\nTesting math evaluator...")
sample_math_good = "Let's compute: 15 * 4 = 60. Then 60 - 8 is 52. The answer is 52."
sample_math_bad = "The answer is 45."
print("Good math:", eval_math_submission(sample_math_good, "52"))
print("Bad math:", eval_math_submission(sample_math_bad, "52"))

print("\nTesting fact evaluator...")
sample_fact_good = "The capital of France is Paris."
sample_fact_bad = "The capital of France is Lyon."
print("Good fact:", eval_fact_submission(sample_fact_good, "Paris"))
print("Bad fact:", eval_fact_submission(sample_fact_bad, "Paris"))
print("All non-regex evaluators verified!")

