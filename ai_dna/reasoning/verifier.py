"""
Reasoning Verifier & Step-by-Step Chain-of-Thought Reward Calculator.
Supports DeepSeek-R1 / OpenAI o1-style self-verification for:
1. Mathematical calculation correctness
2. Chain-of-Thought structure verification (<thought>...</thought>)
3. Multi-modal logical consistency & formatting rules
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional


class ReasoningVerifier:
    """
    Computes rule-based rewards for generated trajectories to guide GRPO policy optimization.
    """
    def __init__(
        self,
        format_reward_weight: float = 0.25,
        accuracy_reward_weight: float = 1.0,
        tag_start: str = "<thought>",
        tag_end: str = "</thought>",
    ):
        self.format_reward_weight = format_reward_weight
        self.accuracy_reward_weight = accuracy_reward_weight
        self.tag_start = tag_start
        self.tag_end = tag_end

    def verify_thought_format(self, text: str) -> float:
        """Checks whether the response contains valid <thought>...</thought> reasoning structure."""
        has_start = self.tag_start in text
        has_end = self.tag_end in text
        if has_start and has_end:
            # Ensure start tag comes before end tag
            if text.find(self.tag_start) < text.find(self.tag_end):
                thought_content = text.split(self.tag_start)[1].split(self.tag_end)[0].strip()
                if len(thought_content) > 5:
                    return 1.0  # Full format reward
        elif has_start or has_end:
            return 0.2  # Partial credit for opening/closing
        return 0.0

    def verify_math_solution(self, generated_text: str, ground_truth_answer: str) -> float:
        """Verifies if the extracted final numerical answer matches ground truth."""
        clean_text = generated_text.strip()
        
        # 1. Exact match in answer section after </thought>
        if self.tag_end in clean_text:
            final_answer = clean_text.split(self.tag_end)[-1].strip()
        else:
            final_answer = clean_text

        # Extract all numbers
        numbers = re.findall(r"[-+]?\d*\.?\d+", final_answer)
        if numbers:
            last_number = numbers[-1]
            try:
                if abs(float(last_number) - float(ground_truth_answer)) < 1e-4:
                    return 1.0
            except ValueError:
                pass

        # Check raw string inclusion
        if str(ground_truth_answer) in final_answer:
            return 0.8

        return 0.0

    def compute_composite_reward(
        self,
        generated_text: str,
        ground_truth_answer: str,
        token_length: int,
        max_length: int = 128,
    ) -> Dict[str, float]:
        """Calculates total scalar reward across format, accuracy, and length penalty."""
        r_format = self.verify_thought_format(generated_text)
        r_acc = self.verify_math_solution(generated_text, ground_truth_answer)
        
        # Length penalty (encourage concise reasoning without infinite loops)
        len_penalty = -0.1 if token_length >= max_length - 2 else 0.0

        total_reward = (
            self.format_reward_weight * r_format
            + self.accuracy_reward_weight * r_acc
            + len_penalty
        )

        return {
            "reward_total": total_reward,
            "reward_accuracy": r_acc,
            "reward_format": r_format,
            "len_penalty": len_penalty,
        }
