import ast
import re
from src.utils.utils import load_json, save_json
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────────────────────────────────
# 2.  Evaluate the generated steps with given nums
# ──────────────────────────────────────────────────────────────────────────
CONST_RE = re.compile(r'\bC_[0-9_]+\b')       # matches C_100, C_3_14, …

def parse_const(name: str) -> float:
    """Convert C_… to float:  C_100 → 100.0,  C_3_14 → 3.14"""
    s = name[2:]                              # strip the leading "C_"
    if "_" in s:
        int_part, frac = s.split("_", 1)
        return float(f"{int_part}.{frac.replace('_','')}")
    return float(s)

def evaluate_steps(steps: List[str],
                   nums : List[float],
                   ) -> float:
    """
    Run the step list and return the numerical answer.
    """
    env: Dict[str, float] = {}

    # map N_i → nums[i]
    env.update({f"N_{i}": v for i, v in enumerate(nums)})

    # map all constants encountered
    for step in steps:
        for cname in CONST_RE.findall(step):
            if cname not in env:
                env[cname] = parse_const(cname)

    # execute steps sequentially
    for step in steps:
        left, expr = map(str.strip, step.split("=", 1))
        env[left]  = eval(expr, {"__builtins__": None}, env)

    return env[left]                           # result of the last assignment


def compute_steps_result(target_text, nums):
    try:
        pred_prefix = target_text.split("Steps:")[-1].strip()

        lst = ast.literal_eval(pred_prefix)
        lst_no_spaces = [item.replace(" ", "") for item in lst]
        lst_no_spaces = [item.replace("^", "**") for item in lst_no_spaces]

        answer = evaluate_steps(lst_no_spaces, nums)
    except:
        answer = None
        
    return answer
