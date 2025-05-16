import ast
import re
from src.utils.utils import load_json, save_json
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────────────────────────────────
# 1.  Expression  →  sequential steps  (m_1, m_2, …)
# ──────────────────────────────────────────────────────────────────────────
def expression_to_steps(expr: str) -> List[str]:
    """
    Turn a fully-parenthesised arithmetic expression into
    ['m_1 = ...', 'm_2 = ...', …, 'm_k = ...'].
    """
    tree        = ast.parse(expr, mode="eval").body
    ops         = {ast.Add:"+", ast.Sub:"-", ast.Mult:"*", ast.Div:"/", ast.Pow: '**',}
    steps, idx  = [], 1                      # m_1, m_2, …

    def dfs(node) -> str:
        nonlocal idx, steps
        if isinstance(node, ast.BinOp):                 # recurse – left, right, parent
            left  = dfs(node.left)
            right = dfs(node.right)
            temp  = f"m_{idx}"; idx += 1
            steps.append(f"{temp} = {left} {ops[type(node.op)]} {right}")
            return temp
        if isinstance(node, ast.Name):      return node.id
        if isinstance(node, ast.Constant):  return str(node.value)
        raise ValueError(f"Unsupported node {ast.dump(node)}")

    dfs(tree)
    return steps


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


# ──────────────────────────────────────────────────────────────────────────
# 3.  Convenience wrapper  (expression + nums  →  answer, steps)
# ──────────────────────────────────────────────────────────────────────────
def solve_expression(expr: str, nums: List[float]) -> Tuple[float, List[str]]:
    steps   = expression_to_steps(expr)
    answer  = evaluate_steps(steps, nums)
    return answer, steps


def compute_prefix_tree_result(target_text, nums):
    try:
        pred_prefix = target_text.split("Steps:")[-1].strip()

        lst = ast.literal_eval(pred_prefix)
        lst_no_spaces = [item.replace(" ", "") for item in lst]
        lst_no_spaces = [item.replace("^", "**") for item in lst_no_spaces]

        answer = evaluate_steps(lst_no_spaces, nums)
    except:
        answer = None

    return answer

# ──────────────────────────────────────────────────────────────────────────
# ✨ Example
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_json('../../data/mathqa/train_steps.jsonl')
    for d in data:
        # if len(d['infix']) == 1:
        #     steps = f"m1 = {d['infix'][0]}"
        #     # answer = evaluate_steps(steps, nums)
        # else:
        #     infix = " ".join(d['infix'])
        #     infix = infix.replace("^", "**")
        #     nums = d['nums']  # [25.0, 20.0]
        #     steps = expression_to_steps(infix)
        #     answer = evaluate_steps(steps, nums)
        #
        # new_steps = []
        # for step in steps:
        #     new_steps.append(step.replace("**", "^"))
        # d['steps'] = new_steps

        steps = d['steps']
        text = d['text']
        nums = d['nums']
        input_text = f"Question: {text} Steps: "
        target_text = f"Question: {text} Steps: {steps}"

        pred_prefix = target_text.split("Steps:")[-1].strip()

        lst = ast.literal_eval(pred_prefix)
        lst_no_spaces = [item.replace(" ", "") for item in lst]
        lst_no_spaces = [item.replace("^", "**") for item in lst_no_spaces]
        # new_steps = []
        # for step in pred_prefix:
        #     new_steps.append(step.replace("^", "**"))
        # pred_prefix = new_steps
        answer = evaluate_steps(lst_no_spaces, nums)

        print(answer)
        # print(steps, answer, d['answer'])
        # if answer - d['answer'] > 1e-6:
        #     print(d['id'])
    # save_json(data, '../../data/math23k/train_steps.jsonl')