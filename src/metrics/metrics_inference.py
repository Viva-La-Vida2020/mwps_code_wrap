def from_prefix_to_infix(prefix):
    """
    Convert a prefix expression to an infix expression without adding parentheses around numbers.

    Args:
        prefix (list): A list representing a prefix expression (e.g., ["+", "3", "4"]).

    Returns:
        str: The corresponding infix expression (e.g., "3 + 4").
        None: If the prefix expression is invalid.
    """
    stack = []
    operators = {"+", "-", "*", "/", "^"}  # Using set for faster lookup

    for token in reversed(prefix):  # Reverse the prefix expression
        if token not in operators:
            stack.append(str(token))  # Ensure token is a string for concatenation
            continue

        if len(stack) < 2:
            return None  # Invalid expression

        a, b = stack.pop(), stack.pop()

        # Sort operands for commutative operations (+, *)
        if token in {"+", "*"}:
            a, b = sorted([a, b])

        # Add parentheses only if operands are not numbers
        a_wrap = f"({a})" if not a.replace('.', '', 1).isdigit() else a
        b_wrap = f"({b})" if not b.replace('.', '', 1).isdigit() else b

        expr = f"{a_wrap} {token} {b_wrap}"
        stack.append(expr)

    return stack[0] if len(stack) == 1 else None  # Ensure only one valid expression remains



def convert_expression(test, nums):
    """
    Converts symbolic math expressions into numeric values where applicable.

    Args:
        test (list): Expression list containing symbolic or numeric elements.
        nums (list): Number values corresponding to symbols.

    Returns:
        list or None: Converted expression list or None if conversion fails.
    """
    try:
        return [
            nums[int(item[2:]) % len(nums)] if len(item) > 1 and item[0].lower() == 'n' else
            item[2:].replace('_', '.') if len(item) > 1 and item[0].lower() == 'c' else
            item
            for item in test
        ]
    except (ValueError, IndexError):
        return None


def compute_expression(expression, notation="prefix"):
    """
    Evaluates a mathematical expression given in prefix, infix, or postfix notation.

    Args:
        expression (list): List of string elements representing a math expression.
        notation (str): Type of notation ('prefix', 'infix', 'postfix').

    Returns:
        float or None: Computed result or None if evaluation fails.
    """
    operators = {"+", "-", "^", "*", "/"}
    stack = []

    if notation == "prefix":
        expression = list(reversed(expression))
    elif notation == "infix":
        try:
            expr_str = ''.join(expression).replace('[', '(').replace(']', ')').replace('^', '**')
            return eval(expr_str)
        except (SyntaxError, ZeroDivisionError, NameError):
            return None

    for token in expression:
        if token not in operators:
            try:
                stack.append(eval(str(token)))
            except (SyntaxError, NameError):
                return None
        elif len(stack) > 1:
            try:
                a, b = stack.pop(), stack.pop() if notation != "postfix" else (stack.pop(), stack.pop())
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                elif token == "/":
                    stack.append(a / b)
                elif token == "^":
                    stack.append(a ** b)
            except OverflowError:
                return None
            except ZeroDivisionError:
                return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
    return stack[0] if len(stack) == 1 else None


def compute_tree_result(tree_out, nums, notation="prefix"):
    pred_prefix = convert_expression(tree_out, nums)
    pred_infix = from_prefix_to_infix(pred_prefix)
    try:
        computed_result = compute_expression(pred_prefix, notation)
        return pred_prefix, pred_infix, computed_result
    except TypeError:
        return None, None, None
