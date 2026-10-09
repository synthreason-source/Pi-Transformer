
#!/usr/bin/env python3
"""Mutation tester for markov_gen.py.

Features:
- 50 attempts per mutation mode by default.
- Independent source mutations, each starting from the original source.
- Unique seed per attempt for stochastic generation.
- Original and mutated programs receive the same seed for fair comparison.
- Prints mutated source and generated output.
- Tracks output differences and duplicate source mutations.
- Saves mutated scripts only when they exit with code zero.

Example:
    python mutation_tester.py markov_gen.py singlekb.txt --prompt "once upon a time"
"""

import argparse
import ast
import copy
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path


MODES = [
    "nudge_number",
    "flip_operator",
    "duplicate_statement",
    "delete_statement",
    "decrement_number",
    "double_number",
    "halve_number",
    "zero_number",
    "negate_number",
    "flip_boolean_operator",
    "flip_comparison",
    "invert_if_condition",
    "wrap_condition_not",
    "flip_boolean_constant",
    "change_string_literal",
    "append_string_literal",
    "prepend_string_literal",
    "swap_comparison_operands",
    "add_zero_to_expression",
    "multiply_expression_by_one",
    "replace_add_with_multiply",
    "replace_multiply_with_add",
    "invert_unary_operator",
    "remove_assert",
    "change_list_to_tuple",
    "reverse_list_literal",
    "swap_call_arguments",
    "change_range_step",
    "change_return_value",
]

OPERATOR_FLIPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Add,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitOr,
}

COMPARISON_FLIPS = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Gt: ast.LtE,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.LtE: ast.Gt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


def is_number(value):
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


def candidates(tree, cls):
    return [node for node in ast.walk(tree) if isinstance(node, cls)]


def string_constants(tree):
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        # Avoid changing module, class, and function documentation strings.
        result.append(node)
    return result


def statement_candidates(tree):
    result = []
    containers = (
        ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
        ast.Try, ast.ExceptHandler,
    )
    eligible = (
        ast.Assign, ast.AnnAssign, ast.Expr, ast.AugAssign,
        ast.Pass, ast.Return, ast.Assert,
    )

    for parent in ast.walk(tree):
        if not isinstance(parent, containers):
            continue

        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue

        for index, node in enumerate(body):
            if isinstance(node, eligible):
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    continue
                result.append((body, index, node))

    return result


def replace_node(tree, old, new):
    """Replace a node by identity in its first parent."""
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, field, new)
                return True
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if item is old:
                        value[index] = new
                        return True
    return False


def mutate_source(source, mode, attempt):
    """Return one independently mutated source, or None if inapplicable."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    changed = False

    numeric_modes = {
        "nudge_number", "decrement_number", "double_number",
        "halve_number", "zero_number", "negate_number",
    }

    if mode in numeric_modes:
        items = [
            n for n in candidates(tree, ast.Constant)
            if is_number(n.value)
        ]
        if items:
            node = items[attempt % len(items)]
            value = node.value
            variation = attempt // len(items) + 1

            if mode == "nudge_number":
                node.value = value + variation
            elif mode == "decrement_number":
                node.value = value - variation
            elif mode == "double_number":
                node.value = value * (variation + 1)
            elif mode == "halve_number":
                node.value = value / (variation + 1)
            elif mode == "zero_number":
                node.value = 0 if value != 0 else variation
            elif mode == "negate_number":
                node.value = -value if value != 0 else variation

            changed = True

    elif mode in {
        "flip_operator", "add_zero_to_expression",
        "multiply_expression_by_one", "replace_add_with_multiply",
        "replace_multiply_with_add",
    }:
        items = candidates(tree, ast.BinOp)
        if items:
            node = items[attempt % len(items)]

            if mode == "flip_operator":
                replacement = OPERATOR_FLIPS.get(type(node.op))
                if replacement:
                    node.op = replacement()
                    changed = True

            elif mode == "replace_add_with_multiply":
                if isinstance(node.op, ast.Add):
                    node.op = ast.Mult()
                    changed = True

            elif mode == "replace_multiply_with_add":
                if isinstance(node.op, ast.Mult):
                    node.op = ast.Add()
                    changed = True

            elif mode == "add_zero_to_expression":
                node.left = ast.BinOp(
                    left=node.left,
                    op=ast.Add(),
                    right=ast.Constant(value=attempt),
                )
                changed = True

            elif mode == "multiply_expression_by_one":
                node.left = ast.BinOp(
                    left=node.left,
                    op=ast.Mult(),
                    right=ast.Constant(value=attempt + 1),
                )
                changed = True

    elif mode == "flip_boolean_operator":
        items = candidates(tree, ast.BoolOp)
        if items:
            node = items[attempt % len(items)]
            if isinstance(node.op, ast.And):
                node.op = ast.Or()
                changed = True
            elif isinstance(node.op, ast.Or):
                node.op = ast.And()
                changed = True

    elif mode == "flip_comparison":
        items = candidates(tree, ast.Compare)
        if items:
            node = items[attempt % len(items)]
            for index, op in enumerate(node.ops):
                replacement = COMPARISON_FLIPS.get(type(op))
                if replacement:
                    node.ops[index] = replacement()
                    changed = True
                    break

    elif mode in {"invert_if_condition", "wrap_condition_not"}:
        items = candidates(tree, ast.If) + candidates(tree, ast.While)
        if items:
            node = items[attempt % len(items)]
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            changed = True

    elif mode == "flip_boolean_constant":
        items = [
            n for n in candidates(tree, ast.Constant)
            if isinstance(n.value, bool)
        ]
        if items:
            node = items[attempt % len(items)]
            node.value = not node.value
            changed = True

    elif mode in {
        "change_string_literal", "append_string_literal",
        "prepend_string_literal",
    }:
        items = string_constants(tree)
        if items:
            node = items[attempt % len(items)]
            value = node.value
            variation = attempt + 1

            if mode == "change_string_literal":
                node.value = f"{value}_mutation_{variation}"
            elif mode == "append_string_literal":
                node.value = value + ("x" * variation)
            else:
                node.value = ("x" * variation) + value

            changed = True

    elif mode == "swap_comparison_operands":
        items = candidates(tree, ast.Compare)
        if items:
            node = items[attempt % len(items)]
            if len(node.ops) == 1 and len(node.comparators) == 1:
                left = node.left
                right = node.comparators[0]
                node.left = right
                node.comparators[0] = left
                reverse = {
                    ast.Gt: ast.Lt,
                    ast.Lt: ast.Gt,
                    ast.GtE: ast.LtE,
                    ast.LtE: ast.GtE,
                }
                replacement = reverse.get(type(node.ops[0]))
                if replacement:
                    node.ops[0] = replacement()
                changed = True

    elif mode in {"duplicate_statement", "delete_statement"}:
        items = statement_candidates(tree)
        if items:
            body, index, node = items[attempt % len(items)]
            if mode == "duplicate_statement":
                body.insert(index + 1, copy.deepcopy(node))
                changed = True
            elif len(body) > 1:
                del body[index]
                changed = True

    elif mode == "invert_unary_operator":
        items = candidates(tree, ast.UnaryOp)
        if items:
            node = items[attempt % len(items)]
            if isinstance(node.op, ast.UAdd):
                node.op = ast.USub()
                changed = True
            elif isinstance(node.op, ast.USub):
                node.op = ast.UAdd()
                changed = True
            elif isinstance(node.op, ast.Not):
                node.operand = ast.UnaryOp(
                    op=ast.Not(), operand=node.operand
                )
                changed = True
            elif isinstance(node.op, ast.Invert):
                node.op = ast.USub()
                changed = True

    elif mode == "remove_assert":
        items = candidates(tree, ast.Assert)
        if items:
            node = items[attempt % len(items)]
            changed = replace_node(tree, node, ast.Pass())

    elif mode == "change_list_to_tuple":
        items = candidates(tree, ast.List)
        if items:
            node = items[attempt % len(items)]
            replacement = ast.Tuple(elts=node.elts, ctx=node.ctx)
            changed = replace_node(tree, node, replacement)

    elif mode == "reverse_list_literal":
        items = candidates(tree, ast.List)
        if items:
            node = items[attempt % len(items)]
            if len(node.elts) > 1:
                node.elts.reverse()
                changed = True
            elif len(node.elts) == 1:
                node.elts.append(copy.deepcopy(node.elts[0]))
                changed = True

    elif mode == "swap_call_arguments":
        items = [
            n for n in candidates(tree, ast.Call)
            if len(n.args) >= 2
        ]
        if items:
            node = items[attempt % len(items)]
            index = attempt % (len(node.args) - 1)
            node.args[index], node.args[index + 1] = (
                node.args[index + 1], node.args[index]
            )
            changed = True

    elif mode == "change_range_step":
        items = [
            n for n in candidates(tree, ast.Call)
            if isinstance(n.func, ast.Name) and n.func.id == "range"
        ]
        if items:
            node = items[attempt % len(items)]
            step = attempt + 2
            if len(node.args) < 3:
                while len(node.args) < 2:
                    node.args.append(ast.Constant(value=0))
                node.args.append(ast.Constant(value=step))
            else:
                node.args[2] = ast.Constant(value=step)
            changed = True

    elif mode == "change_return_value":
        items = candidates(tree, ast.Return)
        if items:
            node = items[attempt % len(items)]
            if node.value is None:
                node.value = ast.Constant(value=attempt + 1)
            else:
                node.value = ast.Constant(value=None)
            changed = True

    if not changed:
        return None

    ast.fix_missing_locations(tree)

    try:
        result = ast.unparse(tree) + "\n"
        compile(result, "<mutation>", "exec")
    except (SyntaxError, TypeError, ValueError):
        return None

    if result.strip() == source.strip():
        return None

    return result


def execute_source(source, script, corpus, prompt, extra_args, seed, timeout):
    """Run source from a temporary file next to the target script."""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix="mutation_run_",
            dir=str(script.parent),
            delete=False,
            newline="\n",
        ) as handle:
            handle.write(source)
            temp_path = Path(handle.name)

        command = [
            sys.executable,
            str(temp_path),
            str(corpus),
            "--prompt", prompt,
            *extra_args,
            "--seed", str(seed),
        ]

        try:
            result = subprocess.run(
                command,
                cwd=str(script.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            return {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timeout": False,
            }

        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {
                "command": command,
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "timeout": True,
            }

    finally:
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def save_source(output_dir, script_stem, mode, attempt, source):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{script_stem}_{mode}_{attempt:03d}.py"
    path.write_text(source, encoding="utf-8", newline="\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run independent Python source mutations and compare output."
    )
    parser.add_argument("script_path", type=Path)
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("--prompt", default="once upon a time")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--modes", nargs="*", choices=MODES, default=None)
    parser.add_argument("--mutations-per-mode", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("successful_source_mutations"),
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    script = args.script_path.resolve()
    corpus = args.corpus_path.resolve()
    output_dir = args.output_dir.resolve()

    if not script.is_file():
        parser.error(f"Target script not found: {script}")
    if not corpus.is_file():
        parser.error(f"Corpus not found: {corpus}")
    if args.mutations_per_mode < 1:
        parser.error("--mutations-per-mode must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        original_source = script.read_text(encoding="utf-8")
        ast.parse(original_source)
    except (OSError, UnicodeError, SyntaxError) as exc:
        parser.error(f"Could not read/parse target source: {exc}")

    modes = args.modes if args.modes is not None else MODES

    stats = {
        "attempts": 0,
        "executed": 0,
        "saved": 0,
        "failed": 0,
        "timeouts": 0,
        "skipped": 0,
        "output_changed": 0,
        "output_unchanged": 0,
        "duplicate_sources": 0,
    }

    # Track source hashes to identify repeated source variants.
    source_hashes = set()
    saved_files = []
    start_time = time.time()
    stop = False

    print("=" * 80)
    print("MARKOV GENERATOR MUTATION TESTER")
    print("=" * 80)
    print(f"Script:              {script}")
    print(f"Corpus:              {corpus}")
    print(f"Prompt:              {args.prompt!r}")
    print(f"Mutation modes:      {len(modes)}")
    print(f"Attempts per mode:   {args.mutations_per_mode}")
    print(f"Planned attempts:    {len(modes) * args.mutations_per_mode}")
    print("Seed strategy:       unique seed per attempt; paired baseline")
    print(f"Save directory:      {output_dir}")
    print("=" * 80)

    for mode in modes:
        if stop:
            break

        print(f"\n\n{'#' * 80}\nMODE: {mode}\n{'#' * 80}")

        for attempt_index in range(args.mutations_per_mode):
            if args.limit and stats["executed"] >= args.limit:
                print("Reached execution limit.")
                stop = True
                break

            stats["attempts"] += 1
            attempt_number = attempt_index + 1
            seed = args.seed + stats["executed"]

            print(
                f"\n--- {mode}: mutation "
                f"{attempt_number}/{args.mutations_per_mode}, seed={seed} ---"
            )

            mutated = mutate_source(original_source, mode, attempt_index)

            if mutated is None:
                stats["skipped"] += 1
                print("SKIPPED: no applicable mutation could be produced.")
                continue

            digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
            if digest in source_hashes:
                stats["duplicate_sources"] += 1
                print("NOTE: source mutation has appeared before.")
            else:
                source_hashes.add(digest)

            print("\n" + "=" * 80)
            print("MUTATED SOURCE")
            print("=" * 80)
            #print(mutated.rstrip())
            print("=" * 80)

            # Run the unmodified source and mutation with the same seed.
            baseline = execute_source(
                original_source, script, corpus, args.prompt,
                args.extra_arg, seed, args.timeout,
            )
            result = execute_source(
                mutated, script, corpus, args.prompt,
                args.extra_arg, seed, args.timeout,
            )
            stats["executed"] += 1

            print("\n--- MUTATED GENERATED STDOUT ---")
            print(result["stdout"] if result["stdout"] else "(no stdout)")

            if result["stderr"]:
                print("\n--- MUTATED STDERR ---")
                print(result["stderr"])

            if result["timeout"]:
                stats["timeouts"] += 1
                stats["failed"] += 1
                print("\nSTATUS: TIMEOUT")

            elif result["returncode"] != 0:
                stats["failed"] += 1
                print(f"\nSTATUS: FAILED (exit code {result['returncode']})")

            else:
                print("\nSTATUS: PROCESS EXITED SUCCESSFULLY")

                path = save_source(
                    output_dir, script.stem, mode, attempt_number, mutated
                )
                saved_files.append(path)
                stats["saved"] += 1
                print(f"SAVED: {path}")

            if baseline["timeout"]:
                print("BASELINE COMPARISON: baseline timed out; comparison unavailable.")
            elif baseline["returncode"] != 0:
                print(
                    "BASELINE COMPARISON: original failed "
                    f"(exit code {baseline['returncode']}); comparison unavailable."
                )
            elif result["returncode"] == 0:
                if result["stdout"] != baseline["stdout"]:
                    stats["output_changed"] += 1
                    print("OUTPUT COMPARISON: CHANGED relative to original.")
                else:
                    stats["output_unchanged"] += 1
                    print(
                        "OUTPUT COMPARISON: IDENTICAL to original for this seed."
                    )

            if args.stop_on_error and (
                result["timeout"] or result["returncode"] != 0
            ):
                stop = True
                break

    elapsed = time.time() - start_time

    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Mutation attempts considered: {stats['attempts']}")
    print(f"Mutation executions:          {stats['executed']}")
    print(f"Successful source files saved:{stats['saved']:>6}")
    print(f"Failed runs:                  {stats['failed']}")
    print(f"Timed-out runs:               {stats['timeouts']}")
    print(f"Skipped inapplicable attempts:{stats['skipped']:>6}")
    print(f"Repeated source variants:     {stats['duplicate_sources']}")
    print(f"Outputs changed vs baseline:  {stats['output_changed']}")
    print(f"Outputs identical to baseline:{stats['output_unchanged']:>6}")
    print(f"Elapsed seconds:              {elapsed:.2f}")
    print(f"Saved files directory:        {output_dir}")

    if saved_files:
        print("\nSuccessful mutation files:")
        for path in saved_files:
            print(f"  {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()

