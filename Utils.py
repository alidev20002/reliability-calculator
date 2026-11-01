from pathlib import Path
import ast

def replace_input_values(
    file_path: str,
    output_path: str | None = None,
) -> int:
    path = Path(file_path)
    source = path.read_bytes()

    tree = ast.parse(source.decode("utf-8"))

    lines = source.splitlines(keepends=True)

    line_offsets = []
    offset = 0

    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    def get_offset(lineno, col_offset):
        return line_offsets[lineno - 1] + col_offset

    replacements = []

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"fill", "send_keys"}
    ]

    calls.sort(key=lambda node: (node.lineno, node.col_offset))

    for call in calls:

        # Determine the argument containing the input value.
        if call.func.attr == "fill":
            if call.args:
                value_node = call.args[0]
            else:
                value_node = next(
                    (kw.value for kw in call.keywords if kw.arg == "value"),
                    None,
                )

        elif call.func.attr == "send_keys":
            value_node = call.args[0] if call.args else None

        else:
            continue

        if value_node is None:
            continue

        input_name = f"input{len(replacements) + 1}"

        start = get_offset(
            value_node.lineno,
            value_node.col_offset,
        )

        end = get_offset(
            value_node.end_lineno,
            value_node.end_col_offset,
        )

        replacement = (
            f'__import__("os").environ["{input_name}"]'
        ).encode()

        replacements.append((start, end, replacement))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]

    ast.parse(source.decode("utf-8"))

    destination = Path(output_path) if output_path else path
    destination.write_bytes(source)

    return len(replacements)


def detect_test_framework(test_case_path: str) -> str:
    """تشخیص فریم‌ورک تست از روی محتوای فایل."""
    source = Path(test_case_path).read_text(encoding="utf-8")

    if (
        "selenium" in source
        or "webdriver" in source
    ):
        if "playwright" not in source:
            return "selenium"

    if (
        "playwright" in source
        or "def test_" in source
        or "async def test_" in source
    ):
        return "playwright"

    return "selenium"