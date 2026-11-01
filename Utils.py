from pathlib import Path
import ast


def replace_fill_values(
    file_path: str,
    output_path: str | None = None,
) -> int:
    path = Path(file_path)
    source = path.read_bytes()

    tree = ast.parse(source.decode("utf-8"))

    lines = source.splitlines(keepends=True)

    # Calculate absolute byte offset from line/column.
    line_offsets = []
    offset = 0

    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    def get_offset(lineno: int, col_offset: int) -> int:
        return line_offsets[lineno - 1] + col_offset

    # Find all fill() calls and their first argument.
    replacements = []

    calls = [
        node for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fill"
        )
    ]

    calls.sort(key=lambda node: (node.lineno, node.col_offset))

    for index, call in enumerate(calls, start=1):
        # Get the value argument.
        if call.args:
            value_node = call.args[0]
        else:
            value_kwarg = next(
                (kw.value for kw in call.keywords if kw.arg == "value"),
                None,
            )
            if value_kwarg is None:
                continue
            value_node = value_kwarg

        start = get_offset(
            value_node.lineno,
            value_node.col_offset,
        )
        end = get_offset(
            value_node.end_lineno,
            value_node.end_col_offset,
        )

        env_name = f"input{index}"

        replacements.append(
            (start, end, f'__import__("os").environ["{env_name}"]'.encode())
        )

    # Apply replacements from end to beginning.
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]

    # Validate generated Python code.
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