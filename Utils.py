from pathlib import Path

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