"""Render a human-readable review sheet from benchmark JSONL files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "eval" / "web_answer_cases.jsonl"
SOURCE_BANK = ROOT / "eval" / "web_answer_source_bank.jsonl"
OUTPUT = ROOT / "eval" / "results" / "WEB_BENCHMARK_REVIEW.md"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_class(case: dict) -> str:
    kind = case.get("source_type", "")
    if kind in {"official_qa", "official", "government", "government_service", "gov.il – רשות המסים"}:
        return "מקור רשמי / ממשלתי — מועמד חזק לקורפוס"
    if kind == "constructed_from_web_topics":
        return "מקרה clarification שנבנה אצלנו — לא מסמך להוספה"
    return "מקור משני — צריך לאמת מול מקור רגולטורי ראשוני"


def render_case(index: int, case: dict, *, executed: bool) -> list[str]:
    lines = [f"### {index}. {case['question']}", "", f"**סוג מקור:** {case.get('source_type', 'לא צוין')}", f"**סיווג:** {source_class(case)}"]
    if case.get("reference_url") or case.get("url"):
        lines.append(f"**מקור:** {case.get('reference_url') or case.get('url')}")
    if case.get("primary_source"):
        lines.append(f"**מקור ראשוני מזוהה:** {case['primary_source']}")
    lines += ["", "**תשובת ייחוס:**", f"> {case.get('reference_answer') or case.get('answer') or 'אין תשובת ייחוס'}", ""]
    if case.get("expected_conclusion"):
        lines.append(f"**מסקנה צפויה:** {', '.join(case['expected_conclusion'])}")
    if case.get("required_concepts"):
        lines.append(f"**מושגים/תנאים שחייבים להופיע:** {', '.join(case['required_concepts'])}")
    if case.get("required_actions"):
        lines.append(f"**פעולה צפויה:** {', '.join(case['required_actions'])}")
    if case.get("must_not_include"):
        lines.append(f"**טענות שאסור להסיק:** {', '.join(case['must_not_include'])}")
    if case.get("requires_clarification"):
        lines.append(f"**נדרשת הבהרה:** כן — {', '.join(case.get('clarification_terms', []))}")
    if executed:
        lines.append("**סטטוס:** הורץ מול RegBot במסגרת benchmark")
    else:
        lines.append("**סטטוס:** נאסף ל־source bank; טרם הורץ וטרם אושר כ־gold answer")
    lines += ["", "---", ""]
    return lines


def main() -> None:
    executed = load(EXECUTED)
    source_bank = load(SOURCE_BANK)
    lines = [
        "# דף בדיקה ידני — Benchmark תשובות RegBot",
        "",
        "> מטרת הקובץ: לאפשר לגיא לבדוק האם השאלות והתשובות אכן שייכות לקורפוס הרגולטורי של RegBot, או שהן דורשות מקור חיצוני/שאלת הבהרה.",
        "",
        "## איך לקרוא את הקובץ",
        "",
        "- **מקור רשמי / ממשלתי:** מועמד חזק להיכלל בקורפוס, בכפוף לתוקף ולכפילויות.",
        "- **מקור משני:** משמש seed לשאלה; יש לאמת מול חוק, תקנות או חוזר ראשוני.",
        "- **מקרה clarification:** לא מסמך להוספה. הוא בודק אם הבוט יודע לבקש נתונים חסרים.",
        "- תשובת ייחוס מאתר אינה בהכרח אמת משפטית; היא נקודת התחלה לבדיקה.",
        "",
        f"## חלק א׳ — {len(executed)} שאלות שהורצו מול הבוט",
        "",
    ]
    for i, case in enumerate(executed, 1):
        lines.extend(render_case(i, case, executed=True))
    lines += [f"## חלק ב׳ — {len(source_bank)} שאלות נוספות שנאספו, טרם הורצו", ""]
    for i, case in enumerate(source_bank, 1):
        lines.extend(render_case(i, case, executed=False))
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(executed)} executed + {len(source_bank)} source-bank cases)")


if __name__ == "__main__":
    main()
