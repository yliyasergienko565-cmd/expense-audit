#!/usr/bin/env python3
import sys
import os
import re
import argparse
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation

import pandas as pd

DATE_KEYWORDS = ["дата", "date", "день", "day"]
CATEGORY_KEYWORDS = ["категория", "category", "категории", "статья", "тип", "type", "rubric"]
AMOUNT_KEYWORDS = ["сумма", "amount", "sum", "total", "cost", "price", "стоимость", "итог", "выплата"]
DESCRIPTION_KEYWORDS = ["описание", "description", "comment", "коммент", "назначение", "note", "детали", "purpose"]


def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def normalize(s):
    return str(s).strip().lower()


def find_column_by_keywords(columns, keywords):
    for col in columns:
        norm = normalize(col)
        for kw in keywords:
            if kw in norm:
                return col
    return None


def normalize_amount(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        parts = s.split(",")
        if len(parts[-1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def detect_amount_column(df, exclude):
    best_col, best_score = None, -1.0
    for c in df.columns:
        if c in exclude:
            continue
        parsed = df[c].map(normalize_amount)
        valid_ratio = parsed.notna().mean()
        if valid_ratio < 0.8:
            continue
        values = [float(v) for v in parsed.dropna()]
        if len(values) < 2:
            continue
        variance = pd.Series(values).var() or 0
        score = variance * valid_ratio
        if score > best_score:
            best_score, best_col = score, c
    return best_col


def detect_date_column(df, exclude):
    for c in df.columns:
        if c in exclude:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return c
    return None


def detect_category_column(df, exclude):
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        series = df[c].dropna()
        if series.empty:
            continue
        nunique = series.nunique()
        ratio = nunique / max(len(series), 1)
        if 0 < nunique <= 50 and ratio < 0.5:
            candidates.append((ratio, c))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None


def detect_description_column(df, exclude):
    candidates = [c for c in df.columns if c not in exclude]
    if not candidates:
        return None
    candidates.sort(key=lambda c: -df[c].nunique(dropna=True))
    return candidates[0]


def detect_columns(df):
    columns = list(df.columns)

    date_col = find_column_by_keywords(columns, DATE_KEYWORDS)
    amount_col = find_column_by_keywords(columns, AMOUNT_KEYWORDS)
    category_col = find_column_by_keywords(columns, CATEGORY_KEYWORDS)
    description_col = find_column_by_keywords(columns, DESCRIPTION_KEYWORDS)

    exclude = set(c for c in [date_col, amount_col, category_col, description_col] if c)

    if amount_col is None:
        amount_col = detect_amount_column(df, exclude)
        if amount_col:
            exclude.add(amount_col)
    if date_col is None:
        date_col = detect_date_column(df, exclude)
        if date_col:
            exclude.add(date_col)
    if category_col is None:
        category_col = detect_category_column(df, exclude)
        if category_col:
            exclude.add(category_col)
    if description_col is None:
        description_col = detect_description_column(df, exclude)

    return {"date": date_col, "category": category_col, "amount": amount_col, "description": description_col}


def prepare_rows(df, cols):
    rows = []
    for i, raw in df.iterrows():
        amount = normalize_amount(raw[cols["amount"]])
        if amount is None:
            continue
        row = {
            "_line": i + 2,
            "amount": amount,
            "date": str(raw[cols["date"]]).strip() if cols["date"] and pd.notna(raw[cols["date"]]) else "",
            "category": str(raw[cols["category"]]).strip() if cols["category"] and pd.notna(raw[cols["category"]]) else "",
            "description": str(raw[cols["description"]]).strip() if cols["description"] and pd.notna(raw[cols["description"]]) else "",
        }
        rows.append(row)
    return rows


def format_row(r, cols):
    parts = []
    if cols["date"]:
        parts.append(r["date"])
    if cols["category"]:
        parts.append(r["category"] or "(без категории)")
    if cols["description"]:
        parts.append(r["description"])
    parts.append(str(r["amount"]))
    return " | ".join(parts)


def total_sum(rows):
    return sum((r["amount"] for r in rows), Decimal("0"))


def sum_by_category(rows):
    sums = defaultdict(lambda: Decimal("0"))
    counts = defaultdict(int)
    for r in rows:
        key = r["category"] or "(без категории)"
        sums[key] += r["amount"]
        counts[key] += 1
    return sorted(sums.items(), key=lambda kv: kv[1], reverse=True), counts


def top_n(rows, n=5):
    return sorted(rows, key=lambda r: r["amount"], reverse=True)[:n]


def find_duplicates(rows, cols):
    key_fields = [f for f in ("date", "description") if cols[f]]
    if not key_fields:
        return None  # not enough info to define a duplicate
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[f] for f in key_fields) + (r["amount"],)
        groups[key].append(r)
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_anomalies(rows, cols):
    anomalies = []
    if cols["category"]:
        by_group = defaultdict(list)
        for r in rows:
            by_group[r["category"] or "(без категории)"].append(r)
    else:
        by_group = {"(вся выборка)": rows}

    for _, group_rows in by_group.items():
        amounts = [float(r["amount"]) for r in group_rows]
        if len(amounts) < 2:
            continue
        mean = statistics.mean(amounts)
        std = statistics.pstdev(amounts)
        if std == 0:
            continue
        for r in group_rows:
            if abs(float(r["amount"]) - mean) > 3 * std:
                anomalies.append((r, mean, std))
    anomalies.sort(key=lambda x: x[0]["amount"], reverse=True)
    return anomalies


def find_negative(rows):
    return [r for r in rows if r["amount"] < 0]


def write_report(path, rows, cols, source_path):
    total = total_sum(rows)
    by_cat, counts = sum_by_category(rows) if cols["category"] else (None, None)
    top5 = top_n(rows, 5)
    duplicates = find_duplicates(rows, cols)
    anomalies = find_anomalies(rows, cols)
    negatives = find_negative(rows)

    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЁТ ПО АУДИТУ РАСХОДОВ")
    lines.append(f"Файл: {source_path}")
    lines.append(
        "Определены столбцы: "
        f"дата={cols['date'] or '-'}, категория={cols['category'] or '-'}, "
        f"сумма={cols['amount']}, описание={cols['description'] or '-'}"
    )
    lines.append("=" * 70)

    lines.append("")
    lines.append("1. ОБЩАЯ СУММА РАСХОДОВ")
    lines.append("-" * 70)
    lines.append(f"{total}  (строк учтено: {len(rows)})")

    lines.append("")
    lines.append("2. СУММА ПО КАТЕГОРИЯМ (по убыванию)")
    lines.append("-" * 70)
    if by_cat is not None:
        for category, amount in by_cat:
            lines.append(f"{category:<25} кол-во={counts[category]:<6} сумма={amount}")
    else:
        lines.append("Столбец категории не найден — разбивка недоступна")

    lines.append("")
    lines.append("3. ТОП-5 САМЫХ КРУПНЫХ ТРАТ")
    lines.append("-" * 70)
    for r in top5:
        lines.append(format_row(r, cols))

    lines.append("")
    if duplicates is not None:
        lines.append(f"4. ДУБЛИКАТЫ (одинаковые ключевые поля + сумма) — найдено групп: {len(duplicates)}")
        lines.append("-" * 70)
        if duplicates:
            for group in duplicates.values():
                sample = format_row(group[0], cols)
                lines.append(f"{sample}  -> повторов: {len(group)} (строки: {', '.join(str(r['_line']) for r in group)})")
        else:
            lines.append("Не найдено")
    else:
        lines.append("4. ДУБЛИКАТЫ")
        lines.append("-" * 70)
        lines.append("Недостаточно столбцов (нет даты и описания) — проверка не выполнена")

    lines.append("")
    lines.append(f"5. АНОМАЛИИ (отклонение от среднего более чем на 3 стандартных отклонения) — найдено: {len(anomalies)}")
    lines.append("-" * 70)
    if anomalies:
        for r, mean, std in anomalies:
            lines.append(f"{format_row(r, cols)}  [среднее={mean:.2f}, std={std:.2f}]")
    else:
        lines.append("Не найдено")

    lines.append("")
    lines.append(f"6. СТРОКИ С ОТРИЦАТЕЛЬНЫМИ СУММАМИ — найдено: {len(negatives)}")
    lines.append("-" * 70)
    if negatives:
        for r in negatives:
            lines.append(format_row(r, cols))
    else:
        lines.append("Не найдено")

    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Аудит расходов из CSV/Excel файла")
    parser.add_argument("input", help="Путь к CSV или Excel файлу с расходами")
    parser.add_argument("-o", "--output", default="report.txt", help="Путь к файлу отчёта (по умолчанию report.txt)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Файл не найден: {args.input}")
        sys.exit(1)

    df = load_table(args.input)
    cols = detect_columns(df)

    if cols["amount"] is None:
        print("Не удалось определить столбец с суммой. Проверьте файл вручную.")
        sys.exit(1)

    rows = prepare_rows(df, cols)
    if not rows:
        print("Не найдено ни одной строки с корректной суммой.")
        sys.exit(1)

    write_report(args.output, rows, cols, args.input)
    print(f"Обработано строк: {len(rows)}")
    print(f"Определены столбцы: {cols}")
    print(f"Отчёт сохранён в {args.output}")


if __name__ == "__main__":
    main()
