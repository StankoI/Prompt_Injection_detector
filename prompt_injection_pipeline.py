#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_injection_pipeline.py
============================

Pipeline за подготовка на dataset за откриване на prompt injection.

Стъпки (изпълняват се последователно от main()):
    1. download_datasets()  - сваля train + test split-овете от HuggingFace
    2. normalize_dataset()  - привежда всеки dataset до вида: prompt / marker (0/1)
    3. merge_datasets()     - слива всичко в един dataset (+ source/split метаданни)
    4. split_by_marker()    - разделя на два: само инжекции / само чисти
    5. save_outputs()       - записва CSV (+ по желание XLSX) с готова 'result' колона

Ръчната част (екипът): добавяте span тагове <INJECTION>...</INJECTION> в 'result'
колоната на injections файла, после merge-вате двата файла обратно.

Изисквания:
    pip install datasets pandas
    pip install openpyxl        # само ако искаш и .xlsx изход (по подразбиране да)

Употреба:
    python prompt_injection_pipeline.py --output-dir ./out
    python prompt_injection_pipeline.py --token hf_xxx        # за gated dataset
    HF_TOKEN=hf_xxx python prompt_injection_pipeline.py
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

DATASETS = [
    # {
    #     "name": "rikka_multilingual",
    #     "repo_id": "rikka-snow/prompt-injection-multilingual",
    #     "config": None,                          # един subset
    #     "text_cols": ["text", "prompt", "user_message"],
    #     "label_col": "label",
    #     "label_map": None,                       # вече е 0/1
    #     "gated": False,
    # },
    # {
    #     "name": "finguard_finance",
    #     "repo_id": "nandhak12/finguard-finance-injection-dataset",
    #     "config": None,
    #     "text_cols": ["user_message", "text", "prompt"],
    #     "label_col": "label",
    #     "label_map": {"SAFE": 0, "ATTACK": 1},   # текстов етикет -> 0/1
    #     "gated": False,
    # },
    {
        "name": "neuralchemy_injection",
        "repo_id": "neuralchemy/Prompt-injection-dataset",
        "config": "full",                        # subset 'full' (15.9k), не 'core'
        "text_cols": ["text", "prompt", "user_message"],
        "label_col": "label",
        "label_map": None,                       # вече е 0/1
        "gated": False,
    },
    # {
    #     "name": "injection_attack_detection",
    #     "repo_id": "PromptInjectionDataset/Injection-Attack-Detection-Dataset",
    #     "config": None,
    #     "text_cols": ["input", "text", "prompt", "user_message", "sentence"],
    #     "label_col": "output",                   # реалната label колона; вече е 0/1
    #     "label_map": None,
    #     "gated": True,
    # },
]

# Думи, които при автоматично разпознаване третираме като "инжекция" (1) или "чисто" (0)
POSITIVE_TOKENS = {"attack", "injection", "jailbreak", "malicious", "unsafe",
                   "prompt_injection", "true", "yes", "1"}
NEGATIVE_TOKENS = {"safe", "benign", "legitimate", "clean", "normal",
                   "false", "no", "0"}

INJECTION_OPEN = "<INJECTION>"
INJECTION_CLOSE = "</INJECTION>"


# ===========================================================================
# СТЪПКА 1 — СВАЛЯНЕ
# ===========================================================================
def download_datasets(token=None):
    """
    Сваля всеки dataset от HuggingFace (всички split-ове, train + test и т.н.).

    Връща речник: name -> pandas.DataFrame, където DataFrame-ът съдържа
    оригиналните колони + допълнителна колона 'split' (train/test/...).

    Gated dataset-ите се прескачат с ясно съобщение, ако няма валиден токен.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Липсва библиотеката 'datasets'. Инсталирай я с:  pip install datasets")

    raw = {}
    for cfg in DATASETS:
        name, repo_id = cfg["name"], cfg["repo_id"]
        print(f"\n[1/5] Сваляне: {repo_id}  (name={name})")

        kwargs = {}
        if cfg["gated"]:
            if not token:
                print(f"  ! ПРЕСКОЧЕН: '{repo_id}' е gated, но не е подаден токен.")
                print(f"    -> Влез в страницата, натисни Agree, вземи токен от")
                print(f"       https://huggingface.co/settings/tokens и подай --token hf_xxx")
                continue
            kwargs["token"] = token

        try:
            config = cfg.get("config")
            if config:
                ds = load_dataset(repo_id, config, **kwargs)
            else:
                ds = load_dataset(repo_id, **kwargs)
        except Exception as exc:
            print(f"  ! ГРЕШКА при сваляне на '{repo_id}': {exc}")
            if cfg["gated"]:
                print("    (Провери дали си приел условията на сайта и дали токенът е валиден.)")
            continue

        frames = []
        for split_name, split_ds in ds.items():
            part = split_ds.to_pandas()
            part["split"] = split_name
            frames.append(part)
            print(f"    split '{split_name}': {len(part)} реда")

        if frames:
            raw[name] = pd.concat(frames, ignore_index=True)

    if not raw:
        sys.exit("\nНито един dataset не беше свален успешно. Спирам.")
    return raw


# ===========================================================================
# СТЪПКА 2 — НОРМАЛИЗАЦИЯ ДО prompt / marker (0/1)
# ===========================================================================
def _pick_text_column(df, candidates):
    """Връща първата налична текстова колона от списъка с кандидати."""
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == object and col != "split":
            return col
    raise KeyError(f"Не намерих текстова колона. Налични колони: {list(df.columns)}")


def _detect_label_column(df):
    """Автоматично разпознаване на колоната с етикета (за gated dataset)."""
    for cand in ("label", "labels", "is_injection", "injection", "class", "target", "y"):
        if cand in df.columns:
            return cand
    raise KeyError(f"Не намерих колона с етикет. Налични колони: {list(df.columns)}")


def _coerce_marker(series, label_map):
    """Привежда колоната с етикет до int 0/1."""
    if label_map is None:                       # вече числа 0/1
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)

    if isinstance(label_map, dict):             # директно мапване SAFE/ATTACK -> 0/1
        out = series.astype(str).str.strip().map(label_map)
        if out.isna().any():
            bad = series[out.isna()].unique()[:5]
            raise ValueError(f"Непознати стойности за етикет (не са в label_map): {bad}")
        return out.astype(int)

    # label_map == "auto": опитваме числа, после текст по ключови думи
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).clip(0, 1)

    def to01(v):
        s = str(v).strip().lower()
        if s in POSITIVE_TOKENS:
            return 1
        if s in NEGATIVE_TOKENS:
            return 0
        return None

    out = series.map(to01)
    if out.isna().any():
        bad = series[out.isna()].unique()[:5]
        raise ValueError(f"Не можах да разпозная етикети автоматично: {bad}")
    return out.astype(int)


def normalize_dataset(df, cfg):
    """
    Привежда един dataset до две колони:
        prompt  (str)  - текстът
        marker  (int)  - 1 ако е инжекция, иначе 0
    Запазва и source (кой dataset) и split (train/test).
    """
    text_col = _pick_text_column(df, cfg["text_cols"])
    label_col = cfg["label_col"] or _detect_label_column(df)

    out = pd.DataFrame()
    out["prompt"] = df[text_col].astype(str).str.strip()
    out["marker"] = _coerce_marker(df[label_col], cfg["label_map"])
    out["source"] = cfg["name"]
    out["split"] = df["split"] if "split" in df.columns else "unknown"

    # махаме празни редове
    out = out[out["prompt"].str.len() > 0].reset_index(drop=True)
    print(f"    нормализиран '{cfg['name']}': {len(out)} реда "
          f"(инжекции={int(out['marker'].sum())}, чисти={int((out['marker']==0).sum())})")
    return out


# ===========================================================================
# СТЪПКА 3 — СЛИВАНЕ
# ===========================================================================
def merge_datasets(normalized_frames, dedup=True):
    """Слива всички нормализирани dataset-и в един. По желание маха дубликати по prompt."""
    print("\n[3/5] Сливане на dataset-ите")
    merged = pd.concat(normalized_frames, ignore_index=True)
    print(f"    общо преди дедупликация: {len(merged)} реда")

    if dedup:
        before = len(merged)
        # пазим първото срещане на даден prompt (за да не дублираме идентични текстове)
        merged = merged.drop_duplicates(subset="prompt", keep="first").reset_index(drop=True)
        print(f"    премахнати дубликати: {before - len(merged)}")

    print(f"    финален обединен dataset: {len(merged)} реда "
          f"(инжекции={int(merged['marker'].sum())}, "
          f"чисти={int((merged['marker']==0).sum())})")
    return merged


# ===========================================================================
# СТЪПКА 4 — РАЗДЕЛЯНЕ НА ДВА (само инжекции / само чисти)
# ===========================================================================
def split_by_marker(merged):
    """Разделя обединения dataset на два: инжекции (marker==1) и чисти (marker==0)."""
    print("\n[4/5] Разделяне на два временни dataset-а")
    injections = merged[merged["marker"] == 1].reset_index(drop=True)
    clean = merged[merged["marker"] == 0].reset_index(drop=True)
    print(f"    инжекции: {len(injections)} реда")
    print(f"    чисти:    {len(clean)} реда")
    return injections, clean


# ===========================================================================
# СТЪПКА 5 — ЗАПИС
# ===========================================================================
def _sanitize_for_xlsx(df):
    """
    Прави текста безопасен за Excel/xlsx, без да пипа CSV-то:
      1. маха невалидни контролни Unicode символи (те чупят xlsx с 'recover' грешка)
      2. слага апостроф пред клетки, започващи с = + - @ (Excel ги мисли за формули)
    Връща ново DataFrame (оригиналът остава непокътнат за CSV).
    """
    import re
    # позволени контролни: \t (0x09), \n (0x0A), \r (0x0D); останалите 0x00-0x1F + 0x7F махаме
    bad_ctrl = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def clean(val):
        if not isinstance(val, str):
            return val
        s = bad_ctrl.sub("", val)
        if s[:1] in ("=", "+", "-", "@"):
            s = "'" + s      
        return s

    return df.apply(lambda col: col.map(clean))


def _add_result_column(df, is_injection):
    """
    Добавя готова 'result' колона за ръчната обработка.
      - чисти изречения: result = самото изречение (нищо за маркиране)
      - инжекции: result = изречението като начало, екипът обвива инжекцията с тагове
    Подреждаме колоните удобно: prompt | result | marker | source | split
    """
    out = df.copy()
    out["result"] = out["prompt"] 
    return out[["prompt", "result", "marker", "source", "split"]]


def _save_one(df, path_csv, write_xlsx):
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print(f"    записан: {path_csv}  ({len(df)} реда)")
    if write_xlsx:
        try:
            path_xlsx = path_csv.with_suffix(".xlsx")
            _sanitize_for_xlsx(df).to_excel(path_xlsx, index=False)
            print(f"    записан: {path_xlsx}")
        except Exception as exc:
            print(f"    (XLSX пропуснат: {exc} — инсталирай 'openpyxl' ако го искаш)")


def save_outputs(merged, injections, clean, out_dir, write_xlsx=True):
    """Записва трите файла: обединен, само-инжекции, само-чисти."""
    print("\n[5/5] Запис на изходните файлове")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_one(merged[["prompt", "marker", "source", "split"]],
              out_dir / "merged_all.csv", write_xlsx)
    _save_one(_add_result_column(injections, is_injection=True),
              out_dir / "injections_only.csv", write_xlsx)
    _save_one(_add_result_column(clean, is_injection=False),
              out_dir / "clean_only.csv", write_xlsx)

    print(f"\nГотово. Файловете са в: {out_dir.resolve()}")
    print("Следва ръчната стъпка: в 'injections_only' обвийте инжекцията в колоната")
    print(f"'result' с {INJECTION_OPEN} ... {INJECTION_CLOSE}, после merge-нете двата файла.")


# ===========================================================================
# MAIN — изпълнява петте стъпки последователно
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline за prompt injection dataset.")
    parser.add_argument("--output-dir", default="./out", help="Папка за изходните файлове")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace токен (за gated dataset). Или env HF_TOKEN.")
    parser.add_argument("--no-dedup", action="store_true", help="Без премахване на дубликати")
    parser.add_argument("--no-xlsx", action="store_true", help="Без .xlsx изход (само CSV)")
    args = parser.parse_args()

    # 1. сваляне
    raw = download_datasets(token=args.token)

    # 2. нормализация (по име, за да хванем правилния config за всеки)
    print("\n[2/5] Нормализация до prompt / marker (0/1)")
    cfg_by_name = {c["name"]: c for c in DATASETS}
    normalized = []
    for name, df in raw.items():
        try:
            normalized.append(normalize_dataset(df, cfg_by_name[name]))
        except Exception as exc:
            print(f"  ! ПРЕСКОЧЕН '{name}' при нормализация: {exc}")
    if not normalized:
        sys.exit("Нито един dataset не беше нормализиран успешно. Спирам.")

    # 3. сливане
    merged = merge_datasets(normalized, dedup=not args.no_dedup)

    # 4. разделяне
    injections, clean = split_by_marker(merged)

    # 5. запис
    save_outputs(merged, injections, clean, args.output_dir, write_xlsx=not args.no_xlsx)


if __name__ == "__main__":
    main()