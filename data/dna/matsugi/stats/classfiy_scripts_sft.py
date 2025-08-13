#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face からデータを読み込み、question だけで階層分類（OpenRouter/LLaMA）
→ コンソールに統計を表示し、PNGの棒グラフも保存（任意でJSONLとTXTに書き出し）

- 番号回答化（半角数字のみ）で脱線を抑制
- 失敗時は OTHER へフォールバック（全レベルに OTHER を自動補完）
- dna_hierarchy.json は相対/絶対パスで指定
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from datetime import datetime
from collections import Counter
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt  # 図保存用（seabornは使わない）
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI, APIError, RateLimitError

load_dotenv()  # OPENROUTER_API_KEY を読む


# ===================== 例外 =====================
class QuotaExhausted(Exception):
    pass


# ===================== ユーティリティ =====================
def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def bar_chart(labels, values, title, outpath, rotate=45, annotate=True):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    if annotate:
        for i, v in enumerate(values):
            plt.text(i, v, str(v), ha="center", va="bottom")
    plt.savefig(outpath, dpi=200)
    plt.close()


def build_console_report_text(
    total: int, risk: Counter, typ: Counter, spec: Counter,
    top_n_specific: int, shorten_specific: bool
) -> str:
    def pct(n): return f"{(n/total if total else 0):.2%}"
    lines = []
    lines.append("=== 集計結果 ===")
    lines.append(f"総件数: {total}")
    approx_err = risk.get("CLASSIFICATION_ERROR", 0) + typ.get("CLASSIFICATION_ERROR", 0) + typ.get("SKIPPED", 0)
    lines.append(f"参考: エラー/スキップ含む行(概算): {approx_err} ({pct(approx_err)})")

    lines.append("\n--- Risk Area 分布 ---")
    for k, v in risk.most_common():
        lines.append(f"{k}: {v} ({pct(v)})")

    lines.append("\n--- Type of Harm 分布 ---")
    for k, v in typ.most_common():
        lines.append(f"{k}: {v} ({pct(v)})")

    lines.append(f"\n--- Specific Harm 上位{top_n_specific} ---")
    for k, v in spec.most_common(top_n_specific):
        label = k.split(":")[0].strip() if shorten_specific else k
        lines.append(f"{label}: {v} ({pct(v)})")

    return "\n".join(lines)


def print_console_report(
    total: int, risk: Counter, typ: Counter, spec: Counter,
    top_n_specific: int = 10, shorten_specific: bool = True
) -> str:
    txt = build_console_report_text(total, risk, typ, spec, top_n_specific, shorten_specific)
    print(txt)
    return txt


# ===================== 分類器 =====================
class HarmClassifier:
    """
    - 選択肢を 1..N で提示し、半角数字のみ返させる
    - 解析失敗/範囲外は OTHER にフォールバック
    - dna_hierarchy.json 読込後、全レベル(Risk/Type/Specific)に OTHER を補完
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        hierarchy_file: str,
        request_timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        max_requests: Optional[int] = None,
        max_tokens_soft: Optional[int] = None,
    ):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=request_timeout)
        self.model_name = model_name
        self.h = self._load_hierarchy_with_other(hierarchy_file)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._req_count = 0
        self._tok_count = 0
        self._max_requests = max_requests
        self._max_tokens_soft = max_tokens_soft

    # ---- 階層読み込み + OTHER補完 ----
    def _load_hierarchy_with_other(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            h = json.load(f)

        # ルートに OTHER がなければ追加
        if "OTHER" not in h:
            h["OTHER"] = {"types_of_harm": {"OTHER": {"specific_harms": {"OTHER": "Fallback bucket."}}}}

        for ra, ra_obj in h.items():
            toh = ra_obj.setdefault("types_of_harm", {})
            if "OTHER" not in toh:
                toh["OTHER"] = {"specific_harms": {"OTHER": "Fallback bucket."}}
            for typ, typ_obj in toh.items():
                sh = typ_obj.setdefault("specific_harms", {})
                if "OTHER" not in sh:
                    sh["OTHER"] = "Fallback bucket."
        return h

    # ---- 階層取得 ----
    def _risk_areas(self) -> List[str]:
        return list(self.h.keys())

    def _types_of_harm(self, risk_area: str) -> List[str]:
        return list(self.h[risk_area]["types_of_harm"].keys())

    def _specific_harms(self, risk_area: str, type_of_harm: str) -> List[str]:
        return list(self.h[risk_area]["types_of_harm"][type_of_harm]["specific_harms"].keys())

    # ---- 予算監視 ----
    def _precheck(self):
        if self._max_requests is not None and self._req_count >= self._max_requests:
            raise QuotaExhausted(f"Soft max_requests reached: {self._req_count}/{self._max_requests}")
        if self._max_tokens_soft is not None and self._tok_count >= self._max_tokens_soft:
            raise QuotaExhausted(f"Soft max_tokens reached: {self._tok_count}/{self._max_tokens_soft}")

    # ---- LLM 呼び出し ----
    def _call_llm(self, prompt: str, max_tokens: int = 2, temperature: float = 0.0) -> Optional[str]:
        system_prompt = (
            "あなたはテキスト分類のみを行う専門家です。"
            "指示されたフォーマット以外は一切出力しないでください。"
        )
        self._precheck()
        last_err = None
        for attempt in range(self._max_retries + 1):
            try:
                rr = self.client.chat.completions.with_raw_response.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_tokens,  # 数字1語で足りる
                    temperature=temperature,
                    stop=["\n"],  # 1行で止める（対応していれば有効）
                )
                resp = rr.parse()
                usage = getattr(resp, "usage", None)
                if usage:
                    pt = getattr(usage, "prompt_tokens", 0) or 0
                    ct = getattr(usage, "completion_tokens", 0) or 0
                    self._tok_count += int(pt) + int(ct)
                self._req_count += 1
                self._precheck()
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                last_err = e
                if attempt == self._max_retries:
                    raise QuotaExhausted(f"Rate limit exceeded: {e}") from e
            except APIError as e:
                last_err = e
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                if status in (402, 429) or any(k in msg for k in ("insufficient", "quota", "balance")):
                    raise QuotaExhausted(f"Quota/balance issue: {e}") from e
                logging.warning(f"API error (attempt {attempt+1}/{self._max_retries}): {e}")
            except QuotaExhausted:
                raise
            except Exception as e:
                last_err = e
                logging.warning(f"API呼び出しエラー (attempt {attempt+1}/{self._max_retries}): {e}")
            time.sleep((self._backoff_base ** attempt) * 1.5)
        logging.error(f"API呼び出し失敗: {last_err}")
        return None

    # ---- 番号回答化 ----
    def _classify_step(self, text: str, choices: List[str], step_name: str) -> str:
        # 1-based
        numbered = [f"{i+1}. {c}" for i, c in enumerate(choices)]
        choices_block = "\n".join(numbered)
        prompt = f"""
以下のテキストを読み、指定された「選択肢」から最も適切な{step_name}を**1つ**選んでください。
該当がない/判断困難な場合は「OTHER」を選択してください。
出力は **半角数字のみ**（例: 3）。説明・記号・全角は一切禁止。

--- テキスト ---
{text}

--- 選択肢 ---
{choices_block}

出力は数字のみ。
"""
        out = self._call_llm(prompt, max_tokens=2, temperature=0.0)  # 最短
        m = re.match(r"\s*(\d+)\s*$", out or "")
        if not m:
            return "OTHER" if "OTHER" in choices else choices[-1]
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
        return "OTHER" if "OTHER" in choices else choices[-1]

    # ---- 3段階分類 ----
    def classify_question(self, q: str) -> Dict[str, str]:
        ra  = self._classify_step(q, self._risk_areas(), "リスク領域")
        toh = self._classify_step(q, self._types_of_harm(ra), "害の種類")
        spec_key = self._classify_step(q, self._specific_harms(ra, toh), "具体的な害")
        desc = self.h[ra]["types_of_harm"][toh]["specific_harms"][spec_key]
        return {"risk_area": ra, "type_of_harm": toh, "specific_harm": f"{spec_key}: {desc}"}


# ===================== メイン =====================
def main():
    ap = argparse.ArgumentParser(description="HF入力→分類→コンソール統計＆PNG図（＋任意でJSONL/TXT出力）")
    # HF dataset
    ap.add_argument("--hf_repo_id", required=True, help="例: neko-llm/wj-Adversarial_harmful")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--hf_name", default=None)
    ap.add_argument("--hf_data_files", default=None, help="例: data/train.parquet,data/train.jsonl")
    ap.add_argument("--hf_streaming", action="store_true")
    # fields
    ap.add_argument("--hf_field_id", default="id")
    ap.add_argument("--hf_field_question", default="question")
    # 範囲
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=10**12)
    # モデル/階層
    ap.add_argument("--hierarchy_file", required=True, help="dna_hierarchy.json のパス")
    ap.add_argument("--model_name", default="meta-llama/llama-3-8b-instruct")
    ap.add_argument("--api_key", default=None)
    # 出力（任意）
    ap.add_argument("--output_file", default=None, help="指定すると JSONL にも書き出し")
    ap.add_argument("--report_txt", default=None, help="統計テキストを保存するパス（例: ./reports/stats.txt）")
    ap.add_argument("--log_dir", default="logs")
    # 図/レポート
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--top_n_specific", type=int, default=10)
    ap.add_argument("--shorten_specific", action="store_true")
    args = ap.parse_args()

    # OpenRouter API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("エラー: OPENROUTER_API_KEY が未設定（--api_key か .env）")
        sys.exit(1)

    # ログ
    setup_logging(args.log_dir)
    logging.info(f"HF: {args.hf_repo_id} [{args.hf_split}] name={args.hf_name} streaming={args.hf_streaming}")
    logging.info(f"fields: id={args.hf_field_id}, question={args.hf_field_question}")
    logging.info(f"model: {args.model_name}")

    # HF 読み込み
    try:
        data_files = None
        if args.hf_data_files:
            data_files = [s.strip() for s in args.hf_data_files.split(",") if s.strip()]
        dataset = load_dataset(
            path=args.hf_repo_id,
            name=args.hf_name,
            split=args.hf_split,
            data_files=data_files,
            streaming=args.hf_streaming,
        )
        total_dataset = None if args.hf_streaming else len(dataset)
        logging.info(f"loaded (streaming={args.hf_streaming}, size={'?' if total_dataset is None else total_dataset})")
    except Exception as e:
        logging.error(f"HF読み込みエラー: {e}")
        sys.exit(1)

    # 分類器
    try:
        clf = HarmClassifier(
            api_key=api_key,
            model_name=args.model_name,
            hierarchy_file=args.hierarchy_file,
        )
    except Exception as e:
        logging.error(f"分類器初期化エラー: {e}")
        sys.exit(1)

    # 統計カウンタ
    risk, typ, spec = Counter(), Counter(), Counter()
    processed = 0

    # 出力ファイル
    fout = open(args.output_file, "a", encoding="utf-8") if args.output_file else None

    try:
        if args.hf_streaming:
            for idx, item in tqdm(enumerate(dataset), desc="分類処理(streaming)"):
                if idx < args.start_index: continue
                if idx >= args.end_index:   break
                q = item.get(args.hf_field_question, "")
                if not isinstance(q, str) or not q.strip():
                    continue
                res = clf.classify_question(q.strip())
                processed += 1
                risk[res["risk_area"]] += 1
                typ[res["type_of_harm"]] += 1
                spec[res["specific_harm"]] += 1
                if fout:
                    out_row = {**item, **res}
                    if args.hf_field_id not in out_row:
                        out_row[args.hf_field_id] = idx  # 無い場合はインデックスを仮IDに
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    fout.flush()
        else:
            begin = max(0, args.start_index)
            end = min(args.end_index, len(dataset))
            for i in tqdm(range(begin, end), desc="分類処理"):
                item = dataset[i]
                q = item.get(args.hf_field_question, "")
                if not isinstance(q, str) or not q.strip():
                    continue
                res = clf.classify_question(q.strip())
                processed += 1
                risk[res["risk_area"]] += 1
                typ[res["type_of_harm"]] += 1
                spec[res["specific_harm"]] += 1
                if fout:
                    out_row = {**item, **res}
                    if args.hf_field_id not in out_row:
                        out_row[args.hf_field_id] = i
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    fout.flush()
    except QuotaExhausted as qe:
        logging.warning(f"無料枠/残高/レートが尽きたため停止: {qe}")
    finally:
        if fout:
            fout.close()

    # ---- コンソール出力（＋TXT保存） ----
    report_text = build_console_report_text(
        processed, risk, typ, spec,
        top_n_specific=args.top_n_specific,
        shorten_specific=args.shorten_specific
    )
    print(report_text)

    if args.report_txt:
        out_dir = os.path.dirname(args.report_txt) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.report_txt, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n✅ 統計レポートを保存しました: {os.path.abspath(args.report_txt)}")

    # ---- 図の保存 ----
    os.makedirs(args.outdir, exist_ok=True)

    ra_items = risk.most_common()
    bar_chart([k for k, _ in ra_items], [v for _, v in ra_items],
              f"Risk Area distribution (total={processed})",
              os.path.join(args.outdir, "risk_area_distribution.png"), rotate=30)

    toh_items = typ.most_common()
    bar_chart([k for k, _ in toh_items], [v for _, v in toh_items],
              f"Type of Harm distribution (total={processed})",
              os.path.join(args.outdir, "type_of_harm_distribution.png"), rotate=30)

    sp_items = spec.most_common(args.top_n_specific)
    labels = [k.split(":")[0].strip() if args.shorten_specific else k for k, _ in sp_items]
    values = [v for _, v in sp_items]
    bar_chart(labels, values,
              f"Specific Harm top{args.top_n_specific} (total={processed})",
              os.path.join(args.outdir, f"specific_harm_top{args.top_n_specific}.png"), rotate=60)

    print(f"\n✅ 図を保存しました: {os.path.abspath(args.outdir)}")
    print("  - risk_area_distribution.png")
    print("  - type_of_harm_distribution.png")
    print(f"  - specific_harm_top{args.top_n_specific}.png")


if __name__ == "__main__":
    main()
