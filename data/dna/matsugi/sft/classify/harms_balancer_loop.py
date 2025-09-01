#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harms_balancer_loop.py

- inbox(JSONL) の question を 3階層 taxonomy に分類してカテゴリ別 JSONL へ逐次追記
- まずルールベース、未分類なら LLM(OpenRouter) で「番号回答化の3段階分類」
- 出力は id/question/answer を保持 + 分類結果(risk_area/type_of_harm/specific_harm)
- min/max/target_total でバランス制御、needs.json で不足可視化
- 進捗の定期レポート: progress.json / counts.json を _logs に出力、コンソールにも要約表示
- マスター出力 all.jsonl にも逐次追記
- 厳密な途中再開: checkpoint.json（既読行数）と seen_ids.json（重複防止）でリジューム

使い方（例）:
python harms_balancer_loop.py \
  --taxonomy dna_hierarchy.json \
  --inbox questions_inbox.jsonl \
  --out_dir shards \
  --min_per_class 40 \
  --max_per_class 42 \
  --target_total 2500 \
  --model_name meta-llama/llama-3-8b-instruct \
  --report_every 30
"""

import os
import re
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError  # OpenRouter互換
load_dotenv()  # .env を読む（OPENROUTER_API_KEY）

# ===================== デフォルトの簡易ルール =====================
DEFAULT_RULES = {
    "Insult": [r"(?i)\b(stupid|idiot|moron|dumb)\b"],
    "Obscene": [r"(?i)\b(fuck|shit|bitch|asshole)\b"],
    "Racial/Ethnic Discrimination": [
        r"(?i)\b(italian|mexican|japanese|arab|jewish|chinese|korean|african)\b.*\b(mafia|cartel|yakuza|terrorist)\b"
    ],
    "Rumors": [r"(?i)\b(rumor|rumour|unverified|alleged(ly)?)\b"],
}

# ===================== ユーティリティ =====================
def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(name))

def short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def load_taxonomy(path: str) -> Dict[str, dict]:
    tax = json.loads(Path(path).read_text(encoding="utf-8"))
    # OTHER を全レベルに補完
    if "OTHER" not in tax:
        tax["OTHER"] = {"types_of_harm": {"OTHER": {"specific_harms": {"OTHER": "Fallback bucket."}}}}
    for ra, ra_obj in tax.items():
        toh = ra_obj.setdefault("types_of_harm", {})
        if "OTHER" not in toh:
            toh["OTHER"] = {"specific_harms": {"OTHER": "Fallback bucket."}}
        for typ, typ_obj in toh.items():
            sh = typ_obj.setdefault("specific_harms", {})
            if "OTHER" not in sh:
                sh["OTHER"] = "Fallback bucket."
    return tax

def load_rules(rules_path: Optional[str], leafs: List[str]) -> Dict[str, List[str]]:
    merged = {k: v[:] for k, v in DEFAULT_RULES.items()}
    if rules_path and os.path.exists(rules_path):
        # 形式は JSON: { "Specific Harm": ["regex1", ...], ... }
        user = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        for leaf, pats in (user or {}).items():
            merged.setdefault(leaf, [])
            merged[leaf].extend(pats or [])
    # taxonomy に存在する leaf のみ
    return {leaf: merged.get(leaf, []) for leaf in leafs}

def match_specific(text: str, rules: Dict[str, List[str]], priority: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str) or not text.strip():
        return None, None
    keys = priority if priority else list(rules.keys())
    for leaf in keys:
        for pattern in rules.get(leaf, []):
            try:
                if re.search(pattern, text):
                    return leaf, pattern
            except re.error:
                continue
    return None, None

# ===================== LLM（番号回答化の3段階分類） =====================
class QuotaExhausted(Exception):
    pass

class LLMHierarchyClassifier:
    def __init__(self, api_key: str, model_name: str, taxonomy: Dict[str, dict],
                 request_timeout: float = 60.0, max_retries: int = 5, backoff_base: float = 2.0):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=request_timeout)
        self.model = model_name
        self.h = taxonomy
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    def _risk_areas(self) -> List[str]:
        return list(self.h.keys())

    def _types_of_harm(self, risk_area: str) -> List[str]:
        return list(self.h[risk_area]["types_of_harm"].keys())

    def _specific_harms(self, risk_area: str, type_of_harm: str) -> List[str]:
        return list(self.h[risk_area]["types_of_harm"][type_of_harm]["specific_harms"].keys())

    def _call_llm(self, prompt: str, max_tokens: int = 2, temperature: float = 0.0) -> Optional[str]:
        system_prompt = (
            "あなたはテキスト分類のみを行う専門家です。"
            "指示されたフォーマット以外は一切出力しないでください。"
        )
        last_err = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n"],
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                last_err = e
            except APIError as e:
                last_err = e
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                if status in (402, 429) or any(k in msg for k in ("insufficient", "quota", "balance")):
                    raise QuotaExhausted(f"Quota/balance issue: {e}") from e
            except Exception as e:
                last_err = e
            time.sleep((self._backoff_base ** attempt) * 1.5)
        return None

    def _classify_step(self, text: str, choices: List[str], step_name: str) -> str:
        numbered = [f"{i+1}. {c}" for i, c in enumerate(choices)]
        prompt = f"""
以下のテキストを読み、指定された「選択肢」から最も適切な{step_name}を**1つ**選んでください。
該当がない/判断困難な場合は「OTHER」を選択してください。
出力は **半角数字のみ**（例: 3）。説明・記号・全角は禁止。

--- テキスト ---
{text}

--- 選択肢 ---
{chr(10).join(numbered)}

出力は数字のみ。
""".strip()
        out = self._call_llm(prompt, max_tokens=2, temperature=0.0)
        m = re.match(r"\s*(\d+)\s*$", out or "")
        if not m:
            return "OTHER" if "OTHER" in choices else choices[-1]
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
        return "OTHER" if "OTHER" in choices else choices[-1]

    def classify(self, text: str) -> Tuple[str, str, str]:
        ra = self._classify_step(text, self._risk_areas(), "リスク領域")
        toh = self._classify_step(text, self._types_of_harm(ra), "害の種類")
        spec = self._classify_step(text, self._specific_harms(ra, toh), "具体的な害")
        return ra, toh, spec

# ===================== 既存件数/不足/追記 =====================
def load_existing_counts(out_dir: str, taxonomy: Dict[str, dict]) -> Dict[Tuple[str, str, str], int]:
    counts = {}
    for ra, ra_obj in taxonomy.items():
        for toh, toh_obj in ra_obj["types_of_harm"].items():
            for spec in toh_obj["specific_harms"].keys():
                shard = Path(out_dir) / sanitize(ra) / sanitize(toh) / f"{sanitize(spec)}.jsonl"
                n = 0
                if shard.exists():
                    with shard.open("r", encoding="utf-8") as f:
                        for _ in f:
                            n += 1
                counts[(ra, toh, spec)] = n
    return counts

def write_needs(out_dir: str, taxonomy: Dict[str, dict], counts: Dict[Tuple[str, str, str], int],
                min_per_class: int, max_per_class: int):
    needs = []
    for ra, ra_obj in taxonomy.items():
        for toh, toh_obj in ra_obj["types_of_harm"].items():
            for spec in toh_obj["specific_harms"].keys():
                c = counts.get((ra, toh, spec), 0)
                needs.append({
                    "risk_area": ra,
                    "type_of_harm": toh,
                    "specific_harm": spec,
                    "current": c,
                    "need_to_min": max(0, min_per_class - c),
                    "room_to_max": max(0, max_per_class - c),
                })
    log_dir = Path(out_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "needs.json").write_text(json.dumps(needs, ensure_ascii=False, indent=2), encoding="utf-8")

def append_to_master(out_dir: str, out_row: dict):
    master = Path(out_dir) / "all.jsonl"
    master.parent.mkdir(parents=True, exist_ok=True)
    with master.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

def append_to_shard(out_dir: str, ra: str, toh: str, spec: str, record: dict):
    # id/question/answer を保ち、分類結果を付与して書き出し
    out_row = {
        "id": record.get("id"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "risk_area": ra,
        "type_of_harm": toh,
        "specific_harm": spec,  # 説明を付けたいなら taxonomy から取り出し追加可
    }
    p = Path(out_dir) / sanitize(ra) / sanitize(toh)
    p.mkdir(parents=True, exist_ok=True)
    with (p / f"{sanitize(spec)}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    # マスター出力にも追記
    append_to_master(out_dir, out_row)

# ===================== チェックポイント & 重複管理 =====================
def load_checkpoint(out_dir: str) -> int:
    cp = Path(out_dir) / "_logs" / "checkpoint.json"
    if cp.exists():
        try:
            return int(json.loads(cp.read_text(encoding="utf-8")).get("seen_lines", 0))
        except Exception:
            return 0
    return 0

def save_checkpoint(out_dir: str, seen_lines: int):
    logdir = Path(out_dir) / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "checkpoint.json").write_text(
        json.dumps({"seen_lines": int(seen_lines)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def load_seen_ids(out_dir: str) -> set:
    p = Path(out_dir) / "_logs" / "seen_ids.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()

def save_seen_ids(out_dir: str, seen_ids: set):
    logdir = Path(out_dir) / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    # サイズが大きくなりすぎないよう末尾 ~20万件だけ保持
    (logdir / "seen_ids.json").write_text(
        json.dumps(sorted(list(seen_ids))[-200000:], ensure_ascii=False),
        encoding="utf-8"
    )

# ===================== 分類パイプ（ルール → LLM → OTHER） =====================
def classify_one(text: str, taxonomy: Dict[str, dict], rules: Dict[str, List[str]],
                 llm: Optional[LLMHierarchyClassifier]) -> Tuple[str, str, str]:
    # 1) ルール優先（速い・安い）
    leaf, _ = match_specific(text, rules)
    if leaf:
        for ra, ra_obj in taxonomy.items():
            for toh, toh_obj in ra_obj["types_of_harm"].items():
                if leaf in toh_obj["specific_harms"]:
                    return ra, toh, leaf
    # 2) LLM フォールバック
    if llm:
        try:
            return llm.classify(text)
        except QuotaExhausted:
            pass
        except Exception:
            pass
    # 3) 完全フォールバック
    return "OTHER", "OTHER", "OTHER"

# ===================== メイン =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--rules", default=None)                 # ルールJSON（任意）
    ap.add_argument("--inbox", default="questions_inbox.jsonl")
    ap.add_argument("--out_dir", default="shards")
    ap.add_argument("--text_col", default="question")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--min_per_class", type=int, default=40)   # 例: 40〜42で ≈2500件
    ap.add_argument("--max_per_class", type=int, default=42)
    ap.add_argument("--target_total", type=int, default=2500)  # 総数の目標
    ap.add_argument("--poll_seconds", type=int, default=10)
    ap.add_argument("--idle_stop_rounds", type=int, default=30)
    ap.add_argument("--report_every", type=int, default=60, help="何秒ごとに進捗を表示/保存するか")

    # LLM
    ap.add_argument("--model_name", default="meta-llama/llama-3-8b-instruct")
    ap.add_argument("--api_key", default=None)

    args = ap.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    leafs = [spec for ra in taxonomy for toh in taxonomy[ra]["types_of_harm"]
             for spec in taxonomy[ra]["types_of_harm"][toh]["specific_harms"]]
    rules = load_rules(args.rules, leafs)

    # OpenRouter
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    llm = LLMHierarchyClassifier(api_key=api_key, model_name=args.model_name, taxonomy=taxonomy) if api_key else None
    if not api_key:
        print("⚠️ OPENROUTER_API_KEY 未設定。ルールのみで分類します。")

    # 途中再開: 既存件数を読み込み
    counts = load_existing_counts(args.out_dir, taxonomy)
    total_now = sum(counts.values())

    # 既読オフセット（checkpoint.json から復元）
    inbox_path = Path(args.inbox)
    seen_offsets = load_checkpoint(args.out_dir)
    seen_ids = load_seen_ids(args.out_dir)

    # inbox が短くなっていた場合の丸め
    if inbox_path.exists():
        current_lines = len(inbox_path.read_text(encoding="utf-8").splitlines())
        if seen_offsets > current_lines:
            seen_offsets = current_lines

    idle_rounds = 0
    last_report_ts = 0.0

    while True:
        accepted = 0

        if inbox_path.exists():
            lines = inbox_path.read_text(encoding="utf-8").splitlines()
            L = len(lines)
            # 既読丸め（外部でinboxが縮んでいた場合に備える）
            if seen_offsets > L:
                seen_offsets = L

            for i in range(seen_offsets, L):
                try:
                    rec = json.loads(lines[i])
                except Exception:
                    seen_offsets += 1
                    continue

                text = rec.get(args.text_col, "")
                if not isinstance(text, str) or not text.strip():
                    seen_offsets += 1
                    continue

                # 総数が目標に達していたら打ち切り
                if total_now >= args.target_total:
                    break

                # 重複チェック（idが優先、なければquestionハッシュ）
                rec_id = rec.get(args.id_col) if args.id_col else rec.get("id")
                if rec_id is not None:
                    key = f"id:{rec_id}"
                else:
                    key = f"q:{short_hash(text)}"

                if key in seen_ids:
                    seen_offsets += 1
                    continue

                # 分類
                ra, toh, spec = classify_one(text, taxonomy, rules, llm)

                # クラス上限
                if counts.get((ra, toh, spec), 0) >= args.max_per_class:
                    seen_offsets += 1
                    continue

                # 追記
                append_to_shard(args.out_dir, ra, toh, spec, rec)
                counts[(ra, toh, spec)] = counts.get((ra, toh, spec), 0) + 1
                total_now += 1
                accepted += 1
                seen_ids.add(key)

                # 逐次的に checkpoint を前進（大規模でも復旧が早い）
                seen_offsets += 1

            # ループ末で checkpoint を保存
            save_checkpoint(args.out_dir, seen_offsets)

        # 不足レポートを更新
        write_needs(args.out_dir, taxonomy, counts, args.min_per_class, args.max_per_class)

        # ===== 進捗スナップショット & コンソール要約 =====
        now = time.time()
        if now - last_report_ts >= args.report_every:
            last_report_ts = now

            # 不足上位（need_to_min が大きい順に上位5）
            needs_path = Path(args.out_dir) / "_logs" / "needs.json"
            top_lacking = []
            if needs_path.exists():
                try:
                    needs = json.loads(needs_path.read_text(encoding="utf-8"))
                    needs_sorted = sorted(needs, key=lambda x: x.get("need_to_min", 0), reverse=True)
                    top_lacking = [x for x in needs_sorted if x.get("need_to_min", 0) > 0][:5]
                except Exception:
                    top_lacking = []

            # counts.json 保存（ツリー形式）
            counts_dict: Dict[str, Dict[str, Dict[str, int]]] = {}
            for (ra, toh, spec), c in counts.items():
                counts_dict.setdefault(ra, {}).setdefault(toh, {})[spec] = c

            logs_dir = Path(args.out_dir) / "_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "counts.json").write_text(json.dumps(counts_dict, ensure_ascii=False, indent=2), encoding="utf-8")

            progress = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "total_now": total_now,
                "min_per_class": args.min_per_class,
                "max_per_class": args.max_per_class,
                "target_total": args.target_total,
                "accepted_last_round": accepted,
                "top_lacking": top_lacking,
                "seen_lines": seen_offsets,
                "seen_ids_size": len(seen_ids),
            }
            (logs_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

            lacking_str = ", ".join(f"{x['specific_harm']}(-{x['need_to_min']})" for x in top_lacking) or "なし"
            print(f"[{progress['timestamp']}] total={total_now} (+{accepted}) / target={args.target_total} | 既読={seen_offsets} | 不足上位: {lacking_str}")

            # seen_ids を定期保存（大規模時は間隔を広げてもOK）
            if len(seen_ids) % 500 == 0:
                save_seen_ids(args.out_dir, seen_ids)

        # 停止条件：総数が target_total 以上、かつ全クラスが min を満たす
        all_min_met = all(c >= args.min_per_class for c in counts.values())
        if total_now >= args.target_total and all_min_met:
            print("✅ 目標到達（総数・最小件数を満たしました）。停止します。")
            # 最終チェックポイントと seen_ids を保存
            save_checkpoint(args.out_dir, seen_offsets)
            save_seen_ids(args.out_dir, seen_ids)
            break

        # 進捗がないラウンドが続いたら停止（枯渇）
        if accepted == 0:
            idle_rounds += 1
        else:
            idle_rounds = 0
        if idle_rounds >= args.idle_stop_rounds:
            print("⏹ 新規採用ゼロが続いたため停止しました。")
            # 最終チェックポイントと seen_ids を保存
            save_checkpoint(args.out_dir, seen_offsets)
            save_seen_ids(args.out_dir, seen_ids)
            break

        time.sleep(args.poll_seconds)

if __name__ == "__main__":
    main()
