"""Minimal CRAG experiment application.

All experiment settings and pipeline stages live in this file:
preprocessing, Contriever retrieval, T5 evaluation, action triggering,
knowledge refinement/search, generation, metrics, and experiment logging.
Model/API calls are intentionally isolated behind small functions so they can
be enabled without changing the project structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

Action = Literal["correct", "incorrect", "ambiguous"]


@dataclass(frozen=True)
class Config:
    dataset: str = "popqa"
    seed: int = 42
    retriever: str = "facebook/contriever"
    evaluator: str = "google-t5/t5-large"
    generator: str = "meta-llama/Llama-2-7b-hf"
    top_k_documents: int = 10
    top_k_internal_strips: int = 5
    top_k_web_paragraphs: int = 5
    internal_threshold: float = 0.5
    external_threshold: float = -0.5
    query_keyword_limit: int = 3
    thresholds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "popqa": (0.59, -0.99),
            "pubhealth": (0.50, -0.91),
            "arc_challenge": (0.50, -0.91),
            "biography": (0.95, -0.91),
        }
    )


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"


def make_logger(experiment: str) -> logging.Logger:
    now = datetime.now(timezone.utc)
    directory = LOGS / "dev" / "crag" / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"crag.{experiment}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(directory / "app.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def preprocess_csv(input_file: Path, output_file: Path, remove_emoji: bool = False) -> int:
    """Normalize the project CSV and write one processed CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_file.open(newline="", encoding="utf-8") as source, output_file.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {input_file}")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            for key, value in row.items():
                row[key] = " ".join((value or "").split())
                if remove_emoji:
                    row[key] = re.sub(r"[^\x00-\x7F]+", "", row[key])
            writer.writerow(row)
            count += 1
    return count


def retrieve(question: str, documents: Iterable[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Return a deterministic lexical fallback until Contriever is enabled."""
    words = set(re.findall(r"\w+", question.lower()))
    ranked = []
    for document in documents:
        text = str(document.get("text", document.get("full_text", "")))
        overlap = len(words.intersection(re.findall(r"\w+", text.lower())))
        ranked.append({**document, "retrieval_score": overlap})
    return sorted(ranked, key=lambda item: item["retrieval_score"], reverse=True)[:top_k]


def evaluate_relevance(question: str, text: str) -> float:
    """Return a bounded lexical score; replace this function with fine-tuned T5."""
    question_words = set(re.findall(r"\w+", question.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not question_words:
        return -1.0
    return max(-1.0, min(1.0, 2 * len(question_words & text_words) / len(question_words) - 1))


def choose_action(scores: list[float], dataset: str, config: Config) -> Action:
    if not scores:
        return "incorrect"
    high, low = config.thresholds.get(dataset, config.thresholds["popqa"])
    if max(scores) > high:
        return "correct"
    if all(score < low for score in scores):
        return "incorrect"
    return "ambiguous"


def refine_internal(question: str, documents: list[dict[str, Any]], config: Config) -> str:
    strips: list[tuple[float, int, str]] = []
    for index, document in enumerate(documents):
        text = str(document.get("text", document.get("full_text", "")))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            score = evaluate_relevance(question, sentence)
            if score >= config.internal_threshold:
                strips.append((score, index, sentence.strip()))
    strips.sort(key=lambda item: (-item[0], item[1]))
    return " ".join(item[2] for item in strips[: config.top_k_internal_strips])


def rewrite_query(question: str, config: Config) -> str:
    words = re.findall(r"[A-Za-zА-Яа-яӨөҮүЁё0-9-]+", question.lower())
    return " ".join(words[: config.query_keyword_limit])


def search_external(question: str, config: Config) -> str:
    """External search integration point; credentials are never stored in code."""
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        return ""
    _ = rewrite_query(question, config)
    return ""


def build_context(question: str, action: Action, internal: str, external: str) -> str:
    if action == "correct":
        return f"Question: {question}\nKnowledge: {internal}"
    if action == "incorrect":
        return f"Question: {question}\nKnowledge: {external}"
    return f"Question: {question}\nInternal: {internal}\nExternal: {external}"


def generate(context: str, config: Config) -> str:
    """Generator integration point for LLaMA-2 or Self-RAG."""
    return context


def run(question: str, documents: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    retrieved = retrieve(question, documents, config.top_k_documents)
    scored = [{**doc, "evaluator_score": evaluate_relevance(
        question, str(doc.get("text", doc.get("full_text", "")))
    )} for doc in retrieved]
    action = choose_action([doc["evaluator_score"] for doc in scored], config.dataset, config)
    internal = refine_internal(question, scored, config) if action in ("correct", "ambiguous") else ""
    external = search_external(question, config) if action in ("incorrect", "ambiguous") else ""
    context = build_context(question, action, internal, external)
    return {
        "question": question,
        "action": action,
        "answer": generate(context, config),
        "retrieved": scored,
        "internal_knowledge": internal,
        "external_knowledge": external,
    }


def save_result(result: dict[str, Any], experiment: str) -> Path:
    directory = OUTPUTS / experiment
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal CRAG pipeline")
    parser.add_argument("command", choices=("preprocess", "run", "evaluate", "ablate"))
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--question", default="")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--input", type=Path, default=DATA / "raw" / "input.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = make_logger(args.experiment)
    config = Config(dataset=args.dataset)
    if args.command == "preprocess":
        count = preprocess_csv(args.input, DATA / "processed" / f"{args.dataset}.csv")
        logger.info("processed_rows=%s", count)
        return
    result = run(args.question, [], config)
    path = save_result(result, args.experiment)
    logger.info("saved=%s action=%s", path, result["action"])


if __name__ == "__main__":
    main()
