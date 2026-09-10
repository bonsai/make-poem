"""PoC workflow: rhyme-finder → SEM (Owlready2) → PHO → ranking."""

import json
import os
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from owlready2 import DataProperty, Thing, get_ontology


class PoemState(TypedDict, total=False):
    seed: str
    context: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    trace: dict[str, Any]


CRITERIA = {
    "phonetic": "deterministic",
    "semantic": "deterministic",
    "context": "rule",
    "rhythm": "deterministic",
    "novelty": "deterministic",
    "style": "llm",
}

RESULT_FILE = "result.json"


def build_semantic_ontology():
    """Build the minimal SEM ontology used by the PoC."""
    onto = get_ontology("http://bonsai.local/make-poem-sem.owl")
    with onto:
        class SemanticWord(Thing):
            pass

        class has_mood(DataProperty):
            range = [str]

        class has_scene(DataProperty):
            range = [str]

        class has_theme(DataProperty):
            range = [str]

        shippori = SemanticWord("shippori")
        shippori.has_mood = ["大人", "しっとり"]
        shippori.has_scene = ["夜", "酒", "静けさ"]
        shippori.has_theme = ["夜の酒"]

        nikkori = SemanticWord("nikkori")
        nikkori.has_mood = ["笑顔", "楽しさ"]
        nikkori.has_scene = ["出会い", "笑顔"]
        nikkori.has_theme = ["楽しい時間"]
    return onto


def semantic_score(word: str, context: dict[str, Any], ontology) -> tuple[float, list[str]]:
    """Score a candidate against SEM concepts stored in Owlready2."""
    lookup = {"しっぽり": "shippori", "にっこり": "nikkori"}
    individual = ontology.search_one(iri=f"*#{lookup.get(word, word)}")
    if individual is None:
        return 0.5, []

    requested = set(context.get("mood", [])) | set(context.get("scene", []))
    known = set(getattr(individual, "has_mood", [])) | set(getattr(individual, "has_scene", []))
    matches = sorted(requested & known)
    score = 0.5 if not requested else min(1.0, 0.5 + 0.45 * len(matches) / len(requested))
    return score, matches


def rhyme_candidates(state: PoemState) -> PoemState:
    """Call rhyme-finder when configured; keep the fixture for offline PoC runs."""
    base_url = os.getenv("RHYME_FINDER_URL")
    if not base_url:
        return {"candidates": _fixture_candidates(state["seed"]),
                "trace": {"rhyme_finder": "fixture"}}

    url = f"{base_url.rstrip('/')}/v1/rhyme"
    payload = json.dumps({"text": state["seed"], "mode": "aggressive", "limit": 20}).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        body = json.load(response)

    candidates = []
    for item in body.get("candidates", []):
        word = item.get("text")
        if not word:
            continue
        candidates.append({
            "word": word,
            "reading": item.get("reading", word),
            "phonetic_score": float(item.get("phonetic", item.get("score", 0.0))),
            "rhythm_score": float(item.get("rhythm", 0.0)),
            "position_score": float(item.get("position", 0.0)),
        })
    return {"candidates": candidates, "trace": {"rhyme_finder": url}}


def _fixture_candidates(seed: str) -> list[dict[str, Any]]:
    fixtures = {
        "日暮里": [
            {"word": "しっぽり", "reading": "しっぽり", "phonetic_score": 0.92},
            {"word": "にっこり", "reading": "にっこり", "phonetic_score": 0.88},
        ]
    }
    return fixtures.get(seed, [])


def score_candidates(state: PoemState) -> PoemState:
    context = state.get("context", {})
    ontology = build_semantic_ontology()
    ranked = []
    for candidate in state.get("candidates", []):
        semantic, semantic_matches = semantic_score(candidate["word"], context, ontology)
        vector = {
            "phonetic": candidate["phonetic_score"],
            "semantic": semantic,
            "context": semantic,
            "rhythm": candidate.get("rhythm_score", 0.90),
            "novelty": 0.80,
        }
        ranked.append({
            **candidate,
            "score_vector": vector,
            "semantic_matches": semantic_matches,
            "score": round(sum(vector.values()) / len(vector), 4),
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {"ranked": ranked, "trace": {**state.get("trace", {}), "semantics": "owlready2"}}


def llm_delegation(state: PoemState) -> PoemState:
    """Only criteria configured as ``llm`` are delegated to LangChain."""
    delegated = [key for key, value in CRITERIA.items() if value == "llm"]
    return {"trace": {**state.get("trace", {}), "delegated_criteria": delegated}}


def build_workflow():
    graph = StateGraph(PoemState)
    graph.add_node("rhyme_finder", rhyme_candidates)
    graph.add_node("score", score_candidates)
    graph.add_node("llm", llm_delegation)
    graph.add_edge(START, "rhyme_finder")
    graph.add_edge("rhyme_finder", "score")
    graph.add_edge("score", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


def save_result(result: PoemState, path: str = RESULT_FILE) -> None:
    """Save the complete PoC result as UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    try:
        result = build_workflow().invoke({
            "seed": "日暮里",
            "context": {"scene": ["夜", "酒", "静けさ"], "mood": ["大人", "しっとり"]},
        })
        save_result(result)
    except (OSError, URLError, TimeoutError, ValueError, KeyError) as exc:
        print(f"エラー：PoCの実行に失敗しました。{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"エラー：予期しない問題が発生しました。{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
