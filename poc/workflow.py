"""PoC workflow: rhyme-finder → SEM (Owlready2) → PHO → ranking."""

import html
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


CRITERIA = {"phonetic": "deterministic", "semantic": "deterministic", "context": "rule", "rhythm": "deterministic", "novelty": "deterministic", "style": "llm"}
RESULT_FILE = "result.json"
HTML_FILE = "result.html"


def build_semantic_ontology():
    onto = get_ontology("http://bonsai.local/make-poem-sem.owl")
    with onto:
        class SemanticWord(Thing): pass
        class has_mood(DataProperty): range = [str]
        class has_scene(DataProperty): range = [str]
        class has_theme(DataProperty): range = [str]
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
    lookup = {"しっぽり": "shippori", "にっこり": "nikkori"}
    individual = ontology.search_one(iri=f"*#{lookup.get(word, word)}")
    if individual is None: return 0.5, []
    requested = set(context.get("mood", [])) | set(context.get("scene", []))
    known = set(getattr(individual, "has_mood", [])) | set(getattr(individual, "has_scene", []))
    matches = sorted(requested & known)
    score = 0.5 if not requested else min(1.0, 0.5 + 0.45 * len(matches) / len(requested))
    return score, matches


def rhyme_candidates(state: PoemState) -> PoemState:
    base_url = os.getenv("RHYME_FINDER_URL")
    if not base_url:
        return {"candidates": _fixture_candidates(state["seed"]), "trace": {"rhyme_finder": "fixture"}}
    url = f"{base_url.rstrip('/')}/v1/rhyme"
    payload = json.dumps({"text": state["seed"], "mode": "aggressive", "limit": 20}).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response: body = json.load(response)
    candidates = []
    for item in body.get("candidates", []):
        word = item.get("text")
        if word:
            candidates.append({"word": word, "reading": item.get("reading", word), "phonetic_score": float(item.get("phonetic", item.get("score", 0.0))), "rhythm_score": float(item.get("rhythm", 0.0)), "position_score": float(item.get("position", 0.0))})
    return {"candidates": candidates, "trace": {"rhyme_finder": url}}


def _fixture_candidates(seed: str) -> list[dict[str, Any]]:
    return {"日暮里": [{"word": "しっぽり", "reading": "しっぽり", "phonetic_score": 0.92}, {"word": "にっこり", "reading": "にっこり", "phonetic_score": 0.88}]}.get(seed, [])


def score_candidates(state: PoemState) -> PoemState:
    ontology = build_semantic_ontology()
    ranked = []
    for candidate in state.get("candidates", []):
        semantic, matches = semantic_score(candidate["word"], state.get("context", {}), ontology)
        vector = {"phonetic": candidate["phonetic_score"], "semantic": semantic, "context": semantic, "rhythm": candidate.get("rhythm_score", 0.90), "novelty": 0.80}
        ranked.append({**candidate, "score_vector": vector, "semantic_matches": matches, "score": round(sum(vector.values()) / len(vector), 4)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {"ranked": ranked, "trace": {**state.get("trace", {}), "semantics": "owlready2"}}


def llm_delegation(state: PoemState) -> PoemState:
    return {"trace": {**state.get("trace", {}), "delegated_criteria": [k for k, v in CRITERIA.items() if v == "llm"]}}


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


def save_result(result: PoemState) -> None:
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_html(result: PoemState) -> None:
    ranked = result.get("ranked", [])
    context = result.get("context", {})
    selected = ranked[0] if ranked else None
    rows = []
    for i, c in enumerate(ranked, 1):
        v = c["score_vector"]
        matches = "、".join(c.get("semantic_matches", [])) or "なし"
        reason = "今回の文脈に意味が合う" if c.get("semantic_matches") else "音は近いが、今回の文脈には合わない"
        rows.append(f"<tr><td>{i}</td><td><b>{html.escape(c['word'])}</b></td><td>{c['score']:.3f}</td><td>{v['phonetic']:.2f}</td><td>{v['semantic']:.2f}</td><td>{v['rhythm']:.2f}</td><td>{html.escape(matches)}</td><td>{reason}</td></tr>")
    page = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>make-poem PoC</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.7;background:#fafafa;color:#222}}section{{background:#fff;border:1px solid #ddd;border-radius:12px;padding:20px;margin:16px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#f3f3f3}}.selected{{font-size:1.7em}}</style></head><body><h1>make-poem PoC 結果</h1><section><h2>今回のテーマ</h2><p>種語：<b>{html.escape(result.get('seed',''))}</b></p><p>場面：{html.escape('、'.join(context.get('scene', [])))}</p><p>気分：{html.escape('、'.join(context.get('mood', [])))}</p></section><section><h2>選ばれた詩語</h2><p class="selected">{html.escape(selected['word']) if selected else '候補なし'}</p><p>音の近さだけでなく、意味と文脈を合わせて順位を決めています。</p></section><section><h2>候補の比較</h2><table><tr><th>順位</th><th>候補</th><th>総合</th><th>音</th><th>意味</th><th>リズム</th><th>意味の一致</th><th>理由</th></tr>{''.join(rows)}</table></section><section><h2>仕組み</h2><p><b>rhyme-finder</b> が音の候補を作り、<b>SEM × PHO</b> で評価して選択します。LLMには必要な役割だけを委譲します。</p></section></body></html>'''
    with open(HTML_FILE, "w", encoding="utf-8") as f: f.write(page)


if __name__ == "__main__":
    try:
        seed = sys.argv[1] if len(sys.argv) > 1 else "日暮里"
        result = build_workflow().invoke({"seed": seed, "context": {"scene": ["夜", "酒", "静けさ"], "mood": ["大人", "しっとり"]}})
        save_result(result)
        save_html(result)
    except (OSError, URLError, TimeoutError, ValueError, KeyError) as exc:
        print(f"エラー：PoCの実行に失敗しました。{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"エラー：予期しない問題が発生しました。{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
