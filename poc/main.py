"""LangChain + LangGraph PoC for Semantics × Phonetics lyric generation."""

import os
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class PoemState(TypedDict, total=False):
    seed: str
    context: dict
    candidates: list[dict]
    ranked: list[dict]
    poem: str


PHONETIC_FIXTURES = {
    "日暮里": [
        {"word": "しっぽり", "reading": "しっぽり", "phonetic_score": 0.92},
        {"word": "にっこり", "reading": "にっこり", "phonetic_score": 0.88},
    ]
}


def phonetics(state: PoemState) -> PoemState:
    return {"candidates": PHONETIC_FIXTURES.get(state["seed"], [])}


def semantics(state: PoemState) -> PoemState:
    context = state.get("context", {})
    mood = set(context.get("mood", []))
    scene = set(context.get("scene", []))
    ranked = []
    for candidate in state.get("candidates", []):
        word = candidate["word"]
        semantic_score = 0.5
        context_score = 0.5
        if word == "しっぽり":
            semantic_score = 0.95 if {"大人", "しっとり"} & mood else 0.65
            context_score = 0.95 if {"夜", "酒", "静けさ"} & scene else 0.55
        elif word == "にっこり":
            semantic_score = 0.95 if {"笑顔", "楽しさ"} & mood else 0.65
            context_score = 0.95 if {"出会い", "笑顔"} & scene else 0.55
        poetic_score = (
            candidate["phonetic_score"] * 0.4
            + semantic_score * 0.3
            + context_score * 0.3
        )
        ranked.append({**candidate, "semantic_score": semantic_score,
                       "context_score": context_score,
                       "poetic_score": round(poetic_score, 4)})
    ranked.sort(key=lambda x: x["poetic_score"], reverse=True)
    return {"ranked": ranked}


def compose(state: PoemState) -> PoemState:
    """Use LangChain for final lyric composition when a model is configured."""
    top = state["ranked"][0]
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LM_STUDIO_API_KEY"):
        return {"poem": f"{state['seed']}、{top['word']}。\n音が似ているだけじゃない、\nいまの景色に似合う言葉。"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは日本語の作詞家。音韻と意味の交差から短い歌詞を作る。説明せず詩だけを書く。"),
        ("human", "種語: {seed}\n文脈: {context}\n選択語: {word}\n音韻: {phonetic}\n意味: {semantic}\n文脈適合: {context_score}"),
    ])
    model = ChatOpenAI(
        model=os.getenv("POEM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LM_STUDIO_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LM_STUDIO_API_KEY") or "lm-studio",
        temperature=0.8,
    )
    chain = prompt | model
    response = chain.invoke({
        "seed": state["seed"],
        "context": state.get("context", {}),
        "word": top["word"],
        "phonetic": top["phonetic_score"],
        "semantic": top["semantic_score"],
        "context_score": top["context_score"],
    })
    return {"poem": response.content}


def build_graph():
    graph = StateGraph(PoemState)
    graph.add_node("phonetics", phonetics)
    graph.add_node("semantics", semantics)
    graph.add_node("compose", compose)
    graph.add_edge(START, "phonetics")
    graph.add_edge("phonetics", "semantics")
    graph.add_edge("semantics", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


if __name__ == "__main__":
    result = build_graph().invoke({
        "seed": "日暮里",
        "context": {"scene": ["夜", "酒", "静けさ"], "mood": ["大人", "しっとり"]},
    })
    print("=== ranked candidates ===")
    for candidate in result["ranked"]:
        print(candidate)
    print("\n=== poem ===")
    print(result["poem"])
