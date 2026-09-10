# Python PoC

`make-poem` の **Semantics × Phonetics** を、LangGraph で最小実装する PoC。

## Flow

```text
seed
  ↓
phonetics
  ↓
semantics + context
  ↓
poetic ranking
  ↓
compose
  ↓
poem
```

LangSmith は LangChain の tracing を利用する。API key がなくてもローカル PoC は実行できる。

## Run

```bash
cd poc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## LangSmith

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=make-poem-poc
```

PowerShell:

```powershell
$env:LANGSMITH_TRACING="true"
$env:LANGSMITH_API_KEY="..."
$env:LANGSMITH_PROJECT="make-poem-poc"
```

## Canonical example

```text
日暮里
 ├─ しっぽり
 └─ にっこり
```

`夜 + 酒 + 静けさ` の context では `しっぽり` を上位にする。

## Next

- `phonetics` → `bonsai/rhyme-finder` contract
- semantic/context → richer graph/context retrieval
- LangGraph → parallel candidate generation and ranking
- LangSmith → trace/evaluation dataset
- LLM composition → final lyric generation
