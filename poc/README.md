# make-poem Python PoC

`make-poem` は、**Semantics × Phonetics から「なぜこの言葉を選んだか」が説明できる詩語選択と詩生成**を行う研究実装です。

## 成果物

このPoCでいう成果物は、単に「詩を1本生成すること」ではありません。

**1回の実行で、次の4つを残せること**を成果物とします。

```text
入力
  ↓
① 候補語
  ↓
② SEM × PHO の評価
  ↓
③ 選択結果と理由
  ↓
④ 最終Poem
```

### ① 候補語（Candidates）

Seedに対して、音が近く詩に使えそうな言葉を複数取得します。

例：

```text
日暮里
 ├─ しっぽり
 └─ にっこり
```

候補生成は `bonsai/rhyme-finder` を担当し、make-poem側で音韻エンジンを再実装しません。

### ② 評価（SEM × PHO Matrix）

候補ごとに、**意味（SEM）と音（PHO）を別々に評価**します。

```text
candidate: しっぽり

PHO
  phonetic: 0.92
  rhythm:   0.90

SEM
  semantic: 0.95
  context:  0.95
  mood:     0.95

→ combined score
```

重み・制約・評価方法はデータとして変更できることを目標にします。

### ③ 選択結果と理由（Selection）

「LLMがなんとなく選んだ」ではなく、**なぜその語が選ばれたのかを観測できること**を成果物にします。

例：

```text
selected: しっぽり
score: 0.904
reason:
  - 夜に適合
  - 酒に適合
  - 静けさに適合
  - 大人・しっとりのmoodに適合
  - 音韻スコアも高い
```

### ④ Poem（最終生成物）

選択された詩語、SEM、PHO、contextを材料として、LLM等に詩を生成させます。

LLMの役割は**候補選択を丸投げすることではなく、指定された評価項目または最終的な詩の表現を担当すること**です。

---

## 成果物の最小セット

| 成果物 | 内容 | 主担当 |
|---|---|---|
| Candidates | 音韻候補一覧 | rhyme-finder |
| Score Vector | SEM / PHO の評価値 | make-poem |
| Ranking | 候補順位 | make-poem |
| Selection Reason | 選択理由・適合項目 | make-poem |
| Poem | 最終的な詩・歌詞 | LLM / composer |
| Trace | workflow・LLM委譲の記録 | LangGraph / LangSmith |

**最重要成果物は `Score Vector + Ranking + Selection Reason` です。**

Poemだけを出すのではなく、**「どの候補を、どの基準で、なぜ選び、その結果どんな詩になったか」まで再現できること**をPoCの完成条件とします。

## Flow

```text
Seed
  ↓
rhyme-finder
  ↓
PHO candidates
  ↓
SEM / Owlready2 + Context
  ↓
SEM × PHO Matrix
  ↓
Score Vector
  ↓
Ranking
  ↓
Selection + Reason
  ↓
LangChain / LLM
  ↓
Poem
  ↓
Trace / Evaluation
```

## Canonical example

```text
Seed: 日暮里
Context: 夜 + 酒 + 静けさ
Mood: 大人 + しっとり

候補
  しっぽり  PHO 0.92
  にっこり  PHO 0.88

結果
  しっぽり → selected
```

同じ音韻候補でもcontextが変われば結果が変わることが、このPoCの重要な性質です。

## Run

```bash
cd poc
chmod +x run.sh
./run.sh
```

`RHYME_FINDER_URL` を設定すると実際の `rhyme-finder` APIを使用します。未設定時は再現可能なfixtureで実行します。

```bash
export RHYME_FINDER_URL=http://localhost:8080
./run.sh
```

## LangSmith

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=make-poem-poc
```

## 完成条件

- [x] 音韻候補を複数持てる
- [x] SEMとPHOを独立した軸として観測できる
- [x] contextによって順位が変わる
- [x] Owlready2でSEMを扱える
- [x] LangGraphでworkflowを実行できる
- [x] LLM委譲対象を限定できる
- [x] rhyme-finder APIとの接続境界がある
- [ ] Score Vector / Ranking / Selection Reasonを安定したJSON成果物として出力する
- [ ] LangSmith評価データセットを追加する
- [ ] 最終PoemをLLMで生成する
