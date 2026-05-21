# RAG Evaluation

## Three Evaluation Dimensions

| Dimension | Question | Why It Matters |
|-----------|----------|----------------|
| **Answer Accuracy** | Is the final answer correct and complete? | Primary user-facing metric |
| **LLM Utilization Quality** | Did the model correctly understand and use the retrieved docs? | Bridge between retrieval and generation |
| **Retrieval Effectiveness** | Were all relevant documents found? Any misses? | Bottleneck locator |

Evaluating only the final answer masks retrieval problems. All three dimensions are necessary.

---

## Retrieval Metrics (Venn Diagram Intuition)

- **A** = all documents in the DB that are truly relevant to the query
- **B** = all documents the system actually retrieved
- **C** = the intersection: retrieved AND relevant

### Recall
$$\text{Recall} = \frac{C}{A} = \frac{\text{retrieved relevant}}{\text{all relevant}}$$

Measures **coverage**: did we miss any critical information?

### Precision
$$\text{Precision} = \frac{C}{B} = \frac{\text{retrieved relevant}}{\text{all retrieved}}$$

Measures **accuracy**: how much of what we retrieved is actually useful?

### F1 Score
$$F1 = \frac{2 \times \text{Recall} \times \text{Precision}}{\text{Recall} + \text{Precision}}$$

Harmonic mean — balances the natural tension between recall and precision.

---

## Precision vs Recall Tradeoff

**Recall-first is the dominant production strategy** for two reasons:
1. Most enterprise knowledge bases have inconsistent quality — better to over-retrieve and let the LLM filter
2. Modern LLMs (Claude Sonnet, GPT-4o) have long context windows and strong noise tolerance

**Precision-first** only makes sense when the knowledge base is curated and high-quality.

---

## Evaluation Methods

| Method | Strength | Weakness |
|--------|----------|----------|
| **Human evaluation** (1–10 scoring) | Gold standard for real UX quality | Slow, expensive, subjective |
| **Automatic evaluation** (BERTScore / Cross-Encoder) | Fast, batch-able, no ground truth needed | Cannot fully replace human judgment |

**Practical approach:** Use automatic evaluation to drive rapid iteration; use human evaluation to validate before launch. Strong models (Claude Sonnet, GPT-4o) can also generate test samples and reference answers to reduce annotation cost.

---

## Evaluation Frameworks

### RAGAS — RAG Pipeline Quality
Reference-free evaluation: no human-annotated ground truth required.

| Metric | Layer | Question |
|--------|-------|----------|
| **Faithfulness** | Generation | Does every claim in the answer have support in the retrieved context? |
| **Answer Relevancy** | Generation | Does the answer actually address the user's question? |
| **Context Precision** | Retrieval | Are relevant chunks ranked near the top? |
| **Context Recall** | Retrieval | Were all relevant documents retrieved? |

**Faithfulness scoring:** LLM decomposes the answer into atomic claims → verifies each against context → score = supported claims / total claims.

**Answer Relevancy scoring:** LLM generates N reverse questions from the answer → cosine similarity with original question → average. No LLM-as-Judge; pure math.

```
pip install ragas
```

---

### TruLens — Full Lifecycle Observability
Core idea: "from vibes to metrics" — turn subjective intuition into trackable numbers.

- Wraps every RAG pipeline step as an **OpenTelemetry span** — no black-box treatment
- Integrates with Jaeger, Grafana, and existing observability infrastructure
- Covers safety & fairness (toxicity, bias, sentiment) in addition to quality metrics

**vs RAGAS:** RAGAS focuses on the 4 RAG-specific dimensions; TruLens covers the full LLM application lifecycle including safety, ethics, and trace-level regression comparison.

---

### ARES (IBM) — Adversarial Robustness Testing
Not a quality evaluator — an **attack-centric red-teaming framework**.

Three-layer pluggable architecture:
```
Goals → Strategy → Evaluation
```
- **Goals:** PII leakage, prompt injection, harmful content generation
- **Strategy:** jailbreak, encoding obfuscation, multi-turn manipulation, gradient-based adversarial examples
- **Evaluation:** keyword matching, LLM-as-Judge, custom detectors

19+ built-in plugins. Maps directly to OWASP LLM Top 10.

---

## Framework Selection Guide

| Scenario | Use | Reason |
|----------|-----|--------|
| RAG development & debugging | **RAGAS** | No ground truth needed; 4 metrics map directly to pipeline layers |
| Full-lifecycle quality + compliance | **TruLens** | OpenTelemetry traces enable precise regression localization |
| Pre-launch security audit | **ARES** | Adversarial testing, OWASP vulnerability scanning, pluggable attack strategies |
