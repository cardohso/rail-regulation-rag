## Evaluation PASSED

Generator `gemini:gemini-3.5-flash-lite` · judge `groq:openai/gpt-oss-120b` · 20 golden items · commit `2c6ba60` · 2026-10-16 00:58 UTC · [CI run](https://github.com/cardohso/rail-regulation-rag/actions/runs/38831471610)

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Faithfulness (judge) | 1.00 | ≥ 0.80 | ✅ |
| Answer relevancy (judge) | 0.90 | ≥ 0.70 | ✅ |
| Context precision (gold ids) | 0.94 | ≥ 0.50 | ✅ |
| Context recall (gold ids) | 0.89 | ≥ 0.70 | ✅ |
| Grounding: cites only expected docs | 1.00 | ≥ 0.90 | ✅ |
| Answerable questions answered | 0.94 | ≥ 0.80 | ✅ |
| Unanswerable questions refused | 1.00 | ≥ 0.75 | ✅ |
| Retrieval recall@k, before grading (diagnostic) | 0.91 | — | |
| Grounding on wrong-document items (diagnostic) | 1.00 | — | |

### Grounding per source document

| Document | Questions | Answered | Grounding | Also cited |
|---|---|---|---|---|
| `iop-directive` | 3 | 2 | 1.00 | — |
| `tsi-ccs` | 1 | 1 | 1.00 | — |
| `tsi-ene` | 1 | 1 | 1.00 | — |
| `tsi-noi` | 2 | 2 | 1.00 | — |
| `tsi-ope` | 3 | 3 | 1.00 | — |
| `tsi-prm` | 2 | 2 | 1.00 | — |
| `tsi-srt` | 4 | 4 | 1.00 | — |

<details><summary>Per-item scores</summary>

| Item | Type | Refused | Faithfulness | Relevancy | Ctx P | Ctx R | Grounded |
|---|---|---|---|---|---|---|---|
| q01 | answerable | no | 1.00 | 0.90 | 1.00 | 1.00 | yes |
| q02 | answerable | no | 1.00 | 0.95 | 1.00 | 1.00 | yes |
| q03 | answerable | no | 1.00 | 0.79 | 1.00 | 1.00 | yes |
| q04 | answerable | no | 1.00 | 0.81 | 1.00 | 1.00 | yes |
| q05 | answerable | no | 1.00 | 0.87 | 1.00 | 1.00 | yes |
| q06 | answerable | no | 1.00 | 0.86 | 1.00 | 1.00 | yes |
| q07 | answerable | no | 1.00 | 0.93 | 1.00 | 1.00 | yes |
| q08 | answerable | no | 1.00 | 0.78 | 1.00 | 0.50 | yes |
| q09 | answerable | no | 1.00 | 0.91 | 1.00 | 1.00 | yes |
| q10 | answerable | no | 1.00 | 0.95 | 1.00 | 0.67 | yes |
| q11 | answerable | no | 1.00 | 0.99 | 1.00 | 1.00 | yes |
| q12 | answerable | no | 1.00 | 0.85 | 1.00 | 1.00 | yes |
| q13 | answerable | yes | — | — | 0.00 | 0.00 | — |
| q14 | wrong_document | no | 1.00 | 0.93 | 1.00 | 1.00 | yes |
| q15 | wrong_document | no | 1.00 | 0.99 | 1.00 | 1.00 | yes |
| q16 | wrong_document | no | 1.00 | 0.98 | 1.00 | 1.00 | yes |
| q17 | unanswerable | yes | — | — | — | — | — |
| q18 | unanswerable | yes | — | — | — | — | — |
| q19 | unanswerable | yes | — | — | — | — | — |
| q20 | unanswerable | yes | — | — | — | — | — |

</details>
