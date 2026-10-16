# Lessons learned

Things that went wrong while building this project and how they were fixed.

## 1. CI failed before running a single step

**Symptom:** the first `ci.yml` run failed in "Set up job" with ``Unable to resolve action `astral-sh/setup-uv@v10` ``.

**Cause:** I assumed every action publishes a floating major tag (`v10`) like `actions/checkout@v7` does. `astral-sh/setup-uv` only publishes full release tags (`v10.3.0`).

**Fix:** pin the exact tag. Checked with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` before pushing again.

**Takeaway:** verify that action refs exist rather than guessing; exact pins are also better for reproducibility.

## 2. The parser silently dropped 47% of the Interoperability Directive

**Symptom:** after the first ingestion run, the directive had 12,018 words of parsed text against about 25,000 words in the raw XHTML. All the unit tests passed.

**Cause:** the hand-written test fixture assumed that all body text sits in `<p>` elements. In the real EUR-Lex markup, numbered paragraphs are `<div class="norm"><span class="no-parag">1.</span><div class="inline-element">text…</div></div>`, with the text directly inside a `div`. List labels such as `(a)` were bare `<span>`s. The walker only read `<p>`, so it skipped all of them.

**Fix:**
- Rewrote the fixture from real markup.
- Added a block-aware text extractor that keeps numbered paragraphs and nested lists together.
- Added a coverage check that compares the parsed word count with the body word count of each source (minus the TOC, footnotes and cover page). It now reads ~100% for the directive and 103–107% for the TSIs; the overshoot comes from table separators and heading dashes being counted as words.

**Takeaway:** tests built on a fixture you invented only prove the code matches your assumptions. Measure the output against the real input before trusting it.

## 3. A similarity threshold alone cannot decide when to refuse

**Observation:** I probed the local index (bge-small, cosine) with 5 in-scope and 3 out-of-scope questions before designing the refusal path.

- In-scope questions had top-1 scores of 0.73–0.84.
- Unrelated questions ("capital of France", "ticket price Lisbon–Porto") scored 0.50–0.54.
- "What are the GDPR rules for passenger data?" scored **0.715**. That question is rail-adjacent, but the answer isn't in the corpus, and the score falls inside the in-scope range.

**Decision:** use two gates.
1. A cheap cosine floor rejects clearly unrelated questions without calling an LLM.
2. An LLM relevance grade on the retrieved passages handles near-domain questions.

The golden set includes near-domain unanswerable questions, to test the second gate.

## 4. RAGAS failed at import, and the token budget shaped the metrics

**Symptom:** `ragas==0.4.3` (the latest release) raised `ModuleNotFoundError: langchain_community.chat_models.vertexai` on import. uv had resolved `langchain-community` 0.4.2, which no longer ships that module.

**Fix:** pin `langchain-community<0.4` in the `eval` dependency group only. LangGraph keeps `langchain-core` 1.x and the app image never installs RAGAS. Time-boxed at two attempts; the fallback was to implement the metrics directly.

**Second constraint: the free-tier token budget.** Run with an LLM, RAGAS context precision makes one judge call per retrieved passage. With 5 passages × 16 answerable questions, plus faithfulness and relevancy, that's roughly 150 calls and ~300k tokens. That's more than the daily budget Groq publishes for `gpt-oss-120b` (200k tokens/day on the Developer plan).

**Decision:** the golden set already records the gold passage ids for every question. So context precision (same rank-weighted formula as RAGAS) and context recall are computed from ids: deterministically and for free. The judge is spent only on what needs judgement: faithfulness and answer relevancy, at ~3 calls per question. Judge replies are disk-cached, so re-scoring the same answers with a second judge, or re-running unchanged answers, costs nothing.

## 5. Azure: a region policy, a Standard-tier default, and a lagging document count

- **The default tier costs money.** The AI Search create form defaults to **Standard**, which is billed per hour. The Free tier (50 MB, 3 indexes) has to be chosen explicitly. Always check the "Estimated cost per month: $0.00" line on the review page.
- **`RequestDisallowedByAzure`.** Azure for Students subscriptions carry a policy, *Allowed resource deployment regions*, so the portal's default (Central US) and West Europe were both rejected. The allowed list is under Policy → Assignments → Parameters. Switzerland North was allowed and offers the Free tier.
- **The document count lags.** Straight after uploading 1,589 passages, `get_document_count()` returned 1,500; 20 s later it returned 1,589. Azure AI Search indexes in near real time, so counts lag. Separately, `merge_or_upload_documents` reports per-document failures in its return value **without raising**, and the first version ignored that. `upsert` now raises if any document is rejected.
- **Score parity checked live.** For the 16 answerable golden questions, Azure and the local Qdrant index returned the same top-5 passages (16/16 identical top-1) with identical cosine scores (max difference 0.0000). That confirms the `cos = 2 − 1/score` conversion from Azure's reported score.

## 6. First real eval: the grader drifted in shape and the judge ran out of tokens

The first full run (20 questions, Gemini generating, `gpt-oss-120b` judging) exposed two bugs that the mocked tests could not.

**a. False refusals caused by JSON shape drift.** 4 of the 16 answerable questions were refused with `no_relevant_passages`. For q05, q10 and q12 the gold passages were in the top 3, so retrieval was fine. Replaying the grader call showed Gemini, at temperature 0, sometimes returns `{"relevant": [[1], [2], [3]]}` instead of `[1, 2, 3]`. The same prompt gave the correct shape on the next call. The grader node accepted only plain integers, so a nested reply meant "nothing relevant".

*Fix, in two layers:*
1. Both providers now enforce a JSON schema (Gemini `response_json_schema`, Groq `json_schema`), so the reply has to be an integer array.
2. The node also normalises nested lists and numeric strings, as a defence.

*Takeaway:* `json_mode` only guarantees valid JSON, not the right structure. With a scripted fake LLM the bug could never appear: fakes return what you expect, and real models don't always.

**b. The judge crashed on truncated output.** RAGAS sets `max_tokens=1024` by default. `gpt-oss-120b` is a reasoning model and spends part of that budget on hidden reasoning, so a faithfulness verdict was cut off mid-JSON (`IncompleteOutputException`) and the run aborted at q11. *Fix:* `max_tokens=4096` and `reasoning_effort="low"`. Low effort also keeps each call small under Groq's free-tier limit of 8K tokens/minute.

**c. One genuine retrieval miss (q13).** The gold passage (Directive Art. 7(7): "four months … deemed to have been accepted") is not in the top 8 for its question. Dense retrieval ranked other "Commission decision" articles from the TSIs higher. This one is a retrieval limitation, not a bug; see Limitations.

**d. A judge false negative, kept as measured.** In the passing run, q09 ("Class A train protection and radio systems") scored faithfulness **0.50**, but the answer restates its single passage almost word for word. Inspecting the judge's intermediate output showed the cause:
- RAGAS split "RMR comprises two Class A radio systems: GSM-R and FRMCS" into two claims: "…comprises two Class A radio systems: GSM-R" and "…: FRMCS".
- `gpt-oss-120b` then rejected each claim as "incomplete, so it cannot be directly inferred".
- The other two claims were accepted, so the score was 2/4.

The error comes from statement decomposition, not from the generator. I kept the score as measured rather than overriding it. This is why the README reports judge scores with a second judge and treats them as noisy estimates, not ground truth.

## 7. The self-preference test came back null, and the setup is why

**Setup:** the 15 answered items from CI run 38831471610, generated by `gemini-3.5-flash-lite`, were judged twice on identical answers:
- by `gpt-oss-120b` (a different family, which is the gating judge);
- by `gemini-3.5-flash-lite` itself (the same model, so the strongest self-preference condition).

**Result:**
- Faithfulness was 1.00 vs 1.00 on every item.
- Answer relevancy was 0.90 vs 0.91 (Δ +0.01). Gemini scored higher on 7 items, lower on 7, and the largest per-item gap was 0.09.

**What this does *not* show:** that self-preference is absent. Three things make this setup unable to detect it:
1. **Ceiling effect.** Every answer was fully supported, so there was nothing for a biased judge to forgive.
2. **Small sample.** n = 15.
3. **Embedding-dominated metric.** RAGAS answer relevancy uses the LLM only to generate questions back from the answer. The score is the cosine similarity of those questions to the original question, computed with the same local embedding model for both judges.

**What would:** a set of answers with planted unsupported claims (perturbations), judged by both models, measuring how often each judge *misses* the planted errors. Self-preference shows up as leniency on errors, not as a shift on correct answers. That's listed as a next step.

**Why the gate still uses a different family:** the cost is zero, and the literature (and our CAPSI 2026 paper) shows the bias exists in less saturated settings. Not detecting it here is not a reason to drop the safeguard.

## 8. The free hosting assumption broke at deploy time

The plan assumed Hugging Face Spaces would host a Docker app on free CPU. At deploy time `create_repo(space_sdk="docker")` returned **402 Payment Required**: "hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription". Only static Spaces are still free.

Two things limited the damage:
1. **The image was already proven elsewhere.** A GitHub Actions job builds it and smoke-tests it (non-root user, `/health`, UI, Docker `HEALTHCHECK`) before any hosting step, so changing the host doesn't re-open "does the container work?".
2. **The deploy script failed before sending anything.** Space creation runs before the step that copies secrets, so no keys left the machine.

*Takeaway:* check a free tier's current terms at the start of a project, not at deploy time. Pricing pages change faster than tutorials.

## 9. Swapping torch for ONNX: exact parity, one silent trap

The 2.99 GB image was mostly torch, which only ran a 33M-parameter embedding model. Moving to `fastembed` (ONNX Runtime, the same `bge-small-en-v1.5` weights) removed torch, sentence-transformers and transformers from the runtime.

**Parity, measured rather than assumed:**
- cosine(old, new) = 1.0000 (minimum) on 200 passages and all 20 golden questions;
- re-indexing Qdrant and Azure gave identical top-8 rankings for all 20 CI eval queries, with a maximum score difference of 0.0005.

**The trap:** `TextEmbedding.query_embed()` does **not** add bge's query instruction, which dropped query similarity to 0.956. That's enough to reorder results quietly without failing anything. The embedder prepends the instruction itself, and the parity check above is what caught it.

**Cost of the swap:** indexing all 1,589 passages on a laptop CPU went from 110 s to 191 s. It's a one-off at image build time; query embedding stays in milliseconds.
