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
