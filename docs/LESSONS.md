# Lessons learned

Things that went wrong while building this project and how they were fixed.

## 1. CI failed before running a single step

**Symptom:** the first `ci.yml` run failed in "Set up job" with ``Unable to resolve action `astral-sh/setup-uv@v10` ``.

**Cause:** I assumed every action publishes a floating major tag (`v10`) like `actions/checkout@v7` does. `astral-sh/setup-uv` only publishes full release tags (`v10.3.0`).

**Fix:** pin the exact tag. Checked with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` before pushing again.

**Takeaway:** verify that action refs exist rather than guessing; exact pins are also better for reproducibility.
