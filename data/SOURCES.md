# Data sources and licences

All documents are EU legal acts in English, published by the Publications Office of the European Union on EUR-Lex. The corpus is 7 consolidated texts: the Interoperability Directive and 6 TSIs. That is about 175,000 words of regulatory text after parsing.

## Documents

| id | Act | Consolidated version | EUR-Lex page |
|---|---|---|---|
| `iop-directive` | Directive (EU) 2016/797, interoperability of the rail system | 02.08.2026 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02016L0797-20260802> |
| `tsi-ccs` | Implementing Regulation (EU) 2023/1695, control-command and signalling TSI | 05.05.2026 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02023R1695-20260505> |
| `tsi-prm` | Regulation (EU) No 1300/2014, persons with reduced mobility TSI | 28.09.2023 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02014R1300-20230928> |
| `tsi-ene` | Regulation (EU) No 1301/2014, energy TSI | 28.09.2023 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02014R1301-20230928> |
| `tsi-srt` | Regulation (EU) No 1303/2014, safety in railway tunnels TSI | 29.01.2024 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02014R1303-20240129> |
| `tsi-noi` | Regulation (EU) No 1304/2014, rolling stock noise TSI | 28.09.2023 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02014R1304-20230928> |
| `tsi-ope` | Implementing Regulation (EU) 2019/773, operation and traffic management TSI | 31.03.2025 | <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02019R0773-20250331> |

Each version was the latest consolidated version listed in the Cellar SPARQL endpoint on 2026-10-15.

## How the files are obtained

`src/rail_rag/ingest.py` downloads each document's English XHTML from the Publications Office **Cellar** API, which is the official machine-readable access route to EUR-Lex content. It uses content negotiation on `http://publications.europa.eu/resource/celex/<CELEX>`. The script sends at most one request per second, with a descriptive User-Agent.

- `https://publications.europa.eu/robots.txt` allows `/` for all user agents (checked 2026-10-15).
- `eur-lex.europa.eu` serves an automated-client challenge to scripts, so it is **not** scraped.
- Raw downloads go to `data/raw/`, which is gitignored and rebuilt by the script.
- Parsed sections go to `data/processed/*.jsonl`, which is committed so the app and CI need no network access to build.

```bash
uv run python -m rail_rag.ingest            # download missing files, then parse
uv run python -m rail_rag.ingest --offline  # re-parse files already in data/raw
```

## Licence and reuse terms

- **EUR-Lex legal notice** (<https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>, read 2026-10-15): "© European Union, 1998-2026". The reuse policy is based on Decision 2011/833/EU, and unless otherwise specified the legal documents published in EUR-Lex can be reused for commercial or non-commercial purposes.
- **Commission Decision 2011/833/EU** on the reuse of Commission documents (<https://eur-lex.europa.eu/eli/dec/2011/833/oj>):
  - Article 4: documents are available for reuse without charge and without an individual application.
  - Article 6: conditions may include acknowledging the source, not distorting the original meaning, and non-liability of the Commission.
- None of the seven acts carries a special condition of use.

**How this project complies:**
- Every answer cites the source document and section.
- This file acknowledges the European Union as the source.
- Passages are quoted verbatim, with no rewording.
- The README states that the assistant is not legal advice.

**Attribution:** Source: EUR-Lex, © European Union, 1998–2026. Text processed (split into passages; figures and footnotes removed) by this project.

## Caveats

- **Consolidated texts have no legal effect.** Each one states that it is "meant purely as a documentation tool and has no legal effect". Only the Official Journal versions are authentic.
- **Standards are not included.** The TSIs reference EN standards and ERA technical documents, which are copyrighted or published separately. The assistant can't answer questions whose answer is only in a referenced standard.
- **Figures and diagrams are dropped** during parsing, because they are embedded images.
- **Two TSIs were left out for size:** infrastructure (Regulation (EU) No 1299/2014, about 35k words) and locomotives and passenger rolling stock (Regulation (EU) No 1302/2014, about 81k words).
