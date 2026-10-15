from pathlib import Path

import pytest

from rail_rag.ingest import Section, load_manifest, parse_xhtml

FIXTURE = Path(__file__).parent / "fixtures" / "mini_tsi.xhtml"


@pytest.fixture(scope="module")
def sections() -> list[Section]:
    return parse_xhtml(FIXTURE.read_text(encoding="utf-8"), doc_id="mini", celex="0X")


def by_id(sections: list[Section]) -> dict[str, Section]:
    return {s.section: s for s in sections}


def test_section_ids_in_document_order(sections):
    assert [s.section for s in sections] == [
        "Article 1",
        "Article 2",
        "Article 3",
        "Annex §1",
        "Annex §4.2.1",
        "Annex §4.2.3.a",
        "Appendix A",
    ]


def test_cover_page_toc_footnotes_and_amendment_markers_are_dropped(sections):
    text = "\n".join(s.text for s in sections)
    for junk in ["COMMISSION REGULATION", "Table of contents", "OJ L 1", "►", "◄", "▼", "M1"]:
        assert junk not in text


def test_article_heading_combines_number_and_title(sections):
    art1 = by_id(sections)["Article 1"]
    assert art1.heading == "Article 1 — Subject matter"
    assert art1.text == (
        "This Regulation establishes the technical specification for interoperability."
    )


def test_list_items_keep_their_label(sections):
    assert by_id(sections)["Article 2"].text == "(a) new units;"


def test_numbered_paragraphs_in_divs_are_kept_with_nested_lists(sections):
    # Real EUR-Lex markup puts numbered paragraphs in <div class="norm">, not <p>.
    assert by_id(sections)["Article 3"].text == (
        "1. This Article applies.\n2. Member States shall: (a) notify; (b) publish."
    )


def test_deleted_provision_placeholders_are_dropped(sections):
    assert "—" not in "\n".join(s.text for s in sections)


def test_whitespace_is_normalised(sections):
    assert by_id(sections)["Annex §1"].text == "Intro text."


def test_heading_only_sections_are_skipped(sections):
    assert "Annex §4.2" not in by_id(sections)


def test_tables_become_pipe_separated_rows(sections):
    s = by_id(sections)["Annex §4.2.1"]
    assert s.heading == "4.2.1. Limits for stationary noise"
    assert s.text == "Table 2\nCategory | LpAeq [dB]\nElectric locomotives | 70"


def test_appendix_label_drops_footnote_ref_and_uses_subtitle(sections):
    app = by_id(sections)["Appendix A"]
    assert app.heading == "Appendix A — Standards referred to"
    assert app.text == "EN 1234."


def test_repeated_section_ids_are_made_unique():
    xhtml = (
        '<html><body><p class="title-article-norm">Article 1</p><p class="norm">x</p>'
        '<p class="title-annex-1">ANNEX</p>'
        '<p class="title-gr-seq-level-1">P case</p><p class="norm">a</p>'
        '<p class="title-gr-seq-level-1">P case</p><p class="norm">b</p></body></html>'
    )
    ids = [s.section for s in parse_xhtml(xhtml, "d", "c")]
    assert ids == ["Article 1", "Annex §P case", "Annex §P case (2)"]


def test_records_carry_document_metadata(sections):
    assert {(s.doc_id, s.celex) for s in sections} == {("mini", "0X")}


def test_manifest_lists_seven_documents_with_unique_ids():
    docs = load_manifest()
    assert len(docs) == 7
    assert len({d.id for d in docs}) == 7
    assert all(d.celex.startswith("0") for d in docs)
