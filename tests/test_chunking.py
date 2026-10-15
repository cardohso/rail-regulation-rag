from rail_rag.chunking import Passage, chunk_section, load_passages


def section(text: str) -> dict:
    return {
        "doc_id": "tsi-noi",
        "section": "Annex §4.2.1",
        "heading": "4.2.1. Limits",
        "text": text,
    }


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_short_section_is_one_passage_with_stable_id():
    [p] = chunk_section(section("Limit is 70 dB."), doc_short="NOI TSI")
    assert p.id == "tsi-noi:Annex §4.2.1#1"
    assert p.text == "Limit is 70 dB."
    assert (p.doc_id, p.section, p.heading) == ("tsi-noi", "Annex §4.2.1", "4.2.1. Limits")


def test_long_section_packs_whole_paragraphs_under_the_limit():
    paras = [words(100, 0), words(100, 100), words(100, 200)]
    ps = chunk_section(section("\n".join(paras)), doc_short="NOI TSI", max_words=250)
    assert [p.text for p in ps] == ["\n".join(paras[:2]), paras[2]]
    assert [p.id[-2:] for p in ps] == ["#1", "#2"]


def test_oversized_paragraph_is_split_into_overlapping_windows():
    ps = chunk_section(section(words(600)), doc_short="NOI TSI", max_words=250, overlap=50)
    assert all(len(p.text.split()) <= 250 for p in ps)
    assert ps[0].text.split()[-50:] == ps[1].text.split()[:50]
    covered = {w for p in ps for w in p.text.split()}
    assert covered == set(words(600).split())


def test_embed_text_gives_document_and_heading_context():
    [p] = chunk_section(section("Limit is 70 dB."), doc_short="NOI TSI")
    assert p.embed_text == "NOI TSI, 4.2.1. Limits\nLimit is 70 dB."


def test_citation_label_is_human_readable():
    [p] = chunk_section(section("x"), doc_short="NOI TSI")
    assert p.citation == "NOI TSI, Annex §4.2.1"


def test_passage_round_trips_through_dict():
    [p] = chunk_section(section("x"), doc_short="NOI TSI")
    assert Passage.from_dict(p.to_dict()) == p


def test_real_corpus_passages_have_unique_ids_and_respect_limit():
    ps = load_passages()
    assert len(ps) > 1000
    assert len({p.id for p in ps}) == len(ps)
    assert max(len(p.text.split()) for p in ps) <= 250
