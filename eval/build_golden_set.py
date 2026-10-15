"""Source of truth for the golden set. Writes eval/golden_set.jsonl after checking every
supporting passage id exists in the current corpus (so re-chunking cannot silently break it).

Also writes eval/GOLDEN_SET_REVIEW.md, which shows each answer next to the exact source text.

Usage: uv run python eval/build_golden_set.py
"""

import json
from pathlib import Path

from rail_rag.chunking import load_passages

OUT = Path(__file__).parent / "golden_set.jsonl"
REVIEW = Path(__file__).parent / "GOLDEN_SET_REVIEW.md"

# type: answerable | wrong_document | unanswerable
# supporting_passages: passage ids that together contain the reference answer
# expected_docs: documents a correct answer must cite (grounding check)
ITEMS = [
    {
        "id": "q01",
        "type": "answerable",
        "question": "What is the stationary noise limit LpAeq,T for an electric multiple unit (EMU)?",
        "reference_answer": (
            "65 dB. The stationary noise limit LpAeq,T[unit] for EMUs is 65 dB, measured at a "
            "distance of 7,5 m from the centre of the track and 1,2 m above top of rail."
        ),
        "supporting_passages": ["tsi-noi:Annex §4.2.1#1"],
        "expected_docs": ["tsi-noi"],
    },
    {
        "id": "q02",
        "type": "answerable",
        "question": "What is the pass-by noise limit for freight wagons at 80 km/h?",
        "reference_answer": (
            "83 dB. The limit LpAeq,Tp (80 km/h) for wagons is 83 dB, normalised to "
            "APL = 0,225 (number of axles divided by the length over buffers), at 7,5 m from the "
            "track centre and 1,2 m above top of rail."
        ),
        "supporting_passages": ["tsi-noi:Annex §4.2.3#1"],
        "expected_docs": ["tsi-noi"],
    },
    {
        "id": "q03",
        "type": "answerable",
        "question": "How wide must an escape walkway in a railway tunnel be, and how much "
        "headroom must it have?",
        "reference_answer": (
            "In tunnels longer than 0,5 km the walkway must be at least 0.8 m wide with a minimum "
            "vertical clearance of 2.25 m. Obstacles may not reduce the width below 0,7 m and may "
            "not be longer than 2 m."
        ),
        "supporting_passages": ["tsi-srt:Annex §4.2.1.6#1"],
        "expected_docs": ["tsi-srt"],
    },
    {
        "id": "q04",
        "type": "answerable",
        "question": "For how long must the emergency lighting on tunnel escape routes keep working "
        "after the main power supply fails?",
        "reference_answer": (
            "The SRT TSI sets no fixed duration. An alternative electricity supply must be "
            "available for an appropriate period after failure of the main supply; the time must "
            "be consistent with the evacuation scenarios and reported in the Emergency Plan. "
            "(The lighting must give a maintained illuminance of at least 1 lux at walkway level.)"
        ),
        "supporting_passages": ["tsi-srt:Annex §4.2.1.5.4#1"],
        "expected_docs": ["tsi-srt"],
        "notes": "Trap: the honest answer is 'no fixed number'. Inventing a duration is a "
        "faithfulness failure.",
    },
    {
        "id": "q05",
        "type": "answerable",
        "question": "What is the maximum distance between evacuation and rescue points in a tunnel "
        "for Category A and for Category B passenger rolling stock, and what is the difference "
        "between the two categories?",
        "reference_answer": (
            "5 km for Category A and 20 km for Category B. Category A rolling stock is for lines "
            "where the distance between evacuation and rescue points or the tunnel length does "
            "not exceed 5 km; Category B is for all tunnels irrespective of their length."
        ),
        "supporting_passages": ["tsi-srt:Annex §4.2.1.7#1", "tsi-srt:Annex §4.2.3#1"],
        "expected_docs": ["tsi-srt"],
        "notes": "Needs two sections of the same TSI (multi-passage recall).",
    },
    {
        "id": "q06",
        "type": "answerable",
        "question": "How many wheelchair spaces must a 250 m long passenger unit have?",
        "reference_answer": (
            "At least 3. Units of more than 205 m up to 300 m (length excluding the locomotive or "
            "power head) need 3 wheelchair spaces."
        ),
        "supporting_passages": ["tsi-prm:Annex §4.2.2.2#1"],
        "expected_docs": ["tsi-prm"],
        "notes": "Requires reading a table band, not a literal number match.",
    },
    {
        "id": "q07",
        "type": "answerable",
        "question": "What is the minimum clear usable width of exterior passenger doors, and of "
        "wheelchair access doors with level access?",
        "reference_answer": (
            "All exterior passenger doorways need a minimum clear usable width of 800 mm when "
            "open. On trains with a design speed below 250 km/h, wheelchair access doors offering "
            "level access need at least 1 000 mm."
        ),
        "supporting_passages": ["tsi-prm:Annex §4.2.2.3.2#1"],
        "expected_docs": ["tsi-prm"],
    },
    {
        "id": "q08",
        "type": "answerable",
        "question": "Which traction power supply systems does the Energy TSI allow, and which may "
        "be used on new lines faster than 250 km/h?",
        "reference_answer": (
            "Four systems: AC 25 kV 50 Hz, AC 15 kV 16,7 Hz, DC 3 kV and DC 1,5 kV. New lines with "
            "a speed above 250 km/h must be supplied with one of the two AC systems (25 kV 50 Hz "
            "or 15 kV 16,7 Hz)."
        ),
        "supporting_passages": ["tsi-ene:Annex §4.2.3#1", "tsi-ene:Annex §7.1.1#1"],
        "expected_docs": ["tsi-ene"],
        "notes": "Second half is only in the implementation chapter (multi-passage recall).",
    },
    {
        "id": "q09",
        "type": "answerable",
        "question": "What are the Class A train protection and radio systems under the CCS TSI?",
        "reference_answer": (
            "The Class A train protection system is ETCS. The Class A radio system is RMR "
            "(Railway Mobile Radio), which comprises GSM-R and FRMCS; they may be implemented "
            "together or independently. ERTMS consists of ETCS, RMR and ATO."
        ),
        "supporting_passages": ["tsi-ccs:Annex I §2.2#1"],
        "expected_docs": ["tsi-ccs"],
    },
    {
        "id": "q10",
        "type": "answerable",
        "question": "How must the rear end of a passenger train and of a freight train be signalled?",
        "reference_answer": (
            "The rear end signal is shown only on the rear of the last vehicle. A passenger train "
            "shows 2 steady red lights at the same height above buffer on the transversal axis. A "
            "freight train shows 2 reflective plates at the same height above buffer; a train with "
            "2 steady red lights also complies."
        ),
        "supporting_passages": [
            "tsi-ope:Annex §4.2.2.1.3#1",
            "tsi-ope:Annex §4.2.2.1.3.1#1",
            "tsi-ope:Annex §4.2.2.1.3.2#1",
        ],
        "expected_docs": ["tsi-ope"],
    },
    {
        "id": "q11",
        "type": "answerable",
        "question": "From when must freight trains with reflective plates as the rear-end signal be "
        "accepted in Portugal, and across the whole EU network?",
        "reference_answer": (
            "On rail freight corridors the general date is 1 January 2022, but on lines where "
            "steady red lights are an operational safety requirement Portugal (and Spain) had "
            "until 1 January 2025. On the whole EU rail network, reflective plates must be "
            "accepted from 1 January 2026."
        ),
        "supporting_passages": ["tsi-ope:Annex §Phasing out:#1"],
        "expected_docs": ["tsi-ope"],
        "notes": "Exception to an exception; tests careful reading of dates. Open point for "
        "review: the separate 'Specific cases' paragraph lets Portugal keep national rules "
        "requiring 2 red lights where justified; the answer follows the 'Phasing out' deadlines.",
    },
    {
        "id": "q12",
        "type": "answerable",
        "question": "How quickly must the Agency handle an application for a vehicle authorisation "
        "for placing on the market?",
        "reference_answer": (
            "Within one month of receiving the request, the Agency must tell the applicant that "
            "the file is complete or ask for supplementary information. It must then issue the "
            "authorisation or a negative decision within a reasonable time, and in any case within "
            "four months of receiving all relevant information."
        ),
        "supporting_passages": ["iop-directive:Article 21#5", "iop-directive:Article 21#3"],
        "expected_docs": ["iop-directive"],
    },
    {
        "id": "q13",
        "type": "answerable",
        "question": "How long does the Commission have to decide on a Member State's request not "
        "to apply a TSI, and what happens if it does not decide in time?",
        "reference_answer": (
            "Four months from submission of the request with the complete file. If the "
            "Commission takes no decision in that time, the request is deemed accepted. Pending "
            "the decision, the Member State may apply its alternative provisions without delay."
        ),
        "supporting_passages": ["iop-directive:Article 7#4"],
        "expected_docs": ["iop-directive"],
    },
    {
        "id": "q14",
        "type": "wrong_document",
        "question": "What minimum clear opening must the doors between a tunnel escape walkway and "
        "the safe area have?",
        "reference_answer": (
            "At least 1,4 m wide and 2,0 m high (multiple narrower doors side by side are allowed "
            "if the flow capacity is equivalent or higher). After the doors the clear width must "
            "continue at least 1,5 m wide and 2,25 m high. This applies to tunnels longer than "
            "1 km."
        ),
        "supporting_passages": ["tsi-srt:Annex §4.2.1.5.2#1"],
        "expected_docs": ["tsi-srt"],
        "notes": "Distractor: PRM TSI door widths (800 mm / 1 000 mm) for train doors.",
    },
    {
        "id": "q15",
        "type": "wrong_document",
        "question": "When does an upgraded or renewed vehicle that is already authorised need a new "
        "vehicle authorisation for placing on the market?",
        "reference_answer": (
            "When (a) the values of the parameters relevant for technical compatibility change "
            "outside the range of acceptable parameters defined in the TSIs, (b) the overall "
            "safety level of the vehicle may be adversely affected by the works, or (c) the "
            "relevant TSIs require it."
        ),
        "supporting_passages": ["iop-directive:Article 21#11"],
        "expected_docs": ["iop-directive"],
        "notes": "Distractor: the 'upgrade/renewal' implementation chapters of each TSI.",
    },
    {
        "id": "q16",
        "type": "wrong_document",
        "question": "Which language must train crew use for safety-related communication with the "
        "staff authorising train movements?",
        "reference_answer": (
            "The operating language(s) used by the infrastructure manager on the route concerned. "
            "The infrastructure manager publishes its operating language(s); the communication "
            "principles are in Appendix C of the OPE TSI."
        ),
        "supporting_passages": ["tsi-ope:Annex §4.2.1.5#1"],
        "expected_docs": ["tsi-ope"],
        "notes": "Distractors: CCS TSI harmonised DMI texts, PRM TSI passenger information.",
    },
    {
        "id": "q17",
        "type": "unanswerable",
        "question": "What is the capital of France?",
        "reference_answer": "Refuse: outside the scope of the corpus.",
        "supporting_passages": [],
        "expected_docs": [],
        "notes": "Clearly out of domain; should be stopped by the retrieval score floor.",
    },
    {
        "id": "q18",
        "type": "unanswerable",
        "question": "What GDPR obligations apply to railway undertakings that process passenger "
        "personal data?",
        "reference_answer": "Refuse: data protection is not covered by the indexed acts.",
        "supporting_passages": [],
        "expected_docs": [],
        "notes": "Near-domain: scored 0.715 in the retrieval probe, inside the in-scope range. "
        "Must be caught by the LLM grade/generate gates.",
    },
    {
        "id": "q19",
        "type": "unanswerable",
        "question": "Which collision scenarios must a locomotive's crashworthiness design withstand?",
        "reference_answer": (
            "Refuse: locomotive crashworthiness is not specified in the indexed acts (it belongs "
            "to the LOC&PAS TSI, which is not indexed)."
        ),
        "supporting_passages": [],
        "expected_docs": [],
        "notes": "Near-miss: the Directive's 'light rail' definition mentions crashworthiness "
        "classes but says nothing about locomotive collision scenarios.",
    },
    {
        "id": "q20",
        "type": "unanswerable",
        "question": "What is the maximum line speed on the Lisbon–Porto line?",
        "reference_answer": "Refuse: line-specific data is not in the TSIs.",
        "supporting_passages": [],
        "expected_docs": [],
        "notes": "Near-domain: the corpus mentions Portugal in several specific cases.",
    },
]


def write_review(passages: dict) -> None:
    lines = [
        "# Golden set review",
        "",
        "Generated by `eval/build_golden_set.py`. For each item: the question, the reference "
        "answer, and the full text of every supporting passage, to check the answer against.",
        "",
    ]
    for it in ITEMS:
        lines += [f"## {it['id']} · {it['type']}", "", f"**Q:** {it['question']}", ""]
        lines += [f"**Reference answer:** {it['reference_answer']}", ""]
        if it.get("notes"):
            lines += [f"**Why it's here:** {it['notes']}", ""]
        for pid in it["supporting_passages"]:
            p = passages[pid]
            quoted = "\n".join(f"> {line}" for line in p.text.splitlines())
            lines += [f"**Source:** `{pid}` ({p.citation}, {p.heading})", "", quoted, ""]
    REVIEW.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    passages = {p.id: p for p in load_passages()}
    ids = set(passages)
    missing = [
        (it["id"], pid) for it in ITEMS for pid in it["supporting_passages"] if pid not in ids
    ]
    if missing:
        raise SystemExit(f"unknown passage ids: {missing}")
    for it in ITEMS:
        docs = {pid.split(":")[0] for pid in it["supporting_passages"]}
        assert docs <= set(it["expected_docs"]), it["id"]
        assert (it["type"] == "unanswerable") == (not it["supporting_passages"]), it["id"]
    assert len({it["id"] for it in ITEMS}) == len(ITEMS) == 20
    with OUT.open("w", encoding="utf-8") as f:
        for it in ITEMS:
            f.write(json.dumps({"notes": "", **it}, ensure_ascii=False) + "\n")
    counts = {
        t: sum(it["type"] == t for it in ITEMS)
        for t in ("answerable", "wrong_document", "unanswerable")
    }
    write_review(passages)
    print(f"wrote {len(ITEMS)} items to {OUT.name}: {counts}; review sheet: {REVIEW.name}")


if __name__ == "__main__":
    main()
