from bhasharag.data import assign_split, collect_distractors, prepare_benchmark
from bhasharag.text import doc_id

LONG = " ".join(f"शब्द{i}" for i in range(30))


def row(qid, query="प्रश्न", passage="अनुच्छेद"):
    return {"query_id": qid, "query": query, "passage": passage, "answer": "उत्तर"}


def test_split_is_deterministic_and_uses_both_splits():
    splits = [assign_split(str(i)) for i in range(200)]
    assert splits == [assign_split(str(i)) for i in range(200)]
    assert set(splits) == {"val", "test"}


def test_duplicate_passages_collapse_to_one_document():
    gold, queries = prepare_benchmark([row("1"), row("2")])
    assert len(gold) == 1
    assert queries["gold_doc_id"].nunique() == 1


def test_duplicate_query_ids_and_empty_rows_are_dropped():
    _, queries = prepare_benchmark([row("1"), row("1", passage="दूसरा"), row("3", query=" ")])
    assert queries["query_id"].tolist() == ["1"]


def test_distractors_never_include_gold_or_short_text():
    other = LONG + " अंतिम"
    frame = collect_distractors(["छोटा", LONG, other, other], 5, {doc_id(LONG)})
    assert frame["text"].tolist() == [other]


def test_distractor_limit_is_respected():
    paragraphs = [f"{LONG} {i}" for i in range(10)]
    assert len(collect_distractors(paragraphs, 3, set())) == 3
