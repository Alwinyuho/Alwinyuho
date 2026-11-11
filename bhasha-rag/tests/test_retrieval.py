from bhasharag.retrieval import BM25Index, rrf_fuse


def test_rrf_prefers_documents_found_by_both_lists():
    assert rrf_fuse([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_rrf_breaks_ties_consistently():
    assert rrf_fuse([[1], [2]]) == [1, 2]


def test_bm25_finds_indic_matches():
    texts = [
        "दिल्ली भारत की राजधानी है",
        "मुंबई एक बड़ा बंदरगाह है",
        "गंगा एक लंबी नदी है",
        "हिमालय ऊंचे पर्वत हैं",
    ]
    assert BM25Index(texts).search("भारत की राजधानी")[0] == 0


def test_bm25_returns_nothing_for_blank_or_unknown_queries():
    index = BM25Index(["दिल्ली भारत", "मुंबई शहर", "गंगा नदी"])
    assert index.search("...") == []
    assert index.search("கேள்வி") == []
