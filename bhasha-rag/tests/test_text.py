import re
import unicodedata

from bhasharag.text import doc_id, normalize, tokenize


def test_combining_vowel_signs_stay_in_the_word():
    assert tokenize("भारत की राजधानी") == ["भारत", "की", "राजधानी"]
    assert tokenize("தமிழ் மொழி") == ["தமிழ்", "மொழி"]


def test_builtin_word_pattern_would_break_this_text():
    # Documents why the project does not use re's \w for tokenizing.
    assert re.findall(r"\w+", "की") != ["की"]


def test_punctuation_and_danda_are_removed():
    assert tokenize("नमस्ते, दुनिया।") == ["नमस्ते", "दुनिया"]


def test_unicode_forms_are_equivalent():
    decomposed = unicodedata.normalize("NFD", "ಕೋ")
    assert normalize(decomposed) == normalize("ಕೋ")


def test_zero_width_joiners_do_not_change_tokens():
    assert tokenize("क्‍ष") == tokenize("क्ष")


def test_doc_id_ignores_spacing_and_punctuation():
    assert doc_id("भारत  की, राजधानी") == doc_id("भारत की राजधानी")
    assert doc_id("भारत") != doc_id("दिल्ली")
