from bhasharag.generation import NO_ANSWER, build_messages, extractive_answer, parse_answer


def test_refusal_token_is_detected():
    assert parse_answer(NO_ANSWER, 3).refused


def test_valid_citations_are_kept_and_invalid_removed():
    parsed = parse_answer("दिल्ली [2] और मुंबई [9] [2]", 3)
    assert parsed.citations == [2]
    assert parsed.grounded


def test_answer_without_valid_citation_is_not_grounded():
    assert not parse_answer("दिल्ली [7]", 3).grounded
    assert not parse_answer("दिल्ली", 3).grounded


def test_thinking_text_is_removed():
    parsed = parse_answer("<think>maybe [1]</think>दिल्ली", 3)
    assert parsed.text == "दिल्ली"
    assert not parsed.grounded


def test_extractive_answer_stops_at_danda():
    parsed = extractive_answer(["पहला वाक्य। दूसरा वाक्य।"])
    assert parsed.text == "पहला वाक्य। [1]"
    assert parsed.citations == [1]


def test_prompt_numbers_passages_and_names_language():
    messages = build_messages("प्रश्न", ["एक", "दो"], "hi")
    assert "Hindi" in messages[0]["content"]
    assert "[2] दो" in messages[1]["content"]
