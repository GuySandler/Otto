"""Sentence segmentation + speech clean-up."""
from otto.pipeline.sentences import SentenceBuffer, clean_for_speech


def feed_all(buffer: SentenceBuffer, text: str, step: int = 3) -> list[str]:
    chunks: list[str] = []
    for i in range(0, len(text), step):
        chunks.extend(buffer.feed(text[i : i + step]))
    chunks.extend(buffer.flush())
    return chunks


def test_splits_on_sentence_boundaries():
    buffer = SentenceBuffer(first_max_chars=60, max_chars=200)
    chunks = feed_all(buffer, "Hello there. How are you today? I am fine!")
    assert chunks == ["Hello there.", "How are you today?", "I am fine!"]


def test_first_chunk_splits_early_on_comma():
    buffer = SentenceBuffer(first_max_chars=40, max_chars=200)
    text = (
        "So here is the thing, there are two meetings tomorrow and one of them is with design. "
        "They start at ten."
    )
    chunks = feed_all(buffer, text, step=4)
    assert len(chunks) >= 2
    assert len(chunks[0]) <= 40
    assert chunks[0].endswith(",")


def test_respects_abbreviations_and_decimals():
    buffer = SentenceBuffer(first_max_chars=200, max_chars=400)
    chunks = feed_all(buffer, "Dr. Smith paid 3.50 for it. Then he left.", step=5)
    assert chunks[0].startswith("Dr. Smith paid 3.50")
    assert chunks[-1] == "Then he left."


def test_short_sentences_are_merged_when_min_chars_is_set():
    buffer = SentenceBuffer(first_max_chars=60, max_chars=200, min_chars=40)
    chunks = feed_all(buffer, "Well.\nThe sea is a restless thing, never quite the same twice.")
    assert len(chunks) == 1
    assert chunks[0].startswith("Well.")


def test_long_text_still_splits_with_min_chars():
    buffer = SentenceBuffer(first_max_chars=60, max_chars=200, min_chars=40)
    chunks = feed_all(buffer, "This first sentence is comfortably long enough to stand alone. And so is this second one, easily.")
    assert len(chunks) == 2


def test_flush_returns_remainder():
    buffer = SentenceBuffer()
    assert buffer.feed("no terminator here") == []
    assert buffer.flush() == ["no terminator here"]
    assert buffer.flush() == []


def test_clean_strips_markdown_links_emoji():
    raw = "**Sure!** Here is the [link](https://example.com/x) and a URL https://a.b/c 🙂 (laughs)"
    cleaned = clean_for_speech(raw)
    assert "*" not in cleaned
    assert "https://" not in cleaned
    assert "link" in cleaned
    assert "laughs" not in cleaned
    assert "🙂" not in cleaned


def test_clean_keeps_plain_text_intact():
    assert clean_for_speech("Your next meeting is at 10 a.m. in room 4.") == "Your next meeting is at 10 a.m. in room 4."


def test_hard_wrap_for_long_text():
    buffer = SentenceBuffer(first_max_chars=40, max_chars=80)
    long_sentence = "word " * 60
    chunks = feed_all(buffer, long_sentence + ".", step=7)
    assert all(len(chunk) <= 90 for chunk in chunks)
    assert len(chunks) > 2
