from langscout.preprocess import chunk_text, clean_text, vectorizer_preprocessor


def test_clean_removes_noise():
    raw = "Hello [12] world http://x.org 2024 \\n again\\xa0done"
    assert clean_text(raw) == "Hello world again done"


def test_vectorizer_preprocessor_lowercases():
    assert vectorizer_preprocessor("  HeLLo  ") == "hello"


def test_chunks_respect_max_length_and_keep_content():
    text = " ".join(f"This is sentence number {'word ' * 8}end." for _ in range(20))
    chunks = chunk_text(text, max_chars=120)
    assert chunks and all(len(c) <= 120 for c in chunks)


def test_very_long_sentence_is_cut():
    chunks = chunk_text("word " * 200, max_chars=100)
    assert len(chunks) > 5 and all(len(c) <= 100 for c in chunks)


def test_hindi_danda_splits_sentences():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
    assert len(chunk_text(text, max_chars=25)) == 2


def test_tiny_text_dropped():
    assert chunk_text("ok", min_chars=5) == []
