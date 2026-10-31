from langscout.scripts import dominant_script, letter_count


def test_known_scripts():
    assert dominant_script("Hello world")[0] == "Latin"
    assert dominant_script("यह एक अच्छा दिन है")[0] == "Devanagari"      # includes combining vowel signs
    assert dominant_script("இது ஒரு நல்ல நாள்")[0] == "Tamil"
    assert dominant_script("Привет мир")[0] == "Cyrillic"
    assert dominant_script("这是一个测试")[0] == "Han"


def test_no_letters():
    assert dominant_script("123 !!! ...") == (None, 0.0)
    assert letter_count("") == 0


def test_mixed_share():
    script, share = dominant_script("Hello мир мир мир")
    assert script == "Cyrillic" and 0.5 < share < 1
