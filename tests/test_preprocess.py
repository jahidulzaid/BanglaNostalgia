from src.preprocess import clean_bengali_text


def test_clean_bengali_text_keeps_bengali_and_punct():
    text = "English বাংলা! 123 ???"
    cleaned = clean_bengali_text(text)
    assert cleaned == "বাংলা!"
