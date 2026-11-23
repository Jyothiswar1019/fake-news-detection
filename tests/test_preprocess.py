from fakenews.preprocess import TextCleaner, clean_text


def test_source_tags_removed():
    t = "WASHINGTON (Reuters) - Senators met. Featured image via Getty pic.twitter.com/abc123"
    out = clean_text(t)
    for bad in ("reuters", "featur", "twitter", "pic"):
        assert bad not in out


def test_flag_off_keeps_tags():
    assert "reuter" in clean_text("WASHINGTON (Reuters) - Senators met.", strip_source_tags=False)


def test_apostrophes_equalised():
    assert clean_text("He didn’t go") == clean_text("He didn t go")


def test_stopwords_stemming_and_short_tokens():
    out = clean_text("The cats are running to the market")
    assert out == "cat run market"


def test_transformer_handles_none():
    assert TextCleaner().transform([None, "Hello world"]) == ["", "hello world"]
def test_new_fingerprints_removed():
    t = "Trump spoke Friday. pic. twitter. com/xyz Image via Getty Images. Breitbart reported."
    out = clean_text(t)
    for bad in ("friday", "pic", "twitter", "getti", "via", "breitbart", "imag"):
        assert bad not in out


def test_dates_flag_independent():
    assert "friday" in clean_text("He left on Friday", strip_dates=False)
def test_stripped_embed_and_url_remnants():
    out = clean_text("My statement: pic. twitter.   —   Peter Tatchell see https:  . Screenshot Flickr")
    for bad in ("pic", "twitter", "http", "screenshot", "flickr"):
        assert bad not in out
    assert "peter" in out and "tatchel" in out  # real content survives
def test_round_three_boilerplate_removed():
    t = ("Become a Member of 21st  Century  Wire. Watch video below. Screen capture from YouTube. "
         "Gateway Pundit. Loading... Please share. Used with permission. Senators voted on weekday.")
    out = clean_text(t)
    for bad in ("member", "wire", "screen", "youtub", "gateway", "load", "permiss", "weekday"):
        assert bad not in out
    assert "senat" in out and "vote" in out
def test_factcheck_boilerplate_removed():
    out = clean_text("Fact Check: Here’s the truth behind this video. Don’t fall for this morphed image")
    for bad in ("fact", "truth", "fall"):
        assert bad not in out
    assert "video" in out and "morph" in out