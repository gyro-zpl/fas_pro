from experiments.eval.readout import parse_rating


def test_parse_direct():
    assert parse_rating("85") == 85
    assert parse_rating(" 85\n") == 85
    assert parse_rating("100") == 100
    assert parse_rating("0") == 0


def test_parse_wrapped():
    assert parse_rating("The answer is 62.") == 62
    assert parse_rating("Score: 40") == 40
    assert parse_rating("Predicted score: 92") == 92


def test_parse_invalid():
    assert parse_rating("no number") is None
    assert parse_rating("") is None
    assert parse_rating("999") is None


def test_parse_decimal():
    assert parse_rating("85.0") == 85


if __name__ == "__main__":
    for fn in [test_parse_direct, test_parse_wrapped, test_parse_invalid, test_parse_decimal]:
        fn()
        print("ok", fn.__name__)
