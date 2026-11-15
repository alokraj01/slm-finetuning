from src.eval_metrics import Counts, dedupe, jaccard, match, normalize, parse_output, score, verbatim_filter

CATS = ["Governing Law", "Parties"]


def test_parse_strict_and_fenced():
    out, f = parse_output('{"Governing Law": ["New York"], "Parties": []}', CATS)
    assert out == {"Governing Law": ["New York"], "Parties": []} and f["strict_json"] and f["keys_ok"]
    out, f = parse_output('```json\n{"Governing Law": "New York"}\n```', CATS)
    assert out == {"Governing Law": ["New York"], "Parties": []}
    assert not f["strict_json"] and f["json_ok"] and not f["keys_ok"]


def test_parse_invalid():
    out, f = parse_output("The governing law is New York.", CATS)
    assert out is None and not f["json_ok"]
    assert parse_output("[1, 2]", CATS)[0] is None


def test_normalize_and_jaccard():
    assert normalize("The  Company, Inc.") == "company inc"
    assert jaccard("laws of the State of New York", "the laws of New York") == 1.0 * 4 / 5
    assert jaccard("", "") == 1.0


def test_match_is_one_to_one():
    tp, _ = match(["Company", "Company"], ["Company"])
    assert tp == 1
    tp, _ = match(["governed by the laws of New York"], ["laws of New York shall govern"])
    assert tp == 1  # Jaccard 4/8 = 0.5, at the threshold
    tp, _ = match(["New York"], ["This Agreement shall be governed by the laws of the State of New York"])
    assert tp == 0  # a fragment of the clause is not a match


def test_dedupe_by_normalized_text():
    assert dedupe(["Company", "company.", "Seller"]) == ["Company", "Seller"]


def test_counts_and_rates():
    c = Counts()
    c.add(["a b c"], ["a b c"])  # TP
    c.add(["x y"], [])           # false alarm
    c.add([], ["q r"])           # miss
    c.add([], [])                # true negative
    s = c.summary()
    assert (s["tp"], s["n_pred"], s["n_gold"]) == (1, 2, 2)
    assert s["P"] == 0.5 and s["R"] == 0.5
    assert s["false_alarm_rate"] == 0.5 and s["miss_rate"] == 0.5


def test_score_groups():
    r = score([("A", ["x"], ["x"]), ("B", [], ["y"])], {"g": ["A", "B"]})
    assert r["per_group"]["g"]["tp"] == 1 and r["per_group"]["g"]["n_gold"] == 2


def test_empty_bucket_is_unscored_and_true_negatives_are_perfect():
    assert Counts().summary()["F1"] is None  # no items scored: no number, not a fake 1.0
    c = Counts()
    c.add([], [])
    assert c.summary()["F1"] == 1.0  # nothing predicted, nothing to find


def test_verbatim_filter():
    ok, bad = verbatim_filter(["New York", "California"], "laws of New York")
    assert ok == ["New York"] and bad == ["California"]
