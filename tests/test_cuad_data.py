import json
import random

import pytest

from src.cuad_data import (HELDOUT_CATEGORIES, TRAIN_CATEGORIES, build_messages, chunk_text, map_span,
                           normalize_whitespace)


def test_normalize_collapses_whitespace():
    norm, _ = normalize_whitespace("ten (10)        years  the  Term\n\n\n\nSection 2.\n  next")
    assert norm == "ten (10) years the Term\n\nSection 2.\nnext"


@pytest.mark.parametrize("seed", range(20))
def test_mapped_spans_equal_normalized_original(seed):
    rng = random.Random(seed)
    alphabet = ["a", "b", " ", "  ", "\n", "\n\n\n", " ", ".", "\t"]
    text = "".join(rng.choice(alphabet) for _ in range(400))
    norm, pos_map = normalize_whitespace(text)
    for _ in range(50):
        s = rng.randrange(len(text))
        e = rng.randrange(s + 1, len(text) + 1)
        m = map_span(norm, pos_map, s, e)
        expected, _ = normalize_whitespace(text[s:e])
        if not expected.strip():
            assert m is None
            continue
        assert norm[m[0]:m[1]] == expected.strip()


def _doc(n_paragraphs=200, seed=0):
    rng = random.Random(seed)
    paras = [" ".join(f"Sentence {i}.{j} has words." for j in range(rng.randint(1, 12))) for i in range(n_paragraphs)]
    return "\n\n".join(paras)


@pytest.mark.parametrize("size,overlap", [(6000, 800), (2000, 300), (12000, 1500)])
def test_chunks_cover_text_with_overlap_and_size_limit(size, overlap):
    text = _doc()
    chunks = chunk_text(text, size, overlap)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(chunks, chunks[1:]):
        assert e1 - s1 <= size
        assert s1 < s2 < e1, "consecutive chunks must overlap and advance"
        assert e1 - s2 >= overlap
    assert all(e - s <= size for s, e in chunks)


def test_chunks_prefer_paragraph_boundaries():
    text = _doc()
    for s, e in chunk_text(text, 6000, 800)[:-1]:
        assert text[e - 2:e] == "\n\n" or text[e - 1] == " "


def test_short_text_is_one_chunk():
    assert chunk_text("short contract", 6000, 800) == [(0, 14)]


def test_messages_target_order_and_json():
    defs = {c: f"definition of {c}" for c in TRAIN_CATEGORIES + HELDOUT_CATEGORIES}
    cats = ["Governing Law", "Parties"]
    msgs = build_messages("excerpt text", cats, defs, {"Governing Law": [], "Parties": ["excerpt"]})
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert list(json.loads(msgs[2]["content"])) == cats
    assert "- Governing Law: definition of Governing Law" in msgs[1]["content"]
    with pytest.raises(AssertionError):
        build_messages("x", cats, defs, {"Parties": [], "Governing Law": []})
