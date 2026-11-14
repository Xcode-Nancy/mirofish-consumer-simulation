"""Small smoke tests: the pure functions behave as documented. Run with:  python -m pytest -q"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import knowledge as K
import evaluate as E

def test_category_rules_check_accessories_first():
    assert K.classify("Fire HD 8 Tablet Case with stand") == "accessory"
    assert K.classify("Fire HD 8 Tablet, 16 GB") == "fire_tablet"
    assert K.classify("Amazon Echo Dot") == "echo_speaker"
    assert K.classify("something unrelated") == "other"

def test_sentiment_handles_negation():
    assert K.sent_score("The screen is great") == 1
    assert K.sent_score("The screen is not great") == -1
    assert K.sent_score("It arrived on Tuesday") == 0

def test_bm25_search_filters_by_product():
    import pandas as pd, pickle, tempfile, os
    texts = ["battery lasts all day", "terrible battery life", "great screen"]
    meta = pd.DataFrame(dict(review_id=["a", "b", "c"], product_id=["P1", "P2", "P1"], rating=[5, 1, 5], text=texts))
    path = os.path.join(tempfile.mkdtemp(), "idx.pkl")
    pickle.dump(dict(index=K.build_bm25(texts), meta=meta), open(path, "wb"))
    hits = K.Retriever(path).search("battery", k=3, product_id="P1")
    assert hits.iloc[0].review_id == "a" and hits.iloc[0].score > 0
    assert (hits[hits.score > 0].product_id == "P1").all()

def test_spearman_without_scipy():
    assert abs(E.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(E.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9
    assert np.isnan(E.spearman([1, 1, 1], [1, 2, 3]))
