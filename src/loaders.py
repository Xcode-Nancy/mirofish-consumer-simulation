"""
Shared loaders: read the Phase-3 outputs once and calibrate the model intercept.

Kept in its own module so that app.py and conversations.py can both import it without importing each other.

Methods and why
  * Aspect satisfaction per product is SHRUNK toward the category average: (n*mean + n0*category_mean) / (n + n0), n0 = 10.
    Why: a product with 3 mentions of "battery" would otherwise get an extreme, noisy score; shrinkage (a simple empirical-Bayes
    idea) lets well-reviewed products keep their own score while thin ones fall back to the category norm.
  * "Nudge" per aspect = how far the retrieved reviews about that aspect sit from the product's mean star rating, so the retrieval
    layer (RAG) genuinely feeds the simulation instead of being decoration.
  * calibrate_b0 bisects the logistic intercept so the simulated average over products equals the real average positive share.
    Why: the data has no sales scale, so the only honest anchor for the absolute level is the observed positive-review share.
"""
import json, os
import numpy as np
import pandas as pd
from knowledge import Retriever
from simulate import simulate, A, ASSUMED, SCENARIOS

def load_state(project_dir):
    D = os.path.join(project_dir, "data", "processed")
    pj = json.load(open(os.path.join(D, "personas.json")))
    personas = {k: v for k, v in pj["personas"].items() if not v["low_data"]}
    am = pd.read_csv(os.path.join(D, "aspect_mentions.csv"))
    ref = pd.read_csv(os.path.join(D, "reference_outcomes.csv"))
    cat = pd.read_csv(os.path.join(D, "category_reviews.csv"), usecols=["review_id", "product_id", "rating"])
    ret = Retriever(os.path.join(D, "retrieval_index.pkl"))
    G = am.groupby("aspect").sentiment.mean().reindex(A).fillna(0).values
    st = am.groupby(["product_id", "aspect"]).sentiment.agg(["mean", "size"]).reset_index()
    n0, SAT = ASSUMED["shrink_n"], {}
    for pid, g in st.groupby("product_id"):
        v = G.copy()
        for r in g.itertuples():
            i = A.index(r.aspect); v[i] = (r.size * r.mean + n0 * G[i]) / (r.size + n0)
        SAT[pid] = v
    prod_mean = cat.groupby("product_id").rating.mean(); NUDGE = {}
    for pid in ref.product_id:
        row = np.zeros(len(A))
        for i, asp in enumerate(A):
            hits = ret.search(asp.replace("_", " "), k=5, product_id=pid); hits = hits[hits.score > 0]
            if len(hits): row[i] = (hits.rating.mean() - prod_mean[pid]) / 2
        NUDGE[pid] = row
    ref = ref[ref.product_id.isin(SAT)].reset_index(drop=True)
    return dict(ret=ret, personas=personas, G=G, SAT=SAT, NUDGE=NUDGE, ref=ref, category=pj["category"], project_dir=project_dir)

def calibrate_b0(S, ag, k):
    """intercept so that the average simulated baseline over all products equals the real average positive share"""
    base, target = SCENARIOS["baseline"], S["ref"].positive_share.mean(); lo, hi = -8.0, 8.0
    for _ in range(30):
        mid = (lo + hi) / 2
        m = np.mean([simulate(ag, S["SAT"][p], S["G"], S["NUDGE"][p], k, mid, base).mean() for p in S["ref"].product_id])
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return (lo + hi) / 2
