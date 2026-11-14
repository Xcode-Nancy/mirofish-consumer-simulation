"""
Phase 7b: leakage-free evaluation of the simulator's product signal.

    python phase7_negative_eval.py --project-dir . [--min-reviews 100] [--splits 200]

Put it in the same folder as phase4_6_simulate.py / phase3_build.py (run both of those first).

Design (fixes the circularity of the first evaluation)
  * Every product's reviews are split at random into half A and half B.
  * The simulator's product signal is built ONLY from half A (aspect satisfaction -> agents' weighted view of the product).
  * It is scored against outcomes measured ONLY on half B:
        pos45 = share of 4-5 star reviews        neg12 = share of 1-2 star reviews (severe complaints)
  * Prediction is leave-one-product-out: a straight line (outcome ~ signal) is fitted on the other products and
    predicts the held-out one. Compared with (a) the constant mean of the other products and (b) the product's own
    half-A mean star rating (a "persistence" baseline that shows how much any review-based predictor could get).
  * Repeated over many random splits. Also reports the split-half reliability of the aspect profile itself.

Methods and why
  * Split-half: a product's profile is built from half of its reviews and scored on the other half. The first evaluation used the same
    reviews for inputs and target (and the aspect sentiment partly comes from star ratings), which inflates any score.
  * Leave-one-product-out fitting: each product is predicted by a line fitted on the other products, so the test product never
    influences its own prediction.
  * Three predictors compared: the simulator signal, a constant mean, and the product's own half-A mean star rating ("persistence").
    The persistence baseline shows how much ANY review-based predictor could achieve.
  * 200 random splits instead of one: with only ~12 products one split is mostly luck; the spread across splits shows how stable the
    conclusion is. Spearman (rank) correlation is used because only the ordering of products is meaningful at this scale.
  * Two targets: share of 4-5 star reviews and share of 1-2 star reviews (the harsher, severe-complaint side).
  * Aspect-profile reliability (split-half correlation of product-by-aspect deviations): tells us whether the model inputs are stable
    signal or noise.
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from simulate import make_agents, simulate, A, ASSUMED, SCENARIOS

warnings.filterwarnings("ignore")

def shrink_profile(am, n0):
    G = am.groupby("aspect").sentiment.mean().reindex(A).fillna(0).values
    SAT = {}
    for pid, g in am.groupby("product_id"):
        v = G.copy()
        for aspect, s in g.groupby("aspect").sentiment.agg(["mean", "size"]).iterrows():
            i = A.index(aspect); v[i] = (s["size"] * s["mean"] + n0 * G[i]) / (s["size"] + n0)
        SAT[pid] = v
    return G, SAT

def spearman(x, y):
    """Spearman = Pearson on ranks (no scipy needed)."""
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    if rx.std() < 1e-12 or ry.std() < 1e-12: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def lopo(signal, y):
    """leave-one-product-out line fit -> predictions"""
    n = len(y); pred = np.zeros(n)
    for i in range(n):
        m = np.arange(n) != i
        if np.std(signal[m]) < 1e-12: pred[i] = y[m].mean()
        else:
            b, a0 = np.polyfit(signal[m], y[m], 1); pred[i] = a0 + b * signal[i]
    return pred

def errs(pred, y): return np.abs(pred - y).mean(), np.sqrt(((pred - y) ** 2).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="."); ap.add_argument("--min-reviews", type=int, default=100)
    ap.add_argument("--splits", type=int, default=200); ap.add_argument("--agents", type=int, default=400); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(); P = lambda *x: os.path.join(a.project_dir, *x); D = P("data", "processed"); out = P("outputs"); os.makedirs(out, exist_ok=True)

    pj = json.load(open(os.path.join(D, "personas.json")))
    personas = {k: v for k, v in pj["personas"].items() if not v["low_data"]}
    am = pd.read_csv(os.path.join(D, "aspect_mentions.csv"), usecols=["review_id", "product_id", "aspect", "sentiment"])
    cat = pd.read_csv(os.path.join(D, "category_reviews.csv"), usecols=["review_id", "product_id", "rating"])
    counts = cat.product_id.value_counts(); prods = counts[counts >= a.min_reviews].index.tolist()
    cat = cat[cat.product_id.isin(prods)].reset_index(drop=True); am = am[am.product_id.isin(prods)]
    print(f"{len(prods)} products with >= {a.min_reviews} reviews ({len(cat):,} reviews), {a.splits} random splits")
    rng = np.random.default_rng(a.seed)
    ag = make_agents(personas, a.agents, np.random.default_rng(a.seed)); zero = np.zeros(len(A)); base = SCENARIOS["baseline"]

    rows, rel = [], []
    for r in range(a.splits):
        inA = pd.Series(rng.random(len(cat)) < 0.5, index=cat.review_id.values)
        catA, catB = cat[inA.values], cat[~inA.values]
        amA = am[am.review_id.map(inA).fillna(False).astype(bool)]; amB = am[~am.review_id.map(inA).fillna(True).astype(bool)]
        G, SAT = shrink_profile(amA, ASSUMED["shrink_n"])
        sig = np.array([simulate(ag, SAT.get(p, G), G, zero, 1.0, 0.0, base).mean() for p in prods])
        starA = catA.groupby("product_id").rating.mean().reindex(prods).values
        yB = {"pos45": catB.groupby("product_id").rating.apply(lambda s: (s >= 4).mean()).reindex(prods).values,
              "neg12": catB.groupby("product_id").rating.apply(lambda s: (s <= 2).mean()).reindex(prods).values}
        for tname, y in yB.items():
            ok = ~np.isnan(y)
            if ok.sum() < 5: continue
            s_, y_, st_ = sig[ok], y[ok], starA[ok]
            e_sim, e_const, e_star = errs(lopo(s_, y_), y_), errs(np.array([np.delete(y_, i).mean() for i in range(len(y_))]), y_), errs(lopo(st_, y_), y_)
            rows.append(dict(split=r, target=tname, mae_sim=e_sim[0], mae_const=e_const[0], mae_star=e_star[0], rmse_sim=e_sim[1], rmse_const=e_const[1], rmse_star=e_star[1],
                             spearman_sim=spearman(s_, y_), spearman_star=spearman(st_, y_)))
        # reliability of the aspect profile: product x aspect cells, product deviation from aspect mean, A vs B
        ca = amA.groupby(["product_id", "aspect"]).sentiment.agg(["mean", "size"]); cb = amB.groupby(["product_id", "aspect"]).sentiment.agg(["mean", "size"])
        j = ca.join(cb, lsuffix="_A", rsuffix="_B", how="inner"); j = j[(j.size_A >= 20) & (j.size_B >= 20)]
        if len(j) >= 8:
            da = j.mean_A - j.groupby(level="aspect").mean_A.transform("mean"); db = j.mean_B - j.groupby(level="aspect").mean_B.transform("mean")
            rel.append(np.corrcoef(da, db)[0, 1])

    R = pd.DataFrame(rows); S = []
    for t, g in R.groupby("target"):
        S.append(dict(target=t, products=len(prods), splits=len(g),
                      MAE_simulation=g.mae_sim.mean(), MAE_constant=g.mae_const.mean(), MAE_star_persistence=g.mae_star.mean(),
                      RMSE_simulation=g.rmse_sim.mean(), RMSE_constant=g.rmse_const.mean(),
                      sim_beats_constant_pct=100 * (g.mae_sim < g.mae_const).mean(),
                      spearman_sim_mean=g.spearman_sim.mean(), spearman_sim_p5=g.spearman_sim.quantile(.05), spearman_sim_p95=g.spearman_sim.quantile(.95),
                      spearman_star_mean=g.spearman_star.mean()))
    S = pd.DataFrame(S).round(4); S.to_csv(os.path.join(out, "negative_eval_summary.csv"), index=False)
    relm, rel5, rel95 = (float(np.mean(rel)), float(np.quantile(rel, .05)), float(np.quantile(rel, .95))) if rel else (np.nan,) * 3
    print("\n" + S.T.to_string()); print(f"\naspect-profile split-half reliability (product x aspect deviations): mean r = {relm:.3f} (5-95%: {rel5:.3f} to {rel95:.3f})")

    L = ["# Leakage-free evaluation (split-half, leave-one-product-out)", "",
         f"- Products: {len(prods)} (>= {a.min_reviews} reviews) | random splits: {a.splits} | agents: {a.agents}",
         "- Signal built from half A of each product's reviews, scored on outcomes from half B (a review never predicts itself).",
         "- 'persistence' = the product's own half-A mean star rating, an upper-bound-style reference for any review-based predictor.", "",
         S.T.to_string(), "",
         f"- Reliability of the aspect profile itself (A vs B, product-by-aspect deviations): mean r = {relm:.3f} (5-95%: {rel5:.3f} to {rel95:.3f}).", "",
         "How to read it: if `sim_beats_constant_pct` is near 50% and the Spearman interval spans 0, the simulator's product signal",
         "adds nothing over a constant. If the persistence baseline is also near the constant, the outcome differences between products are",
         "mostly noise and no review-based model could predict them. Products are the unit of analysis (few), so keep the claims modest."]
    open(os.path.join(out, "negative_eval.md"), "w", encoding="utf-8").write("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, figsize=(11, 4))
        for t, c in (("pos45", "#4c78a8"), ("neg12", "#e45756")):
            g = R[R.target == t]; ax[0].hist(g.spearman_sim, bins=20, alpha=.6, color=c, label=t)
        ax[0].axvline(0, color="k", lw=.8); ax[0].set_title("Spearman: simulator signal vs held-out outcome\n(200 random splits)"); ax[0].legend()
        names = ["simulation", "constant", "star persistence"]; w = .35
        for k, (t, c) in enumerate((("pos45", "#4c78a8"), ("neg12", "#e45756"))):
            g = R[R.target == t]; ax[1].bar(np.arange(3) + k * w, [g.mae_sim.mean(), g.mae_const.mean(), g.mae_star.mean()], w, color=c, label=t)
        ax[1].set_xticks(np.arange(3) + w / 2); ax[1].set_xticklabels(names); ax[1].set_title("Held-out MAE (lower is better)"); ax[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(out, "negative_eval.png"), dpi=130)
    except ImportError:
        pass
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
