"""
Phases 4-7: agent generation, multi-round simulation, scenario experiments, evaluation, report.

    python phase4_6_simulate.py --project-dir .  [--product B01XXXX] [--agents 400] [--seed 7]

Needs the outputs of phase3_build.py (personas.json, aspect_mentions.csv, category_reviews.csv, retrieval_index.pkl,
reference_outcomes.csv). Only pandas, numpy (matplotlib optional for the chart).

How it works
  * Agents: sampled from the data-calibrated personas; each gets aspect weights (persona importance x noise),
    peer susceptibility, influence weight and a "top concern" aspect.
  * Each agent judges a product from that product's REAL aspect satisfaction (shrunk toward the category mean),
    compared with the category average, plus the persona's own baseline propensity.
  * Rounds: agents retrieve real reviews about their top concern (BM25 RAG), then update their opinion from those
    reviews and from the (influence-weighted) crowd.
  * Evaluation: 2 numbers are fitted (intercept b0, signal strength k) on 60% of products; the other 40% are held out
    and compared with the real positive-review share (MAE, RMSE, correlation) against a constant-mean baseline.
  * Scenario effects (price elasticity, marketing, competitor switching) are ASSUMPTIONS: the review files contain no
    prices or sales, so they cannot be fitted. They are listed in ASSUMED below and in report.md.

Methods and why
  * Logistic (sigmoid) decision rule: bounded between 0 and 1, smooth, and every term has a plain meaning (product fit, price, marketing,
    competitor).
  * Agents are sampled from the calibrated personas with lognormal noise on their aspect weights: gives heterogeneity (the proposal's core
    idea) without inventing demographic traits.
  * Constant-elasticity price effect, z -= elasticity * log(price multiplier): a standard demand form, symmetric in percentage terms. The
    elasticity itself is ASSUMED because the review data contains no prices.
  * Peer influence = influence-weighted crowd average (a DeGroot-style opinion-dynamics model). Influencers count 3x and reviewers 1.5x,
    following the proposal's agent types. Simple and explainable; pairwise talk lives in conversations.py.
  * Only two numbers are fitted (intercept and signal strength k) on 60% of the products and tested on the other 40%, shown next to a
    constant-mean baseline: few parameters limit over-fitting, and the baseline shows whether the model adds anything.
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from knowledge import Retriever, ASPECTS

warnings.filterwarnings("ignore")
A = list(ASPECTS)
ASSUMED = dict(price_elasticity=2.5, marketing_boost=0.5, competitor_switch=2.0, review_weight=0.6, peer_rounds=5, shrink_n=10)
COMP_AFFINITY = dict(competitor_customer=1.0, price_sensitive=0.5, budget_conscious=0.5)   # others default 0.2 (assumption)
SCENARIOS = {"baseline": dict(price=1.0, marketing=0.0, comp=1.0), "discount_15pct": dict(price=0.85, marketing=0.0, comp=1.0),
             "price_up_20pct": dict(price=1.2, marketing=0.0, comp=1.0), "stronger_marketing": dict(price=1.0, marketing=1.0, comp=1.0),
             "competitor_price_cut_15pct": dict(price=1.0, marketing=0.0, comp=0.85)}

sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

def make_agents(personas, n, rng):
    names = list(personas); share = np.array([personas[k]["share"] for k in names], float); share /= share.sum()
    pi = rng.choice(len(names), size=n, p=share)
    base = np.mean([personas[k]["buy_propensity"] for k in names])
    W = np.zeros((n, len(A))); off = np.zeros(n); psens = np.ones(n); aff = np.zeros(n); infl = np.ones(n)
    for j, k in enumerate(names):
        m = pi == j; p = personas[k]
        imp = np.array([p["aspect_importance"][a] for a in A]) + 1e-3
        w = imp * rng.lognormal(0, 0.3, size=(m.sum(), len(A))); W[m] = w / w.sum(1, keepdims=True) * len(A)
        off[m] = logit(p["buy_propensity"]) - logit(base)
        lift = p["aspect_lift_vs_all"].get("price"); psens[m] = lift if lift else 1.0
        aff[m] = COMP_AFFINITY.get(k, 0.2); infl[m] = 3.0 if k == "influencer" else 1.5 if k == "product_reviewer" else 1.0
    return dict(names=names, pi=pi, W=W, off=off, psens=psens, aff=aff, infl=infl,
                s=rng.beta(2, 5, size=n), top=W.argmax(1))

def simulate(ag, prod_sat, G, nudge, k, b0, scen):
    dev = (ag["W"] * (prod_sat - G)).sum(1)
    z = b0 + ag["off"] + k * dev
    z = z - ASSUMED["price_elasticity"] * np.log(scen["price"]) * ag["psens"]
    z = z + ASSUMED["marketing_boost"] * scen["marketing"] * (1 - ag["s"])
    z = z - ASSUMED["competitor_switch"] * np.log(1 / scen["comp"]) * ag["aff"]
    o, R = z.copy(), ASSUMED["peer_rounds"]
    for _ in range(R):
        peer = (ag["infl"] * o).sum() / ag["infl"].sum()
        o = (1 - ag["s"]) * o + ag["s"] * peer + ASSUMED["review_weight"] * nudge[ag["top"]] / R
    return sig(o)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="."); ap.add_argument("--product", default=None)
    ap.add_argument("--agents", type=int, default=400); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); P = lambda *x: os.path.join(a.project_dir, *x); D = P("data", "processed")
    out = P("outputs"); os.makedirs(out, exist_ok=True); rng = np.random.default_rng(a.seed)

    pj = json.load(open(os.path.join(D, "personas.json")))
    personas = {k: v for k, v in pj["personas"].items() if not v["low_data"]}
    skipped = [k for k, v in pj["personas"].items() if v["low_data"]]
    am = pd.read_csv(os.path.join(D, "aspect_mentions.csv")); ref = pd.read_csv(os.path.join(D, "reference_outcomes.csv"))
    cat = pd.read_csv(os.path.join(D, "category_reviews.csv"), usecols=["review_id", "product_id", "rating"])
    ret = Retriever(os.path.join(D, "retrieval_index.pkl"))
    G = am.groupby("aspect").sentiment.mean().reindex(A).fillna(0).values
    st = am.groupby(["product_id", "aspect"]).sentiment.agg(["mean", "size"]).reset_index()
    n0 = ASSUMED["shrink_n"]; SAT = {}
    for pid, g in st.groupby("product_id"):
        v = G.copy()
        for r in g.itertuples(): i = A.index(r.aspect); v[i] = (r.size * r.mean + n0 * G[i]) / (r.size + n0)
        SAT[pid] = v
    prod_mean = cat.groupby("product_id").rating.mean()
    NUDGE = {}                                        # (product) -> per-aspect shift from retrieved reviews
    for pid in ref.product_id:
        row = np.zeros(len(A))
        for i, asp in enumerate(A):
            hits = ret.search(asp.replace("_", " "), k=5, product_id=pid)
            hits = hits[hits.score > 0]
            if len(hits): row[i] = (hits.rating.mean() - prod_mean[pid]) / 2
        NUDGE[pid] = row
    ag = make_agents(personas, a.agents, rng)

    # ---- fit b0 (intercept) and k (signal strength) on train products, test on held-out
    ref = ref[ref.product_id.isin(SAT)].reset_index(drop=True)
    idx = rng.permutation(len(ref)); ntr = max(2, int(0.6 * len(ref))); ref["split"] = "test"; ref.loc[idx[:ntr], "split"] = "train"
    base = SCENARIOS["baseline"]
    def predict(pids, k, b0): return np.array([simulate(ag, SAT[p], G, NUDGE[p], k, b0, base).mean() for p in pids])
    def fit_b0(pids, target, k):
        lo, hi = -8.0, 8.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if predict(pids, k, mid).mean() < target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    tr = ref[ref.split == "train"]; te = ref[ref.split == "test"]; best = None
    for k in [0, 0.5, 1, 2, 3, 5, 8, 12]:
        b0 = fit_b0(tr.product_id, tr.positive_share.mean(), k)
        mae = np.abs(predict(tr.product_id, k, b0) - tr.positive_share.values).mean()
        if best is None or mae < best[0] - 1e-9: best = (mae, k, b0)
    _, K, B0 = best
    ref["simulated_positive_share"] = predict(ref.product_id, K, B0)
    ref["constant_baseline"] = tr.positive_share.mean()
    def metrics(d, col):
        e = d[col] - d.positive_share; c = np.corrcoef(d[col], d.positive_share)[0, 1] if d[col].std() > 0 and len(d) > 2 else np.nan
        sp = d[col].rank().corr(d.positive_share.rank()) if d[col].std() > 0 and len(d) > 2 else np.nan
        return dict(MAE=round(float(e.abs().mean()), 4), RMSE=round(float(np.sqrt((e ** 2).mean())), 4), pearson=round(float(c), 3) if c == c else None, spearman=round(float(sp), 3) if sp == sp else None, n=len(d))
    te = ref[ref.split == "test"]
    met = pd.DataFrame([dict(model="simulation (held-out)", **metrics(te, "simulated_positive_share")),
                        dict(model="constant mean (held-out)", **metrics(te, "constant_baseline")),
                        dict(model="simulation (train)", **metrics(ref[ref.split == "train"], "simulated_positive_share"))])
    ref.to_csv(os.path.join(out, "evaluation_by_product.csv"), index=False); met.to_csv(os.path.join(out, "evaluation_metrics.csv"), index=False)
    print(f"fitted k={K}, b0={B0:.3f}\n{met.to_string(index=False)}")

    # ---- scenario experiments: chosen product + category average
    pid0 = a.product or ref.sort_values("n_reviews", ascending=False).product_id.iat[0]
    rows, seg = [], []
    for name, sc in SCENARIOS.items():
        p_one = simulate(ag, SAT[pid0], G, NUDGE[pid0], K, B0, sc)
        p_all = np.mean([simulate(ag, SAT[p], G, NUDGE[p], K, B0, sc).mean() for p in ref.product_id])
        rows.append(dict(scenario=name, product=pid0, buy_intention=round(float(p_one.mean()), 4), category_avg_intention=round(float(p_all), 4)))
        for j, nm in enumerate(ag["names"]):
            m = ag["pi"] == j
            if m.any(): seg.append(dict(scenario=name, persona=nm, agents=int(m.sum()), buy_intention=round(float(p_one[m].mean()), 4)))
    S = pd.DataFrame(rows); b = S.iloc[0]
    S["delta_vs_baseline_pts"] = ((S.buy_intention - b.buy_intention) * 100).round(2)
    S["category_delta_pts"] = ((S.category_avg_intention - b.category_avg_intention) * 100).round(2)
    S.to_csv(os.path.join(out, "scenario_summary.csv"), index=False); pd.DataFrame(seg).to_csv(os.path.join(out, "segment_intention.csv"), index=False)
    segdf = pd.DataFrame(seg); sb = segdf[segdf.scenario == "baseline"].set_index("persona").buy_intention

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 3, figsize=(17, 4.5))
        ax[0].bar(S.scenario, S.delta_vs_baseline_pts, color="#4c78a8"); ax[0].set_title(f"Change in buy intention vs baseline (pts)\nproduct {pid0}"); ax[0].tick_params(axis="x", rotation=40)
        for lab, c in (("train", "#bbbbbb"), ("test", "#e45756")):
            d = ref[ref.split == lab]; ax[1].scatter(d.positive_share, d.simulated_positive_share, c=c, label=lab, s=30)
        lim = [ref.positive_share.min() - .01, ref.positive_share.max() + .01]; ax[1].plot(lim, lim, "k--", lw=.8)
        ax[1].set_xlabel("real positive-review share"); ax[1].set_ylabel("simulated"); ax[1].set_title("Simulation vs real, per product"); ax[1].legend()
        sb.sort_values().plot.barh(ax=ax[2], color="#72b7b2"); ax[2].set_title("Baseline intention by persona")
        plt.tight_layout(); plt.savefig(os.path.join(out, "results.png"), dpi=130)
    except ImportError:
        print("matplotlib not installed - skipped chart")

    L = ["# Simulation report", "", f"- Category: {pj['category']} | agents: {a.agents} | products evaluated: {len(ref)} | seed: {a.seed}",
         f"- Real baseline: {ref.positive_share.mean():.1%} of reviews are 4-5 stars (strong positive skew)", "", "## Held-out evaluation (Phase 7)", "", met.to_string(index=False), "",
         "Fitted parameters: intercept and signal-strength k (on training products only). Everything else is fixed.",
         "If the simulation does not beat the constant-mean row on held-out MAE/correlation, say so: the model then explains little product-level variation.", "",
         f"## Scenarios (product {pid0})", "", S.to_string(index=False), "", "## Baseline intention by persona", "", sb.sort_values(ascending=False).to_string(), "",
         "## Assumptions you must state", "", *[f"- {k} = {v}" for k, v in ASSUMED.items()],
         "- Price, discount and competitor-price effects are assumed, not fitted: the review data contains no prices or sales.",
         "- Persona-to-competitor affinity values are assumed. Personas with too little data were skipped: " + (", ".join(skipped) or "none") + ".",
         "- Ground truth is the share of 4-5 star reviews, which measures satisfaction among buyers, not purchase intention of the market.",
         "- Decision model is rule-based; agent text comes from retrieved real reviews, not an LLM (plug an LLM into `simulate` for Milestone 2)."]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(L)); print("\n" + S.to_string(index=False)); print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
