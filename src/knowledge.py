"""
Phase 3: category filter, aspect extraction, retrieval index, knowledge graph, persona calibration.

    python phase3_build.py --project-dir . --category fire_tablet --list-price 49.99

Reads  data/processed/reviews_canonical.(parquet|csv)   (produced by merge.py)
Writes data/processed/category_reviews.csv      reviews of the chosen category, cleaned, with aspect columns
       data/processed/aspect_mentions.csv       one row per (review, aspect) with sentiment
       data/processed/retrieval_index.pkl    BM25 index used as the RAG layer
       data/processed/knowledge_graph.json      product-aspect graph (node-link JSON)
       data/processed/personas.json             10 personas calibrated from real review data
       data/processed/seed_config.json          scenario baseline (price is an ASSUMPTION, not in the data)
       data/processed/reference_outcomes.csv    real outcomes per product, for Phase 7 (MAE/RMSE/correlation)
       reports/category_counts.csv, reports/phase3_summary.md
Retrieval helper for the simulation:   from phase3_build import Retriever; Retriever(path).search("battery life", k=5)
Install: python -m pip install pandas numpy   (no scikit-learn / networkx needed)

Methods and why
  * Category from the review's OWN title (not the ASIN's): about half of the reviews sit on ASINs that carry several different product
    titles, so ASIN-level categories would mix tablets, speakers and accessories. Reviews with no own title fall back to the ASIN title
    and are flagged. Rules are checked in a fixed order (accessories first) so "tablet case" is not counted as a tablet.
  * Rule-based aspect lexicon (14 aspects): transparent and editable, needs no labelled training data and no heavy libraries.
    Limitation: keyword rules miss paraphrases.
  * Sentiment = small word list + negation window (3 tokens before a word flips it) + star-rating fallback when the sentence is neutral:
    fast and explainable. Limitation: the fallback means part of the aspect sentiment is derived from the stars, which is why the
    evaluation uses a split-half design (see evaluate.py).
  * Retrieval = BM25 written in plain numpy (k1=1.5, b=0.75, the usual defaults): a strong lexical baseline for RAG, fully
    interpretable, and it avoids scikit-learn (no installable build on the newest Python versions). The Retriever class keeps the same
    .search() interface so embeddings + a vector database can replace it in Milestone 2.
  * Knowledge graph as plain JSON (product -> aspect edges with mention counts and mean sentiment): only relations supported by
    evidence are stored, and no graph library is needed.
  * Personas = keyword clusters over reviews (influencer = top 10% helpful votes, reviewer = top 10% review length). Share, aspect
    importance, satisfaction and buy propensity are COMPUTED from the data, not invented. A persona with fewer than 30 reviews is flagged
    LOW DATA instead of receiving made-up numbers. Personas overlap (one review can match several), so shares do not sum to 100%.
  * reference_outcomes.csv: real per-product outcomes kept aside as ground truth for the evaluation.
"""
import argparse, json, os, re, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd

# ---------- 1. category rules (applied to the review's OWN source title; accessories are checked first)
CATEGORY_RULES = [
    ("accessory", r"\b(cover|case|charger|adapter|adaptor|power ?adapter|sleeve|protector|stand|cable|skin|screen protector|usb)\b"),
    ("fire_tv", r"\bfire ?tv\b|\bfire ?stick\b|\bfire tv stick\b"),
    ("echo_speaker", r"\b(echo|amazon tap|tap\b|dot\b|alexa|speaker)\b"),
    ("fire_tablet", r"\bfire\b.*\b(tablet|hd ?\d|hdx|7|8|10|kids)\b|\bkindle fire\b|\bfire (hd|hdx|7|8|10)\b|\btablet\b"),
    ("kindle_ereader", r"\b(kindle|e-?reader|paperwhite|voyage|oasis)\b"),
]
def classify(title):
    t = str(title).lower()
    for name, pat in CATEGORY_RULES:
        if re.search(pat, t): return name
    return "other"

# ---------- 2. aspect lexicon + tiny sentiment lexicon
ASPECTS = {
    "battery": r"battery|charge[sd]?\b|charging|lasts? (all|long|days)",
    "screen": r"screen|display|resolution|brightness|glare|pixels?",
    "price": r"price|cheap|afford|value|worth|bargain|money|cost|\$\d+|expensive|deal",
    "apps": r"\bapps?\b|google play|app ?store|appstore|software|android",
    "kids": r"\bkids?\b|child|children|grand(son|daughter|kids?)|my (son|daughter)|toddler",
    "camera": r"camera|photos?|pictures?|selfie",
    "speed": r"fast|slow|speed|lag|laggy|performance|processor|freez|responsive|snappy",
    "sound": r"sound|speaker|audio|volume|music",
    "ease_of_use": r"easy to use|easy|simple|intuitive|user[- ]friendly|set ?up|interface|navigate",
    "durability": r"durab|sturdy|broke|broken|crack|died|stopped working|defect|build quality|flimsy",
    "games": r"\bgames?\b|gaming|play(ing)? (games|minecraft)",
    "video": r"video|movies?|netflix|prime video|stream|youtube|hulu",
    "reading": r"read(ing|er)?\b|books?|ebook|kindle (unlimited|store)",
    "support": r"customer (service|support)|warranty|refund|return|replacement|amazon (service|support)",
}
POS = set("good great excellent love loved loves perfect amazing awesome fantastic nice happy easy fast best works well wonderful recommend pleased satisfied outstanding sturdy solid smooth bright crisp worth cheap affordable impressed useful reliable".split())
NEG = set("bad poor terrible awful hate hated slow broke broken died useless waste disappointed disappointing problem problems issue issues difficult hard worse worst horrible junk defective cheaply flimsy freezes freezing laggy cracked glare expensive overpriced unreliable frustrating".split())
NEGATORS = {"not", "no", "never", "n't", "dont", "doesnt", "didnt", "isnt", "wasnt", "cant", "wont", "hardly", "barely"}

def sent_score(sentence):
    toks = re.findall(r"[a-z']+", sentence.lower()); s = 0
    for i, t in enumerate(toks):
        v = 1 if t in POS else -1 if t in NEG else 0
        if v and any(x in NEGATORS or x.endswith("n't") for x in toks[max(0, i - 3):i]): v = -v
        s += v
    return int(np.sign(s))

def squash(s): return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(s))).strip()
def split_sentences(t): return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", t) if len(x.strip()) > 3]

def extract_aspects(row):
    out = []
    rating_sign = 1 if row.rating >= 4 else -1 if row.rating <= 2 else 0
    for sent in split_sentences(row.clean_text):
        low = sent.lower(); sc = None
        for a, pat in ASPECTS.items():
            if re.search(pat, low):
                if sc is None: sc = sent_score(sent)
                s = sc if sc != 0 else rating_sign          # fall back to the star rating if the sentence is neutral
                out.append((row.review_id, row.product_id, a, s, "lexicon" if sc != 0 else "rating", sent[:240]))
    return out

# ---------- 3. persona definitions (keyword clusters over reviews; data decides weights, shares, satisfaction)
PERSONAS = {
    "budget_conscious": r"cheap|budget|afford|value|bargain|worth the (money|price)|low price|inexpensive",
    "technology_enthusiast": r"processor|\bram\b|resolution|root(ed|ing)?|sd card|micro ?sd|specs?|google play|sideload|android|cpu",
    "professional_user": r"\bwork\b|office|business|email|productiv|meeting|presentation|document|travel for",
    "brand_loyal": r"another (kindle|fire|amazon)|second (kindle|fire)|upgrade|replace(d|ment)? (my|our|an old)|always (buy|use)|love amazon|have (several|many|three|two) (kindles|fires)",
    "price_sensitive": r"expensive|overpriced|too much|price (went|drop)|on sale|sale|discount|deal|coupon|\$\d+",
    "camera_focused": r"camera|photos?|pictures?|selfie|front[- ]facing|megapixel",
    "gamer": r"\bgames?\b|gaming|minecraft|fortnite|play(ing)? games|graphics",
    "influencer": None,          # defined by helpful votes (top decile), not keywords
    "product_reviewer": None,    # defined by review length (top decile), not keywords
    "competitor_customer": r"ipad|samsung|galaxy|nexus|google nexus|apple|iphone|nook|kobo|lenovo|asus|compared to (my|the) (ipad|samsung)|switch(ed)? from",
}

def persona_flags(df):
    f = pd.DataFrame(index=df.index)
    txt = df.clean_text.str.lower()
    for p, pat in PERSONAS.items():
        if pat: f[p] = txt.str.contains(pat, regex=True)
    hv = pd.to_numeric(df.helpful_votes, errors="coerce").fillna(0)
    f["influencer"] = hv >= max(1, hv.quantile(0.90))
    ln = df.clean_text.str.len()
    f["product_reviewer"] = ln >= ln.quantile(0.90)
    return f[list(PERSONAS)]

TOK = re.compile(r"[a-z0-9']+")
STOP = set("the a an and or of to in is it this that for on with was are be as at my i you have has had but not so very it's".split())
def tokenize(t): return [w for w in TOK.findall(str(t).lower()) if w not in STOP and len(w) > 1]

def build_bm25(texts, k1=1.5, b=0.75):
    """BM25 inverted index in pure numpy (no scikit-learn needed). term -> (doc ids, precomputed scores)."""
    post, lens = {}, np.zeros(len(texts))
    for i, t in enumerate(texts):
        toks = tokenize(t); lens[i] = len(toks); c = {}
        for w in toks: c[w] = c.get(w, 0) + 1
        for w, n in c.items():
            ids, tfs = post.setdefault(w, ([], [])); ids.append(i); tfs.append(n)
    N, avg, index = len(texts), max(lens.mean(), 1), {}
    for w, (ids, tfs) in post.items():
        ids = np.array(ids); tf = np.array(tfs, float)
        idf = np.log(1 + (N - len(ids) + 0.5) / (len(ids) + 0.5))
        index[w] = (ids, idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[ids] / avg)))
    return index

class Retriever:
    """Thin RAG layer: BM25 over review text. Replace with embeddings + vector DB later (same .search interface)."""
    def __init__(self, path):
        import pickle
        with open(path, "rb") as f: o = pickle.load(f)
        self.index, self.meta = o["index"], o["meta"]
    def search(self, query, k=5, product_id=None):
        sc = np.zeros(len(self.meta))
        for w in tokenize(query):
            if w in self.index: ids, s = self.index[w]; sc[ids] += s
        if product_id is not None: sc = np.where(self.meta.product_id.values == product_id, sc, -1)
        idx = np.argsort(-sc)[:k]
        return self.meta.iloc[idx].assign(score=sc[idx])[["review_id", "product_id", "rating", "text", "score"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="."); ap.add_argument("--category", default="fire_tablet")
    ap.add_argument("--list-price", type=float, default=49.99, help="ASSUMED public list price (not in the data); verify and cite")
    ap.add_argument("--min-reviews", type=int, default=30)
    a = ap.parse_args(); P = lambda *x: os.path.join(a.project_dir, *x)
    os.makedirs(P("reports"), exist_ok=True); os.makedirs(P("data", "processed"), exist_ok=True)

    src = next((p for p in (P("data/processed/reviews_canonical.csv"), P("data/processed/reviews_canonical.parquet")) if os.path.exists(p)), None)
    if not src: raise SystemExit("Run merge.py first (reviews_canonical not found).")
    df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src, low_memory=False)
    print(f"loaded {len(df):,} reviews from {src}")

    # ---- category: use the review's own title; fall back to the ASIN's canonical title (flagged)
    title_col = "review_source_title" if "review_source_title" in df.columns else None
    if title_col is None: print("WARNING: review_source_title missing - re-run the patched merge.py; falling back to canonical ASIN title.")
    own = df[title_col] if title_col else pd.Series(np.nan, index=df.index)
    df["title_used"] = own.fillna(df["product_title_canonical"])
    df["title_confidence"] = np.where(own.notna(), "review_level", "asin_fallback")
    df["category_class"] = df["title_used"].map(classify)
    cc = df.groupby(["category_class", "title_confidence"]).agg(reviews=("review_id", "size"), products=("product_id", "nunique"),
                                                               mean_rating=("rating", "mean")).round(3).reset_index()
    cc.to_csv(P("reports", "category_counts.csv"), index=False); print(cc.to_string(index=False))

    d = df[df.category_class == a.category].copy()
    if d.empty: raise SystemExit(f"No reviews for category '{a.category}'. See reports/category_counts.csv")
    d["clean_text"] = (d["review_title"].fillna("").map(squash) + ". " + d["review_text"].fillna("").map(squash)).str.strip(". ")
    d = d[d.clean_text.str.len() >= 15].reset_index(drop=True)
    print(f"\ncategory '{a.category}': {len(d):,} reviews, {d.product_id.nunique()} ASINs")

    # ---- aspects
    rows = [m for r in d.itertuples() for m in extract_aspects(r)]
    am = pd.DataFrame(rows, columns=["review_id", "product_id", "aspect", "sentiment", "sentiment_source", "sentence"])
    am.to_csv(P("data", "processed", "aspect_mentions.csv"), index=False)
    for asp in ASPECTS: d["asp_" + asp] = d.review_id.isin(am.loc[am.aspect == asp, "review_id"])
    flags = persona_flags(d)
    d = pd.concat([d, flags.add_prefix("persona_")], axis=1)
    d.drop(columns=["review_text"]).to_csv(P("data", "processed", "category_reviews.csv"), index=False)

    # ---- retrieval index (BM25, pure numpy)
    import pickle
    meta = d[["review_id", "product_id", "rating"]].assign(text=d.clean_text.str[:600])
    with open(P("data", "processed", "retrieval_index.pkl"), "wb") as f:
        pickle.dump(dict(index=build_bm25(d.clean_text.tolist()), meta=meta), f)

    # ---- knowledge graph (product -> aspect edges weighted by mentions and mean sentiment), plain JSON
    nodes = [dict(id=f"P:{pid}", type="product", title=str(g.title_used.mode().iat[0])[:90], n_reviews=int(len(g)), mean_rating=round(float(g.rating.mean()), 3))
             for pid, g in d.groupby("product_id")] + [dict(id=f"A:{k}", type="aspect") for k in ASPECTS]
    edges = [dict(source=f"P:{pid}", target=f"A:{asp}", relation="mentioned_with", weight=int(len(g)), mean_sentiment=round(float(g.sentiment.mean()), 3))
             for (pid, asp), g in am.groupby(["product_id", "aspect"])]
    json.dump(dict(nodes=nodes, edges=edges), open(P("data", "processed", "knowledge_graph.json"), "w"), indent=1)

    # ---- persona calibration from data
    base_buy = float((d.rating >= 4).mean())
    asp_imp_all = {k: float(d["asp_" + k].mean()) for k in ASPECTS}
    personas, N = {}, len(d)
    for p in PERSONAS:
        m = d["persona_" + p]; sub = d[m]
        if len(sub) < a.min_reviews:
            personas[p] = dict(n_reviews=int(len(sub)), low_data=True, share=None); continue
        imp = {k: float(sub["asp_" + k].mean()) for k in ASPECTS}
        ams = am[am.review_id.isin(sub.review_id)]
        sat = ams.groupby("aspect").sentiment.mean().round(3).to_dict()
        lift = {k: round(imp[k] / asp_imp_all[k], 2) if asp_imp_all[k] > 0 else None for k in ASPECTS}
        rec = sub.do_recommend.dropna() if "do_recommend" in sub else pd.Series(dtype=float)
        personas[p] = dict(n_reviews=int(len(sub)), low_data=False, share=round(len(sub) / N, 4),
                           buy_propensity=round(float((sub.rating >= 4).mean()), 3), mean_rating=round(float(sub.rating.mean()), 3),
                           recommend_rate=round(float(rec.astype(float).mean()), 3) if len(rec) else None,
                           aspect_importance={k: round(v, 3) for k, v in imp.items()}, aspect_lift_vs_all=lift, aspect_satisfaction=sat,
                           top_aspects=sorted(imp, key=imp.get, reverse=True)[:4])
    json.dump(dict(category=a.category, baseline_buy_propensity=round(base_buy, 3), overall_aspect_importance=asp_imp_all, personas=personas),
              open(P("data", "processed", "personas.json"), "w"), indent=2)

    json.dump(dict(category=a.category, baseline_list_price_usd=a.list_price,
                   price_note="ASSUMPTION: the review files contain no price. Verify the list price for the product/year and cite its source.",
                   scenarios={"baseline": {"price_mult": 1.0}, "discount_15pct": {"price_mult": 0.85}, "price_up_20pct": {"price_mult": 1.2},
                              "stronger_marketing": {"marketing": 1.0}, "competitor_price_cut": {"competitor_price_mult": 0.85}}),
              open(P("data", "processed", "seed_config.json"), "w"), indent=2)

    # ---- real outcomes per product (ground truth for Phase 7)
    ref = d.groupby("product_id").agg(n_reviews=("review_id", "size"), mean_rating=("rating", "mean"),
                                      positive_share=("rating", lambda s: (s >= 4).mean()), negative_share=("rating", lambda s: (s <= 2).mean()),
                                      recommend_rate=("do_recommend", lambda s: s.dropna().astype(float).mean() if s.notna().any() else np.nan)).round(4).reset_index()
    ref[ref.n_reviews >= a.min_reviews].to_csv(P("data", "processed", "reference_outcomes.csv"), index=False)

    top = am.groupby("aspect").agg(mentions=("sentiment", "size"), mean_sentiment=("sentiment", "mean")).sort_values("mentions", ascending=False).round(3)
    lines = [f"# Phase 3 summary ({a.category})", "", f"- Reviews: {len(d):,} | ASINs: {d.product_id.nunique()} | positive (4-5 stars): {base_buy:.1%}",
             f"- Titles from review level: {(d.title_confidence=='review_level').mean():.1%}; ASIN fallback: {(d.title_confidence=='asin_fallback').mean():.1%}",
             f"- Aspect mentions: {len(am):,} (sentiment from lexicon {(am.sentiment_source=='lexicon').mean():.1%}, from star rating the rest)",
             "", "## Aspects", "", top.to_string(), "", "## Personas", ""]
    for p, v in personas.items():
        lines.append(f"- {p}: " + ("LOW DATA (n=%d)" % v["n_reviews"] if v["low_data"] else f"n={v['n_reviews']}, share={v['share']:.1%}, positive={v['buy_propensity']:.1%}, top aspects={v['top_aspects']}"))
    open(P("reports", "phase3_summary.md"), "w", encoding="utf-8").write("\n".join(lines)); print("\n".join(lines[:12])); print("\nDone. See reports/phase3_summary.md")

if __name__ == "__main__":
    main()