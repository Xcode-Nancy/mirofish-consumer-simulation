"""
Phase 2: schema harmonisation + merging of the three raw Datafiniti files into canonical tables.

    python merge.py --project-dir .          # reads data/raw/*.csv (copied there by phase0_1_inventory.py)

Writes (always csv, so results are identical on every machine):
    data/processed/reviews_canonical      one row per distinct review, with provenance
    data/processed/products_canonical     one row per product (ASIN string), many-to-one with reviews
    reports/merge_audit.csv               row counts before/after every step
    reports/unmatched_records.csv         reviews that could not be attached to a usable product
    reports/conflict_audit.csv            products whose title/brand differ between sources
    reports/schema_mapping.json           original column -> canonical field
Not produced (no supporting data in these files): market_observations (no prices), competitors_canonical.
Raw files are only read, never modified.

Methods and why
  * Schema mapping (original column -> canonical field): one documented, auditable translation instead of ad-hoc renaming.
  * concat for the three files (they are the same logical table); merge only for the product table, with validate="many_to_one"
    and an assert: a join that silently duplicates rows is a classic bug, and this stops it.
  * De-duplicate on (product, user, normalised text) and keep provenance in found_in_sources: exact-key matching is conservative
    (no fuzzy guessing that could merge two different reviews), and provenance means nothing is lost silently.
  * Deterministic sort (completeness, source, row) before dropping duplicates: ties used to be broken arbitrarily, so the surviving
    row, and the product category derived from it, could change between runs and machines.
  * ASIN strings containing commas are kept whole: splitting them could merge distinct product variants.
  * Titles: the canonical title is the most frequent variant, and every variant goes to conflict_audit.csv instead of being overwritten
    silently. About half of the reviews belong to ASINs carrying several titles, so each review also keeps its own title.
  * review_id = hash of (product, user, text): stable across reruns, so results can be compared between machines.
  * Unusable rows are written to unmatched_records.csv, never silently dropped.
"""
import argparse, glob, hashlib, json, os, re
import numpy as np
import pandas as pd

MAPPING = {   # original column -> canonical field
    "asins": "product_id_original", "name": "product_title_raw", "brand": "brand", "manufacturer": "manufacturer",
    "primaryCategories": "category", "categories": "category_path", "keys": "product_keys",
    "reviews.id": "source_review_id", "reviews.date": "review_date", "reviews.rating": "rating",
    "reviews.doRecommend": "do_recommend", "reviews.didPurchase": "verified_purchase",
    "reviews.numHelpful": "helpful_votes", "reviews.title": "review_title", "reviews.text": "review_text",
    "reviews.username": "username"}
log = []

def audit(step, before, after, note=""):
    log.append(dict(step=step, rows_before=before, rows_after=after, change=after - before, note=note))
    print(f"{step:<48} {before:>8,} -> {after:>8,}  {note}")

def squash(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(s))).strip()

def title_from_raw(s):
    if pd.isna(s): return np.nan
    return squash(re.split(r"\r?\n|,,,", str(s))[0]).strip(" ,") or np.nan

def slug_from_keys(k):
    for tok in str(k).split(","):
        slug = tok.split("/")[0]
        if len(slug) >= 6 and slug != "amazon" and re.search(r"[a-z]{4,}", slug):
            return slug
    return np.nan

def mode_or_nan(x):
    x = x.dropna()
    return x.mode().iat[0] if len(x) else np.nan

def save(df, path_no_ext):
    """Always CSV. (Parquet keeps empty strings, CSV turns them into blanks; mixing the two changed the category counts
    between machines, so the pipeline uses one format everywhere.)"""
    df.to_csv(path_no_ext + ".csv", index=False); return path_no_ext + ".csv"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project-dir", default="."); ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args(); P = lambda *x: os.path.join(a.project_dir, *x)
    raw = a.raw_dir or P("data", "raw")
    for d in ["data/processed", "reports"]: os.makedirs(P(d), exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(raw, "*.csv")) if "clean" not in os.path.basename(f).lower())
    if not files: raise SystemExit(f"No raw CSVs in {raw}")

    # ---- 1. load + rename (original ids kept as *_original)
    parts = []
    for i, f in enumerate(files, 1):
        d = pd.read_csv(f, low_memory=False)
        d = d[[c for c in MAPPING if c in d.columns]].rename(columns=MAPPING)
        d["source_id"] = f"S{i:02d}"; d["source_row"] = np.arange(len(d))
        audit(f"load {os.path.basename(f)[:30]}", 0, len(d)); parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    audit("concat (same logical table: reviews)", sum(len(p) for p in parts), len(df))

    # ---- 2. identifiers: normalise without destroying originals
    df["product_id"] = df["product_id_original"].astype("string").str.strip().str.upper()
    df.loc[df["product_id"].isin(["", "NAN", "<NA>"]), "product_id"] = pd.NA
    df["product_title_clean"] = df["product_title_raw"].map(title_from_raw)

    # ---- 3. review text / rating / date typing (original text preserved)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_title"] = df["review_title"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    df["do_recommend"] = df["do_recommend"].astype(str).str.lower().map({"true": True, "false": False})
    df["verified_purchase"] = df.get("verified_purchase", pd.Series(index=df.index, dtype=object)).astype(str).str.lower().map({"true": True, "false": False})
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce")
    df["username"] = df["username"].fillna("").astype(str)
    df["_text_key"] = df["review_text"].map(squash).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

    # ---- 4. unmatched / unusable records (kept in audit, not silently dropped)
    bad = df["product_id"].isna() | (df["_text_key"].str.len() == 0) | df["rating"].isna() | ~df["rating"].between(1, 5)
    reason = np.select([df["product_id"].isna(), df["_text_key"].str.len() == 0, df["rating"].isna(), ~df["rating"].between(1, 5)],
                       ["missing product id", "empty review text", "missing rating", "rating outside 1-5"], "")
    unmatched = df.loc[bad, ["source_id", "source_row", "product_id_original", "rating", "review_text"]].assign(reason=reason[bad.values])
    unmatched["review_text"] = unmatched["review_text"].str[:120]
    unmatched.to_csv(P("reports", "unmatched_records.csv"), index=False)
    n = len(df); df = df[~bad].copy(); audit("exclude unusable reviews (see unmatched_records.csv)", n, len(df))

    # ---- 5. duplicates across/within sources: keep one row, remember every source it appeared in
    df["_filled"] = df.notna().sum(axis=1)
    df = df.sort_values(["_filled", "source_id", "source_row"], ascending=[False, True, True], kind="stable")   # deterministic: ties broken by source/row
    key = ["product_id", "username", "_text_key"]
    found = df.groupby(key, dropna=False)["source_id"].agg(lambda s: "|".join(sorted(set(s)))).rename("found_in_sources")
    n = len(df); df = df.drop_duplicates(key, keep="first").merge(found, on=key, how="left", validate="one_to_one")
    audit("deduplicate (product + user + text) across sources", n, len(df), "found_in_sources keeps provenance")

    # ---- 6. product master table (many-to-one with reviews)
    g = df.groupby("product_id")
    prod = pd.DataFrame({
        "product_title": g["product_title_clean"].agg(mode_or_nan),
        "brand": g["brand"].agg(mode_or_nan), "manufacturer": g["manufacturer"].agg(mode_or_nan),
        "category": g["category"].agg(mode_or_nan), "category_path": g["category_path"].agg(mode_or_nan),
        "n_title_variants": g["product_title_clean"].nunique(), "n_brand_variants": g["brand"].nunique(),
        "sources": g["source_id"].agg(lambda s: "|".join(sorted(set(s)))),
        "product_keys": g["product_keys"].agg(mode_or_nan)}).reset_index()
    prod["title_source"] = np.where(prod["product_title"].notna(), "source_title", "")
    miss = prod["product_title"].isna()
    prod.loc[miss, "product_title"] = prod.loc[miss, "product_keys"].map(slug_from_keys)
    prod.loc[miss & prod["product_title"].notna(), "title_source"] = "recovered_from_keys (slug, not a real title)"
    prod["title_source"] = prod["title_source"].replace("", "UNKNOWN")
    prod["category"] = prod["category"].fillna(prod["category_path"].astype(str).str.split(",").str[0].replace("nan", np.nan))
    audit("build products master", df["product_id"].nunique(), len(prod), "one row per product_id")
    assert prod["product_id"].is_unique

    # conflicts between sources (title/brand variants) -> audit, not silent overwrite
    var = (df.dropna(subset=["product_title_clean"]).groupby(["product_id", "product_title_clean"])["source_id"]
             .agg(lambda s: "|".join(sorted(set(s)))).reset_index())
    multi = var.groupby("product_id")["product_title_clean"].transform("nunique") > 1
    var[multi].rename(columns={"product_title_clean": "title_variant", "source_id": "seen_in"}).to_csv(P("reports", "conflict_audit.csv"), index=False)
    log.append(dict(step="title conflicts", rows_before="", rows_after=int(var[multi].product_id.nunique()), change="",
                    note="products with >1 distinct title variant; canonical = most frequent (see conflict_audit.csv)"))

    # ---- 7. validated join: reviews -> product attributes (must not multiply rows)
    n = len(df)
    rev = df.merge(prod[["product_id", "product_title", "brand", "category"]].rename(columns={"brand": "brand_p", "category": "category_p", "product_title": "product_title_canonical"}),
                   on="product_id", how="left", validate="many_to_one")
    assert len(rev) == n, "merge multiplied rows"
    audit("join reviews -> products (many_to_one)", n, len(rev), f"{int(rev.product_title_canonical.isna().sum())} reviews without title")

    rev.insert(0, "review_id", ["R" + hashlib.sha1(f"{p}|{u}|{t}".encode()).hexdigest()[:12] for p, u, t in zip(rev.product_id, rev.username, rev._text_key)])
    assert rev["review_id"].is_unique, "review_id collision"
    cols = ["review_id", "product_id", "product_id_original", "product_title_canonical", "brand_p", "category_p", "review_title",
            "review_text", "rating", "review_date", "do_recommend", "verified_purchase", "helpful_votes", "username",
            "source_review_id", "source_id", "source_row", "found_in_sources", "product_title_clean"]
    rev = rev[cols].rename(columns={"brand_p": "brand", "category_p": "category", "product_title_clean": "review_source_title"}).sort_values(["product_id", "review_date"]).reset_index(drop=True)
    rev["rating"] = rev["rating"].astype(int)

    paths = [save(rev, P("data", "processed", "reviews_canonical")), save(prod, P("data", "processed", "products_canonical"))]
    pd.DataFrame(log).to_csv(P("reports", "merge_audit.csv"), index=False)
    json.dump(dict(mapping=MAPPING, notes={
        "product_id": "ASIN string as given (comma-separated variant lists kept whole to avoid merging distinct variants), stripped+uppercased",
        "product_id_original": "untouched original value",
        "review_id": "sha1(product_id|username|normalised text)[:12], stable across reruns",
        "not_available": ["price", "discount", "competitor", "product description/specifications", "launch date"]}),
        open(P("reports", "schema_mapping.json"), "w"), indent=2)
    print("\nSaved:", *paths, sep="\n  ")
    print(f"\nFINAL: {len(rev):,} reviews | {len(prod)} products | {rev.brand.nunique()} brands | "
          f"{int((rev.found_in_sources.str.contains('|', regex=False)).sum()):,} reviews present in >1 source")

if __name__ == "__main__":
    main()