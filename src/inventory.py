"""
Phase 0 + Phase 1: environment check, raw-data inventory, profiling, source registry.

    python src/data/phase0_1_inventory.py --raw-dir dataset --project-dir .

Reads the original CSVs (never modifies them), copies them untouched into data/raw/ with SHA-256 hashes,
and writes:
    data/source_registry.csv
    reports/environment.md
    reports/data_inventory.md
    reports/initial_data_profile.csv     (one row per source x column)
    reports/file_summary.csv             (one row per source)

Methods and why
  * SHA-256 hash of every raw file: anyone can later prove the raw data was never edited (provenance, reproducibility).
  * Raw files are copied, never modified: all cleaning happens in later, separate steps, so mistakes can always be undone.
  * Column profile (type, % missing, unique values): decides which columns are usable. Several columns here are ~100% empty.
  * Overlap check on (ASIN, user, text): measures how many reviews repeat across the three files, which justifies the
    de-duplication rule used in merge.py.
  * Search for price columns: tells us whether price experiments can be learned from data (here they cannot).
"""
import argparse, glob, hashlib, os, platform, shutil, sys, datetime as dt
import importlib.metadata as md
import pandas as pd

PKGS = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "transformers", "torch", "gradio",
        "networkx", "sentence-transformers", "langchain", "langgraph", "openai", "jupyter", "pyarrow"]

def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

def environment_report():
    lines = ["# Environment report", "", f"- Python: {sys.version.split()[0]}", f"- OS: {platform.platform()}",
             f"- CPU cores: {os.cpu_count()}"]
    try:
        import psutil
        lines.append(f"- RAM: {psutil.virtual_memory().total / 2**30:.1f} GB")
    except ImportError:
        lines.append("- RAM: (install psutil to report)")
    free = shutil.disk_usage(".").free / 2**30
    lines.append(f"- Free disk: {free:.1f} GB")
    try:
        import torch
        lines.append(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none detected'}")
    except ImportError:
        lines.append("- GPU: torch not installed (cannot detect)")
    lines += ["- LLM API key set: " + ("yes" if os.getenv("OPENAI_API_KEY") else "no (OPENAI_API_KEY not found)"),
              "", "## Packages", ""]
    for p in PKGS:
        try: lines.append(f"- {p}: {md.version(p)}")
        except md.PackageNotFoundError: lines.append(f"- {p}: NOT INSTALLED")
    return "\n".join(lines)

def profile_file(path, source_id):
    df = pd.read_csv(path, low_memory=False)
    col_rows = []
    for c in df.columns:
        s = df[c]
        try: nun = int(s.nunique(dropna=True))
        except TypeError: nun = -1                      # unhashable values
        col_rows.append(dict(source_id=source_id, column=c, dtype=str(s.dtype), missing_n=int(s.isna().sum()),
                             missing_pct=round(100 * s.isna().mean(), 2), n_unique=nun))
    # dates
    dcol = next((c for c in ("reviews.date", "reviews.dateAdded") if c in df.columns), None)
    dates = pd.to_datetime(df[dcol], errors="coerce", utc=True) if dcol else pd.Series(dtype="datetime64[ns, UTC]")
    rcol = "reviews.rating" if "reviews.rating" in df.columns else None
    price_cols = [c for c in df.columns if "price" in c.lower()]
    summary = dict(
        source_id=source_id, file=os.path.basename(path), size_mb=round(os.path.getsize(path) / 2**20, 2),
        rows=len(df), columns=df.shape[1], memory_mb=round(df.memory_usage(deep=True).sum() / 2**20, 1),
        unique_asins=df["asins"].nunique() if "asins" in df else None,
        unique_names=df["name"].nunique() if "name" in df else None,
        unique_brands=df["brand"].nunique() if "brand" in df else None,
        unique_review_ids=df["reviews.id"].nunique() if "reviews.id" in df else None,
        date_column=dcol, date_min=str(dates.min())[:10] if len(dates.dropna()) else None,
        date_max=str(dates.max())[:10] if len(dates.dropna()) else None,
        rating_min=df[rcol].min() if rcol else None, rating_max=df[rcol].max() if rcol else None,
        exact_duplicate_rows=int(df.astype(str).duplicated().sum()),
        price_columns=";".join(price_cols) or "NONE")
    return df, summary, col_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True); ap.add_argument("--project-dir", default=".")
    a = ap.parse_args()
    P = lambda *x: os.path.join(a.project_dir, *x)
    for d in ["data/raw", "data/interim", "data/processed", "data/external", "data/reference", "reports", "src/data", "outputs"]:
        os.makedirs(P(d), exist_ok=True)

    open(P("reports", "environment.md"), "w", encoding="utf-8").write(environment_report())

    files = sorted(glob.glob(os.path.join(a.raw_dir, "*.csv")))
    files = [f for f in files if "clean" not in os.path.basename(f).lower()]       # skip derived files
    if not files: raise SystemExit(f"No raw CSV files found in {a.raw_dir}")
    registry, summaries, cols = [], [], []
    for i, f in enumerate(files, 1):
        sid = f"S{i:02d}"
        dst = P("data/raw", os.path.basename(f))
        if not os.path.exists(dst): shutil.copy2(f, dst)                           # raw copy, never edited
        df, s, c = profile_file(dst, sid)
        summaries.append(s); cols += c
        registry.append(dict(source_id=sid, source_name="Datafiniti Consumer Reviews of Amazon Products",
            source_url="NOT VERIFIED - add the page you downloaded it from (e.g. Kaggle/Datafiniti)",
            access_date=dt.date.today().isoformat(), file_path=dst.replace("\\", "/"),
            sha256=sha256(dst), license="NOT VERIFIED - check the dataset page before publishing",
            collection_method="manual download (provided by team)", dataset_version=os.path.basename(f),
            date_coverage=f"{s['date_min']} to {s['date_max']}", product_coverage=f"{s['unique_asins']} ASINs",
            notes=f"{s['rows']} rows; price columns: {s['price_columns']}"))
    pd.DataFrame(registry).to_csv(P("data/source_registry.csv"), index=False)
    pd.DataFrame(cols).to_csv(P("reports/initial_data_profile.csv"), index=False)
    S = pd.DataFrame(summaries); S.to_csv(P("reports/file_summary.csv"), index=False)

    # cross-file overlap
    raws = {s["source_id"]: pd.read_csv(P("data/raw", s["file"]), low_memory=False, usecols=lambda c: c in ("asins", "reviews.text", "reviews.username")) for s in summaries}
    keyed = {k: set(zip(v["asins"].astype(str), v["reviews.username"].astype(str), v["reviews.text"].astype(str))) for k, v in raws.items()}
    ids = list(keyed); ov = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ov.append(f"- {ids[i]} vs {ids[j]}: {len(keyed[ids[i]] & keyed[ids[j]]):,} identical reviews (same ASIN + user + text)")

    md_lines = ["# Data inventory (Phase 1)", "", f"Generated {dt.date.today()}. Raw files copied unchanged to `data/raw/`; hashes in `data/source_registry.csv`.", "",
                "## Files", "", "| source | file | size MB | rows | cols | ASINs | names | brands | dates | rating range | exact dup rows |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        md_lines.append(f"| {s['source_id']} | {s['file']} | {s['size_mb']} | {s['rows']:,} | {s['columns']} | {s['unique_asins']} | {s['unique_names']} | {s['unique_brands']} | {s['date_min']} to {s['date_max']} | {s['rating_min']}-{s['rating_max']} | {s['exact_duplicate_rows']:,} |")
    md_lines += ["", "## Overlap between files", ""] + ov
    md_lines += ["", "## Price availability", "", *[f"- {s['source_id']}: price columns = {s['price_columns']}" for s in summaries],
                 "", "## Highest-missingness columns (any source)", "",
                 pd.DataFrame(cols).sort_values("missing_pct", ascending=False).head(12)[["source_id", "column", "missing_pct"]].to_markdown(index=False) if _has_tab() else "(install tabulate to show table)",
                 "", "## Still to verify manually", "",
                 "- Exact source URL, license and version of each file (registry fields marked NOT VERIFIED).",
                 "- Whether the files match the ~34,000-review dataset named in the proposal."]
    open(P("reports/data_inventory.md"), "w", encoding="utf-8").write("\n".join(md_lines))
    print("\n".join(md_lines)); print("\nDone. Files written under", a.project_dir)

def _has_tab():
    try: import tabulate; return True
    except ImportError: return False

if __name__ == "__main__":
    main()
