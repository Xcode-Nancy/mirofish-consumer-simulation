# Data inventory (Phase 1)

Generated 2026-11-14. Raw files copied unchanged to `data/raw/`; hashes in `data/source_registry.csv`.

## Files

| source | file | size MB | rows | cols | ASINs | names | brands | dates | rating range | exact dup rows |
|---|---|---|---|---|---|---|---|---|---|---|
| S01 | 1429_1.csv | 46.72 | 34,660 | 21 | 41 | 48 | 6 | 2011-11-18 to 2018-04-18 | 1.0-5.0 | 0 |
| S02 | Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv | 94.95 | 5,000 | 24 | 24 | 23 | 1 | 2014-10-24 to 2018-09-15 | 1-5 | 95 |
| S03 | Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv | 253.34 | 28,332 | 24 | 65 | 65 | 3 | 2009-02-26 to 2019-03-25 | 1-5 | 0 |

## Overlap between files

- S01 vs S02: 1,995 identical reviews (same ASIN + user + text)
- S01 vs S03: 3,550 identical reviews (same ASIN + user + text)
- S02 vs S03: 2,161 identical reviews (same ASIN + user + text)

## Price availability

- S01: price columns = NONE
- S02: price columns = NONE
- S03: price columns = NONE

## Highest-missingness columns (any source)

| source_id   | column               |   missing_pct |
|:------------|:---------------------|--------------:|
| S01         | reviews.id           |        100    |
| S01         | reviews.userProvince |        100    |
| S01         | reviews.userCity     |        100    |
| S01         | reviews.didPurchase  |        100    |
| S03         | reviews.didPurchase  |         99.97 |
| S03         | reviews.id           |         99.86 |
| S02         | reviews.id           |         99.42 |
| S02         | reviews.dateAdded    |         78.96 |
| S03         | reviews.doRecommend  |         43.22 |
| S03         | reviews.numHelpful   |         43.12 |
| S01         | reviews.dateAdded    |         30.64 |
| S01         | name                 |         19.5  |

## Still to verify manually

- Exact source URL, license and version of each file (registry fields marked NOT VERIFIED).
- Whether the files match the ~34,000-review dataset named in the proposal.