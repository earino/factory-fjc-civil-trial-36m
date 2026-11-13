---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-86c6129e8102da78**

[Dataset website](https://earino.github.io/factory-fjc-civil-trial-36m/versions/v-86c6129e8102da78/) · [GitHub release](https://github.com/earino/factory-fjc-civil-trial-36m/releases/tag/v-86c6129e8102da78) · [Hugging Face](https://huggingface.co/datasets/earino/factory-fjc-civil-trial-36m/tree/v-86c6129e8102da78) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: 12c60dcef222d140c644711233662e099b37df1325fe5ad1c96ff2d94d0aeff5
<!-- /dataset-factory:links -->

# Will a newly filed federal civil case reach trial within 36 months? Forward trial prediction from FJC Integrated Database filing records

Will a newly filed federal civil case reach trial within 36 months?

Derived from Federal Judicial Center (FJC), U.S. federal judiciary. This is not its official product.

Prediction time: At case filing (the filing date, which is also the carried time column), using only civil cover-sheet and case-opening fields; no docket events after filing.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.7345856253648118,
    "brier": 0.005388478583624853
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.005019440674730971
  },
  "calendar_identity": {
    "auc": 0.49254135370450874,
    "brier": 0.005020267645680341
  }
}
```

## Limitations

- Current-vintage retrospective: the IDB is a 2026 snapshot and records may be overwritten within the reporting year, so filing-time fields such as JURY may reflect later corrections; this does not prove historical operational availability.
- Trial is rare (roughly 0.6-1% of civil filings), so the release is strongly class-imbalanced and should be evaluated with precision/recall or PR-AUC rather than accuracy.
- PROCPROG is missing for some terminated records; missing is treated as no trial, which may undercount trials.
- Trial start dates are not usable because TRIBEGAN and TRIALEND are no longer maintained; the 36-month horizon is measured from TERMDATE, so a trial that began just before 36 months but terminated after is coded 0.
- The IDB redacts judge identifiers, so judge-level effects cannot be modeled and district codes are the finest court control.
- The source contains first-listed party names (PLT, DEF); the release must drop them and must not republish names even though the underlying file is public.
- ORIGIN=1 excludes removed, transferred, remanded, reopened, and MDL-transferred cases, so findings may not generalize to those populations.
- JURY coverage is incomplete; many records have missing jury-demand values, which should be retained as an explicit missing category.
- U.S. public-domain status does not resolve non-U.S. copyright or database rights; reusers outside the United States should verify local law, and any CC0 applied to our derived release covers only our contribution.
- No publisher-suggested citation was found; the recommended credit is 'Federal Judicial Center, Integrated Database (IDB)' and no upstream attribution is legally required.
- Current-vintage retrospective snapshot: the IDB is a 2026 file and records may be overwritten across updates, so this is not proof of historical operational availability.
- Trial is rare, roughly 0.6-1% of civil filings; the release is strongly class-imbalanced and should be evaluated with precision/recall or PR-AUC rather than accuracy.
- PROCPROG is missing for some terminated records; the frozen task treats missing PROCPROG as no trial, which may undercount trials.
- Trial start dates are not usable because TRIBEGAN and TRIALEND are no longer maintained; the 36-month horizon is measured from TERMDATE.
- The source contains first-listed party names and judge identifiers; the release drops them and does not republish names.
- ORIGIN equal to 1 excludes removed, transferred, remanded, reopened, and MDL-transferred cases, so findings may not generalize to those populations.
- JURY coverage is incomplete; missing jury-demand values are retained as an explicit missing category.
- DISTRICT is carried as the codebook text code because seven districts use a letter suffix (3A, 3C, 3E, 3G, 3J, 3L, 3N) and Alaska uses the hyphen code 7-; any numeric coercion or hyphen stripping would merge or mislabel distinct courts. The build preserves the hyphen code and reports raw hyphen rows, but that district check only covers rows inside the fixed filing windows.
- U.S. public-domain status does not resolve non-U.S. copyright or database rights; reusers outside the United States should verify local law.
- No publisher-suggested citation was found; the recommended credit is Federal Judicial Center, Integrated Database (IDB).
- No full-row historical availability ledger is available; feature status is assumed and training label timing is unverified.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
