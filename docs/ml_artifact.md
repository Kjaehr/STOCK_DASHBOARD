## ML model artifact (for Vercel inference)

This describes the JSON schema used by the serverless inference in `web/src/lib/ml/model.ts` and `web/src/lib/ml/features.ts`.

- Location in Supabase Storage:
  - `ml/models/model_v*.json` — concrete models
  - `ml/models/latest.json` — pointer file: `{ "path": "ml/models/model_vX.json", "version": "vX" }`

### Schema
```
{
  "version": "v1_YYYYMMDDHHMM",
  "features": [string, ...],              // must match keys from toBaseFeatures()
  "intercept": number,
  "coef": [number, ...],                  // same length as features
  "norm": {
    "mean": { [feature: string]: number },
    "std":  { [feature: string]: number }
  }
}
```

- Features must correspond to `toBaseFeatures(x)` in `web/src/lib/ml/features.ts`.
- Inference transforms each input feature `v` to `(v - mean[k]) / std[k]` when `std[k]>0`.
- Final logit: `z = intercept + sum_i coef[i] * normalized_feature[i]`, and probability `p = sigmoid(z)`.

### Baseline training (v1)
- The initial `train_model.py` uses existing screener outputs in `data/`:
  - Label (temporary): `y = 1` if `score >= 60`, else `0`.
  - Features: a subset mirroring `toBaseFeatures()` (ratios to SMAs, RSI normalization, ATR buckets, selected fundamentals and sentiment).
- Trained model and normalization stats are exported to `ml_out/`:
  - `model_v1_YYYYMMDDHHMM.json`
  - `latest.json` → `{ "path": "ml/models/model_v1_YYYYMMDDHHMM.json", "version": "..." }`

### GitHub Action
- Workflow: `.github/workflows/train-model.yml`
- Trigger: `workflow_dispatch` (manual). It trains, writes `ml_out/*`, and uploads to Supabase Storage using repo secrets:
  - `SUPABASE_URL`, `SUPABASE_BUCKET`, `SUPABASE_SERVICE_ROLE` (Service role token)

### Notes
- Once you have a real supervised label (e.g., N‑day forward return > threshold), replace the baseline label logic in `scripts/ml/train_model.py` and re‑train.
- Keep `features.ts` and Python `to_base_features()` in parity whenever new features are added.

