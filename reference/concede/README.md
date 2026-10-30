# Concede-HRL reference results

This directory is where the **real, externally produced** Concede-HRL results
go before `scripts/import_concede.py` is allowed to import them.  The project
never trains an approximation of Concede-HRL here and never fabricates
reference numbers (specification sections 0.1, 25, 41).

## Files expected

### `manifest.json`

Fill in the actual configuration of the Concede checkpoints that produced the
predictions (see `manifest.example.json` for the shape):

| field | meaning |
|---|---|
| `method` | must be exactly `Concede-HRL` |
| `objective` | must be `makespan` |
| `evaluation_mode` | the concrete deterministic evaluation mode used (non-empty) |
| `assign_mode`, `soft_aggregation`, `wait_aggregation`, `wait_softmin_normalized` | the actual aggregation settings of the trained Concede model |
| `training_episode_budget` | the episode budget the reference was trained with (9600 for the matched-budget protocol) |
| `checkpoint_selection` | e.g. `best_validation` |
| `training_seeds` | the Concede training seeds present in the predictions file |
| `dataset_manifest_hash` | the dataset fingerprint printed by `scripts/generate_dataset.py`; if non-empty it is verified on import |

### `predictions.csv`

One row **per instance per training seed**:

```csv
instance_id,instance_sha256,method,train_seed,checkpoint_id,eval_mode,makespan,inference_time_ms,schedule_file
test_small_0001,<sha256 from data/manifests/test_small.json>,Concede-HRL,1,ckpt_best,deterministic_hard,123.4,85.2,schedules/test_small_0001_seed1.json
```

* `instance_sha256` must equal the frozen dataset hash of that instance
  (import aborts on any mismatch).
* `schedule_file` is optional; when provided, the schedule is re-validated
  with the project's own validator and its makespan is recomputed — the
  reported `makespan` must match or the import aborts.
* Every instance of every covered split must appear for each declared seed —
  partially imported sets are refused (no silent filtering of failed
  instances).

## Import

```bash
python scripts/import_concede.py --input reference/concede/predictions.csv
```

The verified rows are written to `results/raw/concede_predictions_verified.csv`
together with an import-meta sidecar (`concede_import_meta.json`) recording
any warnings (e.g. single training seed).
