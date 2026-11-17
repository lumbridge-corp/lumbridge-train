# Changelog

## 0.2.0

- Labeling page (`python -m lumbridge_train label`): a teacher model pre-labels, a person confirms or fixes with one key press; disagreements first. Answers override the source in every build, and `split.eval_from = "human"` makes them the eval set.
- Train and evaluate on a rented GPU (`[train.cloud]`, Prime Intellect): picks an offer in the job's GPU order (spot first), trains and evaluates there, copies weights and scores back, and always deletes the machine. Refuses unless the job sets `data_leaves_fleet = true`. `cloud-reap` deletes leftover machines.
- Sources can keep unlabeled rows, for labeling.

## 0.1.0

- The learning loop for decision jobs: sources (SQLite, JSONL, Langfuse exports) → synthetic top-up with blind relabeling → SFT with lumbridge-rl → frozen time-split eval gate → promote with a log.
- Outputs of models whose terms forbid training are kept eval-only.
- Per-job training day (`schedule.training_day`) and `due` command for timers.
- Trace profile command (counts only).
