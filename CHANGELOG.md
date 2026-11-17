# Changelog

## 0.1.0

- The learning loop for decision jobs: sources (SQLite, JSONL, Langfuse exports) → synthetic top-up with blind relabeling → SFT with lumbridge-rl → frozen time-split eval gate → promote with a log.
- Outputs of models whose terms forbid training are kept eval-only.
- Per-job training day (`schedule.training_day`) and `due` command for timers.
- Trace profile command (counts only).
