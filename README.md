# Lumbridge Train

Train small decision models from your own work, on your own machine. For anyone running agents on their own hardware (a DGX Spark, a workstation GPU) who wants a fast, private model that gets better at their jobs every week. Your data, the training and the weights never leave the box.

Part of [Lumbridge](https://github.com/lumbridge-corp). Early.

## The learning loop

```text
 your agents ──traces──▶ sources ──▶ synth ──▶ train ──▶ eval gate ──▶ promote
 (Langfuse, SQLite,        real rows   a teacher   SFT on     frozen eval    `current` link
  JSONL)                   + labels    tops up     your GPU   set: beat the  + promotions log
                                       rare cases              old version   (rollback = relink)
```

1. **Sources** read examples you already have: Langfuse trace exports, any SQLite table, or JSONL.
2. **Synth** asks a teacher model (any OpenAI-compatible endpoint, usually a model on your own box) to write made-up inputs for answers you have too few of, then labels them blind. Only rows where the two passes agree are kept.
3. **Train** runs supervised fine-tuning with [lumbridge-rl](https://github.com/lumbridge-corp/lumbridge-rl) (our light fork of Prime Intellect's prime-rl) and exports Hugging Face weights.
4. **Eval gate** scores the old and the new model on a frozen set split off by time (the newest rows), so the test is always "tomorrow's data". The new model goes live only if it wins.
5. **Promote** moves `<workdir>/current` to the new weights and appends the decision to `promotions.jsonl`.

Each job picks its training day (`schedule.training_day`, for example `saturday` or `sunday`); `python -m lumbridge_train due <job>` exits 0 on that day, for a timer or cron job.

### Outputs that may not teach

Some providers' terms forbid using their outputs to train another model. Rows whose answers came from those models (by default `claude-*`, `gpt-*`, `o*`, `gemini-*`) are marked eval-only: they can grade your model but never go into its training set. Change the list per source with `eval_only_models`. Your own inputs (the prompts and tasks) are yours to reuse; regenerate answers with a model whose license allows it.

## First result

Our first job triages a founder's own mail on a DGX Spark. The data was 1,167 real threads (labels from rules and a local model) plus 94 synthetic emails, and the test set was the 400 newest threads, held out by date. A Qwen3-0.6B student trained for 80 steps (about 5 minutes) went from 11.8% to 79.3% on category and from 12.8% to 96.0% on "needs a reply". The data stays private; the code that did it is this repository.

## Status

- **Version:** 0.1.0. Interfaces can change until 1.0.
- **Runs on:** Linux with one NVIDIA GPU, tested on a DGX Spark (GB10, aarch64, CUDA 13).
- **Jobs:** decisions with fixed answers: one choice field plus optional true/false fields, answered as one JSON object.
- **Not yet:**
  - Logit-only decisions with calibrated probabilities (one forward pass, no generated tokens).
  - RL from verdicts (the trainer supports it; no job uses it yet).
  - Image and audio inputs.
  - Serving: you load `<workdir>/current` into your own server (vLLM, llama.cpp).

## Install

You need: Python 3.11+, `zstd` (for Langfuse exports), and a working [lumbridge-rl](https://github.com/lumbridge-corp/lumbridge-rl) checkout for training and evals. Lumbridge Train itself uses only the Python standard library.

```sh
git clone https://github.com/lumbridge-corp/lumbridge-rl.git ~/src/lumbridge-rl   # then follow its README
git clone https://github.com/lumbridge-corp/lumbridge-train.git ~/src/lumbridge-train
cd ~/src/lumbridge-train
python3 -m unittest discover -s tests
```

Runs write to `~/lumbridge-train-runs/<job>/` (set `workdir` in the job). Remove that folder to remove everything a job made.

## Quick start

Point the example job at any OpenAI-compatible chat endpoint (a local vLLM, llama.cpp server or hosted API), then:

```sh
python3 -m lumbridge_train build jobs/example-ticket-routing.toml   # data only: look before you train
python3 -m lumbridge_train run   jobs/example-ticket-routing.toml   # build, train, gate, promote
```

The example has no real data, so it trains on synthetic tickets alone. Add a `[[source]]` with your own rows to make it yours.

## Configuration

One TOML file per job. The parts:

| Table | What |
|---|---|
| `[job]` + `[[job.fields]]` | Instructions and the answer fields: `choices` (with optional `descriptions`) or `type = "bool"` |
| `[[source]]` | `sqlite` (a query returning `input`, `ts`, one column per field), `jsonl`, or `langfuse_archive` |
| `[teachers.<name>]` | `base_url`, `model`, optional `api_key_env`, `headers`, `concurrency`, `extra_body` |
| `[synth]` | `writer`, `labeler`, `per_choice_min`, `skip_choices` |
| `[split]` | `eval_fraction` (newest rows), `max_eval` |
| `[train]` | `student` (HF id), `rl_dir`, `max_steps`, `batch_size`, `seq_len`, `lr`, `max_per_choice`, `renderer` |
| `[gate]` | `metric` (default `<first field>_acc`), `min_gain`, `min_valid_json` |
| `[schedule]` | `training_day` |

See `jobs/example-ticket-routing.toml` for a complete file.

## Privacy

Lumbridge Train never sends data anywhere you did not configure. Teachers are the only network calls, and they go to the endpoints in your job file. Profiles (`python -m lumbridge_train profile <archive>`) print counts only, never message text. Keep job files that point at private data in a private repository.

## License

Apache-2.0. See [LICENSE](LICENSE).
