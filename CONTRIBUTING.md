# Contributing to Lumbridge Train

Small, reviewable pull requests are welcome. Development happens in our private repository; accepted changes reach this repository with the next release.

```sh
python3 -m unittest discover -s tests
```

Rules: standard library only in `lumbridge_train/` (torch and transformers are used only inside the lumbridge-rl environment by `evaluate.py`). Never add real data, traces, keys or private hostnames to examples, tests or docs; use made-up rows.
