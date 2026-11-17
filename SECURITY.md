# Security policy

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on this repository, not in a public issue.

Lumbridge Train reads private data (traces, mail, logs) by design. Treat job files as trusted local configuration: a `sqlite` source runs the query you write, and teachers receive the text of your examples. Point teachers only at endpoints you trust with that data, keep API keys in the environment (`api_key_env`), and keep job files that name private data out of public repositories. Run directories hold your data in plain JSONL; protect them like the source data.
