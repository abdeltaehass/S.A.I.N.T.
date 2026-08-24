# S.A.I.N.T. — Network Intrusion Detection

Network intrusion detection system that classifies traffic flows across five
attack categories — and uses an LLM agent to explain *why* each alert fired,
in plain English.

**Live project page:** https://abdeltaehass.github.io/S.A.I.N.T./

**Stack:** Python · PyTorch · Flask · Redis · Plotly · Docker

> Research and educational project, built on the NSL-KDD benchmark dataset.
> Not hardened for production network defense.

---

## What it does

Most intrusion detectors emit a score and leave an analyst to work out what it
meant. S.A.I.N.T. pairs detection with an explanation layer: every alert is
attributed to the features that drove it, then written up as a readable incident
narrative.

- **41-feature traffic classification** over the NSL-KDD flow schema
- **Five attack categories** — `normal`, `dos`, `probe`, `r2l`, `u2r`
- **Feature attribution** for each alert, so a decision can be audited
- **LLM incident reports** turning structured detections into analyst-readable text
- **Live dashboard** of traffic, alerts, and category breakdowns
- **Replay harness** that streams the dataset back through the stack as live traffic

## Feature schema

| Group | Features |
|---|---|
| Basic | `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, … |
| Content | `hot`, `num_failed_logins`, `logged_in`, `root_shell`, `su_attempted`, … |
| Time-based | `count`, `srv_count`, `serror_rate`, `same_srv_rate`, … *(last 2 s, same host)* |
| Host-based | `dst_host_count`, `dst_host_srv_count`, `dst_host_serror_rate`, … *(last 100 connections)* |

## Attack categories

| Class | Meaning |
|---|---|
| `normal` | Benign traffic |
| `dos` | Denial of service |
| `probe` | Surveillance and scanning |
| `r2l` | Remote-to-local intrusion |
| `u2r` | User-to-root privilege escalation |

## Layout

```
model/classifier.py     traffic-flow classifier over the 41-feature schema
agent/reasoning.py      correlates signals across connections
agent/explainer.py      attributes alerts to the features that drove them
agent/llm_reporter.py   structured detection -> incident narrative
api/routes.py           Flask service, Redis-backed
dashboard/app.py        live Plotly dashboard
data/loader.py          NSL-KDD ingestion
scripts/train.py        train the classifier
scripts/replay.py       replay traffic through the live pipeline
```

## Run it

```bash
git clone https://github.com/abdeltaehass/S.A.I.N.T.
cd S.A.I.N.T.
cp .env.example .env

docker compose up          # API + Redis + dashboard

python scripts/train.py    # train the classifier
python scripts/replay.py   # replay traffic through the live pipeline
```

## License

MIT
