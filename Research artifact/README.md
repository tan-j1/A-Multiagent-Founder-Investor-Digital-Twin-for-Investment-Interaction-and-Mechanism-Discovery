# Shared Text Pitch with Multiple Investor Digital Twins

Code for the first-stage *Organization Science* proposal. A founder and at least two investor-conditioned digital twins share one chronological public text conversation. Each run yields a hash-linked trajectory ending in `deal`, `no_deal`, `censored`, or `unsupported`; a deal identifies the accepted investor and cash-for-equity terms. Methods are specified in the proposal and Technical Appendix.

| Module | Role |
| --- | --- |
| `schema.py` | Public turns, founder claims, investor replies, speaker-owned offers, and policy identity. |
| `annotation.py` | Raw-text registry, character-offset evidence spans, independent coder judgments with adjudication, directed peer links, founder choices, venture outcomes, and venture partitions. |
| `protocol.py` | Fixed-prefix inputs, the five-class historical action labels, the peer-text masking diagnostic, and component scoring. |
| `training.py` | Unique target-turn observations; a shared investor-conditioned low-rank adapter on a frozen Qwen backbone; prompt-masked reply loss with action, terms, stance, and consistency objectives. |
| `knowledge.py` | Read-only historical snapshot with same-investor, cross-venture, three-channel BGE retrieval. |
| `knowledge_lifecycle.py` | Ingest, partition, build, audit, and atomic publication of retrieval snapshots for new sessions. |
| `prompt_builder.py` | Versioned local-model prompts built from the permitted public prefix and retrieved cases. |
| `local_model.py` | Hash-verified local Qwen adapter inference and BGE encoding. |
| `local_backend.py` | Request serialization, retrieval, structured response parsing, and model/index/prompt pinning. |
| `panel.py` | Event-sourced founder and investor state machine with atomic commits, owned offers, and branch snapshots. |
| `rollout.py` | Complete-pitch scheduling under a prespecified investor-turn budget. |
| `audit.py` | Independent hash-chain replay and outcome reconciliation. |
| `evaluation.py` | Component, economic, interaction-trace, outcome, system-validity, matched-baseline, and prospective-correspondence measures. |

Core contracts: every substantive label points to an exact own-speaker source span and keeps its coder provenance; each unique `(venture, target investor, target turn)` reply enters training once, with prompt tokens masked; ventures are partitioned separately for fitting, retrieval, tuning, and final evaluation; a twin reads a peer's words only after that peer's turn is validated and committed; generated text is never edited, and a rejected reply leaves the state unchanged and is retried within a fixed budget before the session is censored; live sessions never write into the historical index; and only an explicit founder acceptance of an active offer produces a deal.
