# E2E Pipeline

Unified pipeline combining:
- Rule-Based (lexicon matching)
- Hybrid Lexicon (Rules + SetFit)

## Structure
- `src/common` — shared utilities (text normalization, IO, metrics, current job selection)
- `src/algorithms` — algorithm implementations
- `config` — parameters and paths
- `pipelines` — entry points for running inference/validation
- `artifacts` — produced prediction CSVs
- `models` — saved checkpoints for SetFit

## Interactive single-title run
- Rule-Based: `python pipelines/interactive.py "<job title>" --algo rule_based`
- Hybrid (Lexicon + SetFit): `python pipelines/interactive.py "<job title>" --algo hybrid_lexicon`
- Both: `python pipelines/interactive.py "<job title>" --algo all`
- Loop mode: add `--loop` to keep entering titles until blank input.
