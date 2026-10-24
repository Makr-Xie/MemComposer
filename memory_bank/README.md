# Memory bank construction

Each executor model gets its own memory bank, built from its own no-memory trajectories on the
training split and reviewed by the same model.

| benchmark | code | banks |
|---|---|---|
| τ²-bench (airline, retail, telecom) | [`tau2/`](tau2/README.md) | one per agent and domain |
| ALFWorld | [`alfworld/`](alfworld/README.md) | one per executor model |

Both pipelines write under `MEMCOMPOSER_DATA` (default `<repo>/data`), in `tau2/` and
`alfworld/` respectively. Credentials are read from environment variables only.
