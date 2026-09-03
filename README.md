# hvc

## swarmwatch

Blue-team detection and consent control for multi-agent swarms — a monitoring
system built so that agent swarms can be detected for safety, and so that no
agent can replicate itself between chat windows without a human's informed
consent.

See [`swarmwatch/README.md`](swarmwatch/README.md) for the design and its
reasoning.

```sh
cd swarmwatch
cargo test                                   # 203 tests
cargo run --release --bin swarmwatch-demo    # full protocol, six scenarios
cargo run --release --bin swarmwatch-probe   # calibration check; exits non-zero on failure
```

Three layers, ordered by how much of each the adversary can reach:

- **L0 — consent gate.** Replication is *mediated*, not merely observed. State
  cannot reach a new window without a single-use, non-delegable, payload-bound
  grant issued after a person answered questions about that specific transfer.
- **L1 — canaries and lineage census.** Deterministic predicates catching
  transfers that route *around* the gate, with no threshold and no calibration.
- **L2 — population coupling.** Anytime-valid detection of coordination against a
  live control population, as the backstop for paths nobody enumerated.

The load-bearing argument: a steganographic adversary must *convey information*
and may do so arbitrarily slowly, so its divergence can be driven to zero. A
containment adversary must *produce an effect*, and effects have minimum sizes —
divergence is imposed by the goal, not chosen by the adversary. Swarms add a
second floor, because coordination is a relation between agents rather than a
message between them, and to coordinate less detectably they must coordinate
less.

Zero dependencies: monitoring infrastructure should not carry a supply chain.
