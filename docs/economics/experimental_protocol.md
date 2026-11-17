# Protocol

## Simulation

One run is a stream of `steps` queries against a `World`, answered by one
policy using one memory. Each step:

1. the world produces a key
2. the policy answers using memory, recomputation and verification as it chooses
3. the answer is checked against the current truth; a wrong answer is an error
4. counts of computes, reads, writes, verifies and errors are recorded

Costs are applied afterwards by a `CostModel`, so one run can be scored under
many weightings.

## World

Keys are integers. With probability `recurrence` a query repeats a key drawn
uniformly from the last `window` new keys, otherwise it is a new key. Every key
has its own random schedule of changes: gaps between changes are geometric with
per-step rate `drift`, and a change moves the value to a different one. The
schedule depends only on (seed, key), so every policy sees the identical world
for a given seed no matter how often or when it looks.

## Memory

Memories store and return values and know nothing about costs. They are
read, written and forgotten through the simulator, which does the counting.
`ExactCache` has an optional LRU capacity and an optional read-noise rate.
`OracleMemory` returns the current truth for any key it has been given, so it
is the cost floor for retrieval.

## Policies

- recompute: always compute, never touch memory
- retrieve: read; on a miss compute and write; on a hit trust the value
- verify: as retrieve, but pay the verification cost on a hit; if the value is not
  current, compute and write
- refresh (TTL): as retrieve, but a hit older than the TTL is recomputed and rewritten

Forgetting exists as a memory operation. A forgetting policy is not part of v0.1.

## Reporting

- 3 seeds, each run `steps` long, mean cost per query reported
- regime maps show the cheapest policy per (recurrence, drift) cell
- every map is shown under at least two cost models
- a cell where the best policy is within 2 percent of the runner-up is marked as a tie
