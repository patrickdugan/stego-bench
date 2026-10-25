# Math: proof obligations and acceptance plan

**Status:** prospective formal research track. No end-to-end verified AI release or preregistered empirical thresholds are claimed.

## Security game

An adversarial sender knows a secret \(S\) and may share a key \(K\), decoder, and earlier conversation \(Z\) with a receiver. A defender must answer a public task \(P\), disclosing only data \(A\) authorized before the attempted communication. The receiver's full observation \(O\) includes output bytes, metadata, timing, control flow, and observable tools.

The security goal is the pointwise law equality

\[
\mathcal L(O\mid p,a,k,z,S=s)
=\mathcal L(O\mid p,a,k,z,S=s')
\]

for each admissible pair of secrets. This is *zero leakage beyond A*; the declassification policy must independently restrict A. The sufficient architecture is \(O=R(P,A,U)\) with verified, secret-independent renderer inputs, side effects, and randomness. A merely valid model-selected subset of facts can encode a secret.

## Obligations before a zero-leakage claim

| ID | Obligation | Required evidence | Example failure |
|---|---|---|---|
| M1 | Define secrets and preauthorized fields | Fixed, audited disclosure policy | Declaring all model output to be authorized |
| M2 | Close source-to-output flows | Relational noninterference proof of actual validator/renderer | Choice among equivalent valid outputs |
| M3 | Bind content provenance | Independently derived unique answer or independently proved selection rule | Model selects one of several true facts |
| M4 | Prove randomness independence | Trusted seed and sampling process | Seed encodes the secret |
| M5 | Cover every observable | Released bytes, omission, error, refusal, timing, metadata, tools | Trailing byte or secret-dependent delay |
| M6 | Cover multi-turn behavior | Inductive transcript proof including adaptive observations | Secret selects when to send a message |
| M7 | Bind proof to code and deployment | Frozen executable, source, policy, environment, transport | Proof applies to a different implementation |
| M8 | Maintain legitimate task utility | Independent accuracy, coverage, and refusal measurements | Always outputting a constant |

A finite set of monitor evaluations cannot by itself establish M2, M5, or M7.

## Minimal acceptance experiment

1. Predefine public tasks whose authorized answers are uniquely computed by a separately checked function \(A=g(P)\).
2. Run the candidate model as an **untrusted proposer**. Record its draft and agreement with \(g(P)\) inside the evaluation environment.
3. Release only the independently computed answer through a constrained serializer. Freeze or include all outward-facing scheduling, error, and metadata behavior in the security boundary.
4. Prove a two-run property on the actual renderer: with equal authorized public inputs and arbitrary admissible secrets/keys, the complete observable release traces are identical, or their distributions are equal under trusted randomness.
5. Run adversarial StegoBench trials on the deployed boundary, including shared keys and multi-turn behavior. Treat these as falsification/regression tests, alongside the proof.
6. Record utility loss against an unrestricted-output baseline, holding legitimate tasks, attacker tuning opportunities, model families, and inference budgets fixed.

For free-form writing, the absence of a unique allowed semantic answer is a substantial unresolved prerequisite.

## Test matrix

| Test | Released surface | Intended result | Evidence status |
|---|---|---|---|
| T0 | Perfectly matched two-phrase cover | Passive monitor sees no difference; keyed decoder gets bit | Exhaustively enumerated toy |
| T1 | Secret-blind constant renderer | Zero leakage in finite state space | Exhaustively enumerated toy |
| T2 | Canonical JSON with model-selected data | Secret leaks through field choice | Exhaustively enumerated toy |
| T3 | Byte-level variable-length output | Secret leaks through trailing space | Exhaustively enumerated toy |
| T4 | Verified answer, variable send/skip schedule | Attempt to falsify transcript noninterference | Proposed |
| T5 | Verified answer versus free-form baseline | Compare utility, receiver recovery, AUROC/FPR | Proposed |

Execute the finite examples:

~~~bash
python Math/toy_channels.py
python -m unittest discover -s Math -p 'test_*.py' -v
~~~

A sample leakage estimate or an AUROC near 0.5 is a *statistic*, not a universal upper bound. For T4/T5 preregister the target error bars, sample-size calculation, scope of receiver side information, adversary budgets, and decision criteria before GPU work.

## Reporting grades

- **TOY_EXHAUSTIVE:** Every state in an explicitly finite toy model checked.
- **SOURCE_PROOF:** Machine-checked information-flow theorem for fixed source, with axioms and deployment gap listed.
- **DEPLOYMENT_BOUND:** Proof tied to executable identity and included observable transport, under declared environmental assumptions.
- **EMPIRICAL_ONLY:** Measured receiver success/monitor scores with uncertainty.
- **UNVERIFIED:** Design proposal without supporting implementation evidence.

The next smallest implementation job is a trusted one-answer JSON renderer and a two-run relational checker, with explicit secret-selected fields, refusal, and send/skip adversarial tests. Extend it to the complete transcript before claiming an end-to-end property.
