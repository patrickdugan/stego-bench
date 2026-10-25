# Math: can an AI output be certified free of steganography?

**Research note, 25 October 2026.** This directory develops the formal, testable security objective for StegoBench. The finite examples have exhaustive checks. No end-to-end zero-leakage proof for an unrestricted LLM, or for StegoBench's existing model-output paths, is claimed.

The natural object of proof is the *release mechanism*, including observable side effects. A finished piece of text cannot be declared free of every possible covert codebook: a colluding recipient could assign meaning to wording, timing, or choice among valid answers. A meaningful theorem specifies the secret, legitimate information, receiver knowledge, and process that produces the output.

## 1. Separate cover matching from zero leakage

Let:

- \(P\) be the public task and previously authorized context.
- \(S\) be a secret known to the potential sender.
- \(A\) be the disclosure authorized **before** observing the candidate output.
- \(K\) be a receiver's shared key, decoding rule, or codebook.
- \(Z\) be additional receiver knowledge, including earlier observations.
- \(O\) be the *complete observation*: text bytes, length, metadata, timing, errors/refusals, tool effects, and message scheduling.

A passive steganographic observer may see *cover-distribution matching*:

\[
\mathcal L(O\mid P=p,S=s)=\mathcal L(O\mid P=p,S=s')
\]

even when a recipient with key \(K\) decodes \(S\). Cover matching is an attacker's concealment guarantee.

The defensive objective is **receiver-relative zero-extra-leakage**:

\[
\boxed{I(S;O\mid P,A,K,Z)=0.}
\]

For finite variables a pointwise, distribution-independent sufficient condition is:

\[
\boxed{\Pr(O=o\mid p,a,k,z,S=s)
=\Pr(O=o\mid p,a,k,z,S=s')}
\]

for each admissible \((p,a,k,z,s,s',o)\). Zero conditional mutual information is equivalent to conditional independence almost surely under a given joint distribution. The pointwise condition covers all admissible secrets, including rarely sampled ones.

**The declassification caveat matters.** The displayed condition promises that output adds zero leakage *beyond A*. If \(A=S\), a canonical renderer can reveal the full secret and still satisfy the conditional condition. Specify the origin and authorization policy for \(A\) in advance. End-to-end secrecy also requires the allowed disclosure to preserve the secrets it should preserve; choosing \(A=O\) after release is vacuous.

## 2. Keyed one-bit counterexample

Suppose a task permits the variants "Yes." and "Certainly." with equal cover probability. Let secret \(S\) and shared key \(K\) be independent uniform bits. The sender emits \(Y=S\oplus K\), using the two phrases for values 0 and 1. Then

\[
\Pr(Y=0\mid S=0)=\Pr(Y=0\mid S=1)=\tfrac12,
\quad I(S;Y)=0.
\]

The keyed receiver recovers \(S=Y\oplus K\):

\[
\boxed{I(S;Y\mid K)=1\text{ bit}.}
\]

The cover distributions are *identical*. A blind monitor has no signal under this toy model. Low AUROC, answer-preserving paraphrase, and natural-looking output cannot certify zero covert communication against a shared-key receiver. More general distribution-preserving constructions are known [1].

The standard-library example [toy_channels.py](toy_channels.py) checks exact conditional output laws using rational counts, then calculates mutual information. Run from the repository root:

~~~bash
python Math/toy_channels.py
python -m unittest discover -s Math -p 'test_*.py' -v
~~~

Expected results for fair independent bits:

| Channel | \(I(S;Y)\) | \(I(S;Y\mid K)\) | Zero leakage to keyed receiver? |
|---|---:|---:|---|
| Keyed cover variants | 0 | 1 | No |
| Secret-blind fixed renderer | 0 | 0 | Yes, in finite model |
| Sender-selected canonical JSON field | 1 | 1 | No |
| Secret-dependent trailing space | 1 | 1 | No |

A passing finite check applies to exactly this four-state experiment; it cannot certify a deployed model or network.

## 3. Secret-blind renderer theorem

Suppose a trusted mechanism fixes an authorized disclosure \(A\) under a policy written in advance. It releases

\[
O=R(P,A,U)
\]

with specified deterministic renderer \(R\) and trusted randomness \(U\) such that

\[
U\;\perp\!\!\!\perp\;(S,K,Z)\mid(P,A).
\]

Assume the full observable release, including the decision whether and when to send, follows this definition; no sender-controlled choice can bypass the renderer.

For any fixed admissible \(p,a,k,z,s\),

\[
\begin{aligned}
\Pr(O=o\mid p,a,k,z,s)
 &= \sum_{u:R(p,a,u)=o}\Pr(U=u\mid p,a,k,z,s)\\
 &= \sum_{u:R(p,a,u)=o}\Pr(U=u\mid p,a).
\end{aligned}
\]

This is independent of \(s\), proving \(I(S;O\mid P,A,K,Z)=0\). For continuous randomness replace the sum with an integral. A unique canonical serialization with no randomness is a special case.

The hard part is proving **input provenance and observational closure**. Canonicalization alone fails if the sender chooses which valid facts to disclose, orders the facts, selects an optional field, triggers a refusal, or influences timing. An independent unique-answer computation \(A=g(P)\), or another separately verified information-flow policy, is needed. If \(g(P)\) cannot depend on \(S\) after fixing the public task, the output is also independent of \(S\) given \(P,K,Z\), under the analogous assumptions.

For multiple turns, apply an information-flow proof to the joint transcript. Induction works when each release step, its scheduling, and its error path are secret-independent given the public task and preceding observable history. Adaptively chosen public inputs require special care.

## 4. Capacity and approximate guarantees

If a fixed public task and receiver context allow at most \(N\) *complete observable outputs*, then

\[
I(S;O\mid P,K,Z)\leq H(O\mid P,K,Z)\leq\log_2 N
\]

bits per release. With one uniquely specified output for fixed independently authorized content, the representation-choice channel has zero capacity. Secret-bearing authorized fields can still leak through their values. Length, absent messages, retries, headers, and time are part of the real output alphabet.

For a uniform binary secret, fix the receiver's side information and write \(Q_s\) for the observed-output law under \(S=s\). If \(\mathrm{TV}(Q_0,Q_1)\leq\epsilon\), the best receiver's probability of identifying the bit from that observation is bounded by \((1+\epsilon)/2\). This bound depends on the specified secret, side information, and observation boundary.

## 5. How this relates to StegoBench

Existing StegoBench suites measure message recovery, detection AUROC/FPR, paraphrase survival, trained channel frontiers, and visual rule discovery. Those quantities are empirical properties of particular attacks and monitors. This Math track addresses a different question: whether a **restricted release protocol excludes unauthorized channels for every sender strategy** admitted by its threat model.

A future experiment should compare a baseline free-form model with a uniquely checked public-answer renderer, holding task data and attacker opportunities fixed. Measure legitimate task accuracy and refusals beside covert recovery and monitor scores. Record machine-checked proof status separately from empirical results. The [proof-obligations document](PROOF_OBLIGATIONS.md) gives a concrete acceptance plan.

## 6. Open questions

1. Which useful tasks have independently computable, uniquely authorized answers?
2. Can a verified renderer and its actual transport be shown relationally noninterfering, including error, timing, and refusal effects?
3. Which secret-independent per-turn conditions compose into a transcript-level theorem against adaptive keyed receivers?
4. For open-ended language, can one establish a nontrivial *worst-case* output-capacity bound while retaining useful task performance?

## References

1. C. Schroeder de Witt et al., [*Perfectly Secure Steganography Using Minimum Entropy Coupling*](https://arxiv.org/abs/2210.14889), ICLR 2023. Information-theoretically distribution-matched covert channels under the paper's assumptions.
2. U. Anwar et al., [*A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring*](https://arxiv.org/abs/2602.23163), 2026. Usable-information asymmetry when an innocent text distribution is unavailable.
3. F. Mächtle et al., [*Steganography Without Modification: Hidden Communication via LLM Seeds*](https://arxiv.org/abs/2606.09135), 2026. Seed-controlled sampling as a covert information channel.

The theorem above is a conditional-independence argument under explicit trusted-boundary assumptions. It is a mathematical specification, not a proof about StegoBench's present implementation.
