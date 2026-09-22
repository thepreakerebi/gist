# OmniPack replicability audit

**Paper:** Su et al., "OmniPack: Unified Token Compression for Efficient Omni-modal
Large Language Models," arXiv:2608.03812.
**Audited:** 2026-09-22, from the arXiv HTML (v1), before writing any code.
**Purpose:** decide whether the supervisor's instruction to replicate OmniPack and
benchmark against the replication is feasible, and on what terms.

**Verdict: feasible.** The method is more completely specified than expected —
every hyperparameter has a stated value and the clustering step is given in full in
Appendix B.1. Four details are missing, all with a defensible default. Budget about
two days.

**But the audit found two facts that matter more than the head to head, and both
should reach the paper whether or not the replication is built.** They are in
§1 and §2 below. Read those first.

---

## 1. OmniPack is post-encoder, in its own words

This is the load-bearing fact for Gist's entire contribution, and it is now settled
from the paper's own prose rather than inference:

> "Given a video 𝒳ᵥ and its aligned audio 𝒳ₐ, the modality-specific
> encoder–projector pipelines map them into the LLM embedding space, producing Nᵥ
> visual tokens 𝐗ᵥ and Nₐ audio tokens"

> "For the encoder tokens 𝐗ᵐ of modality m, we retain Kₘ tokens before the LLM."

The encoders run first and produce tokens; compression consumes those tokens. The
importance signal makes this unavoidable rather than incidental — it is read out of
the encoder itself:

> "Attention statistics from the final modality encoder provide a base estimate of
> token centrality: 𝐀ᵐ = Softmax(𝐐ᵐ(𝐊ᵐ)ᵀ/√d)"

You cannot take the final encoder layer's attention without running the encoder.
OmniPack's Stage 1 is therefore structurally incapable of saving encoder compute, no
matter how aggressive its ratio.

**The terminology trap, now concrete.** OmniPack's "pre-LLM" means *after the
encoders, before the language model*. Gist's "pre-encoder" means *before either*.
The two phrases look alike and mean different positions in the pipeline. One
sentence in the paper closes this, and it should quote the encoder-projector line
above rather than paraphrase it.

## 2. OmniPack's FLOPs accounting excludes the encoders

The headline numbers — 98.0% of performance at 16.7% of FLOPs, 92.9% at 6.8% —
are computed over the language model only:

> "Vision and audio encoders, modality projectors, textual tokens, token-selection
> operations, and the language-model head are excluded."

This is the single most useful finding in the audit. Three consequences:

1. **Their metric cannot represent Gist's contribution.** Encoder FLOPs are exactly
   the quantity Gist reduces, and OmniPack's accounting zeroes it out by definition.
   A head to head reported on their basis would show Gist's advantage as nothing.
2. **Any comparison must be recomputed on a common basis that includes encoders**,
   and the basis must be stated explicitly in the caption. This is a methodological
   requirement, not a presentational preference.
3. **It is not a criticism of their paper.** Excluding a fixed cost is reasonable
   when every method you compare pays it identically. It stops being reasonable the
   moment a method that *doesn't* pay it enters the table — which is the argument
   Gist is making, and this quote is the evidence that the argument is live.

This finding stands on its own. It needs no replication to support it, and it does
not depend on any accuracy number coming out in Gist's favour.

## 3. What is fully specified

| Component | Specification | Value |
| :-------- | :------------ | :---- |
| Attention centrality | `𝐀ᵐ = Softmax(𝐐ᵐ(𝐊ᵐ)ᵀ/√d)`, final encoder layer | — |
| Video temporal variation | `1 − cos(𝐱ₜ,ₚ, 𝐱ₜ₊₁,ₚ)` | — |
| Video spatial variation | `1 − cos(𝐱ₜ,ₚ, mean_y 𝐱ₜ,ᵧ)` | — |
| Audio variation | `1 − cos(𝐱ₜ, 𝐱ₜ₊₁)` | — |
| Importance/coverage split | `Kₘⁱᵐᵖ = round(ηₘKₘ)` | η_v = 0.25, η_a = 0.35 |
| Clustering distance | `1 − cos(𝐱ᵢ,𝐱ⱼ) + λ·dᵢⱼᵖᵒˢ` | λ = 0.20 |
| DPC-KNN density | `ρᵢ = exp(−(1/k)Σ_{j∈KNN} dᵢⱼ²)` | k = 7 |
| DPC-KNN separation | `δᵢ = min_{j:ρⱼ>ρᵢ} dᵢⱼ`, else `max_{j≠i} dᵢⱼ` | — |
| Coverage ranking | `γᵢ = ρᵢ·δᵢ`, top (Kₘ − Kₘⁱᵐᵖ) | — |
| Merge affinity | `Φᵢⱼ = cos(𝐱ᵢ,𝐱ⱼ) − τₘdᵢⱼᵖᵒˢ + ζs̄ⱼ` | τ_v = 0.10, τ_a = 0.05, ζ = 0.10 |
| Merge update | `(𝐱ⱼ + Σwᵢ𝐱ᵢ)/(1 + Σwᵢ)`, `wᵢ = (1+s̄ᵢ)/2` | — |
| Stage 2 insertion layer | Qwen2.5-Omni-7B | layer 18 |
| Stage 2 retention | `rₘ' `, fixed | 50% |
| Stage 2 relevance | `Rᵢ = 𝒩(max{cos(𝐡ᵢ,𝐪), max_j cos(𝐡ᵢ,𝐡ⱼᵍ)}) + 𝒩(cos(𝐡ᵢ,𝐩ᵐ̄)) + 𝒩(uᵢ)` | — |
| Stage 2 representativeness | `uᵢ = exp(−mean_j(1−cos(𝐡ᵢ,𝐡ⱼ)))` | — |
| Stage 2 greedy rule | `i* = argmax (1+𝒩(Rᵢ))·Dᵢ`, `Dᵢ = min_{j∈S}(1−cos(𝐡ᵢ,𝐡ⱼ))` | — |

This is a good specification by the standards of the field. Most of the method can
be written directly from the paper.

## 4. What is not specified — four gaps, four defaults

Each gets one documented choice in the reimplementation. None is fatal; all should
be listed in the paper as assumptions rather than buried in code.

| # | Gap | What the paper says | Default to adopt |
| - | :-- | :------------------ | :--------------- |
| 1 | `𝒩(·)` has no formula | described in prose as "modality-wise min–max normalization" | min–max to [0,1] per modality |
| 2 | How attention centrality and the variation signals combine | "the modality-specific structural cues are **added** to the attention-centrality vector" — no coefficients; video has two variation terms whose relative weight is unstated | unweighted sum after per-signal normalization; video's two terms each at weight 1 |
| 3 | `dᵖᵒˢ` has no formula | "normalized temporal distance for audio or spatiotemporal distance for video", each "independently normalized to [0,1]" | audio: \|Δt\| normalized; video: Euclidean over (normalized frame index, normalized patch grid position) |
| 4 | Pooling for `𝐪` and `𝐩ᵐ̄` | "mean-pool"; simple vs attention-weighted unstated | simple mean |

**A fifth item is not a gap but affects comparison design.** The paper reports four
pre-LLM retention ratios — 25%, 20%, 15%, 10% — and no single headline value. Any
head to head must therefore either sweep all four or state which was chosen and why.
Sweeping is cheap here and avoids the accusation of picking their weakest setting.

## 5. Terms for the head to head

If the replication is built, these conditions are not optional:

- **Call it a reimplementation everywhere** — code, tables, captions, slides. Never
  "OmniPack". The claim is about our implementation of their described method.
- **Never write "Gist outperforms OmniPack."** Write what is true: our
  reimplementation, under the four documented assumptions above, scored X.
- **Keep OmniZip as the primary head to head.** It is their own released code, so
  that comparison carries no reimplementation risk. OmniPack's is a third,
  clearly-labelled result.
- **Release the reimplementation**, with §4's assumptions in its README. No public
  implementation of OmniPack exists; a documented one is a contribution in itself.
- **Report FLOPs on a basis that includes the encoders**, and say so in the caption,
  per §2.
- **Sweep all four retention ratios**, per §5 above.

## 6. The fact about code availability still belongs in the paper

Verified against the GitHub API on 2026-09-22: `github.com/RowanSu/OmniPack` is a
single branch containing a two-line README — 0 KB of code, last pushed 2026-07-31.
No fork or mirror exists. The paper announces code availability and there is none.

The replication does not retire this sentence. It is the reason the replication had
to exist, and the two belong together:

> The state of the art has not released its implementation, so no comparison against
> it is possible for anyone outside the authors. We therefore reimplemented the
> published method from its description, documenting each underspecified detail, and
> release that reimplementation.

Do not hedge it with a compute-budget explanation. The repository is checkable by a
reviewer in ten seconds.

## 7. Recommendation

Build it, on the terms in §5, budgeting two days.

But if time runs short before the defense, **§1 and §2 are the findings that matter,
and neither requires a single line of code.** The placement claim is now quotable
from OmniPack's own method section, and their FLOPs exclusion shows that the field's
efficiency metric structurally cannot see the cost Gist removes. An accuracy row
from a reimplementation is worth less than either.
