"""A reimplementation of OmniPack, from its paper. **This is not OmniPack.**

OmniPack (Su et al., arXiv:2608.03812) is the current state of the art in
training-free omni-modal token compression. Its authors announce code
availability; the repository `github.com/RowanSu/OmniPack` contains a two-line
README and no implementation, verified against the GitHub API on 2026-09-22.
Nobody outside the authors can run their method, so this file exists to make a
comparison possible at all.

Everything here is written from the paper's description. Where the paper
specifies a value, that value is used and cited. Where it does not, a default is
chosen, marked **ASSUMPTION** in the code, and listed in RUN.md. Results from
this file must be labelled "our reimplementation", never "OmniPack", because the
four assumptions below are ours and not theirs.

Assumptions, also tabulated in RUN.md:

  A1  `N(.)` is described in prose as "modality-wise min-max normalization" with
      no formula. Implemented as min-max to [0,1] per modality.
  A2  The paper says structural cues "are added to the attention-centrality
      vector" with no coefficients, and video has two variation terms whose
      relative weight is unstated. Implemented as an unweighted sum of the
      normalized signals.
  A3  `d_pos` is "normalized temporal distance for audio or spatiotemporal
      distance for video", with no formula, "independently normalized to [0,1]".
      Implemented as absolute temporal distance for audio, and Euclidean
      distance over (normalized frame index, normalized patch row, normalized
      patch column) for video.
  A4  Query and prototype pooling is "mean-pool"; simple versus attention
      weighted is unstated. Implemented as a simple mean.

The two stages are pure tensor functions on encoder outputs and hidden states,
so they are testable on CPU without any model. Wiring stage 2 into a live
Qwen2.5-Omni forward pass is the runner's job, not this module's.

Note on placement, since it is the whole point of the comparison: stage 1 runs
**after** the encoders. The paper is explicit -- "the modality-specific
encoder-projector pipelines map them into the LLM embedding space, producing Nv
visual tokens" -- and its importance signal is the final encoder layer's own
attention, which cannot be obtained without running the encoder. Their "pre-LLM"
is post-encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Paper-specified hyperparameters. Section and table references in RUN.md.
ETA_VISUAL = 0.25
ETA_AUDIO = 0.35
LAMBDA_POS = 0.20
TAU_VISUAL = 0.10
TAU_AUDIO = 0.05
ZETA = 0.10
DPC_KNN_K = 7
INNER_RETENTION = 0.50
INNER_LAYER_QWEN_7B = 18

EPS = 1e-8


def minmax(x: torch.Tensor) -> torch.Tensor:
    """ASSUMPTION A1: N(.) as min-max to [0,1], per modality."""
    lo = x.min()
    hi = x.max()
    if (hi - lo).abs() < EPS:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    normed = x / (x.norm(dim=-1, keepdim=True) + EPS)
    return normed @ normed.T


def attention_centrality(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention each token receives, from the final encoder layer.

    `attn` is (heads, N, N) or (N, N) as produced by Softmax(QK^T/sqrt(d)).
    Centrality is the mean over queries of the attention paid *to* each token,
    averaged over heads.
    """
    if attn.dim() == 3:
        attn = attn.mean(dim=0)
    return attn.mean(dim=0)


def visual_variation(embeds: torch.Tensor, frames: int, patches: int) -> torch.Tensor:
    """Eq. 4: adjacent-frame change and spatial distinctiveness.

    `embeds` is (frames * patches, D), ordered frame-major.
    """
    grid = embeds.view(frames, patches, -1)
    normed = grid / (grid.norm(dim=-1, keepdim=True) + EPS)

    temporal = torch.zeros(frames, patches, device=embeds.device, dtype=embeds.dtype)
    if frames > 1:
        cos_next = (normed[:-1] * normed[1:]).sum(dim=-1)
        temporal[:-1] = 1.0 - cos_next
        temporal[-1] = temporal[-2] if frames > 1 else 0.0

    frame_mean = grid.mean(dim=1, keepdim=True)
    frame_mean = frame_mean / (frame_mean.norm(dim=-1, keepdim=True) + EPS)
    spatial = 1.0 - (normed * frame_mean).sum(dim=-1)

    return temporal.reshape(-1), spatial.reshape(-1)


def audio_variation(embeds: torch.Tensor) -> torch.Tensor:
    """Eq. 5: adjacent-token change along the audio track."""
    normed = embeds / (embeds.norm(dim=-1, keepdim=True) + EPS)
    n = normed.shape[0]
    variation = torch.zeros(n, device=embeds.device, dtype=embeds.dtype)
    if n > 1:
        variation[:-1] = 1.0 - (normed[:-1] * normed[1:]).sum(dim=-1)
        variation[-1] = variation[-2]
    return variation


def importance(
    embeds: torch.Tensor,
    attn: torch.Tensor | None,
    *,
    modality: str,
    frames: int | None = None,
    patches: int | None = None,
) -> torch.Tensor:
    """Combined importance: attention centrality plus structural variation.

    ASSUMPTION A2: the paper says the structural cues "are added to" the
    attention-centrality vector, giving no coefficients and no relative weight
    between video's two variation terms. Each signal is normalized and summed
    with weight 1.
    """
    n = embeds.shape[0]
    if attn is None:
        centrality = torch.zeros(n, device=embeds.device, dtype=embeds.dtype)
    else:
        centrality = minmax(attention_centrality(attn).to(embeds.dtype))

    if modality == "visual":
        if frames is None or patches is None:
            raise ValueError("visual importance needs frames and patches")
        temporal, spatial = visual_variation(embeds, frames, patches)
        return centrality + minmax(temporal) + minmax(spatial)
    if modality == "audio":
        return centrality + minmax(audio_variation(embeds))
    raise ValueError(f"unknown modality: {modality}")


def positional_coords(
    *, modality: str, n: int, frames: int | None = None, patches: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """ASSUMPTION A3: coordinates whose Euclidean distance is d_pos.

    Audio is a normalized time index. Video is (normalized frame index,
    normalized patch row, normalized patch column) on a square-ish grid. Each
    axis is normalized to [0,1] independently, per the paper's note that
    distances are "independently normalized to [0,1] within each modality".
    """
    device = device or torch.device("cpu")
    if modality == "audio":
        t = torch.arange(n, device=device, dtype=torch.float32)
        return (t / max(n - 1, 1)).unsqueeze(1)
    if frames is None or patches is None:
        raise ValueError("visual positions need frames and patches")
    side = int(patches ** 0.5) or 1
    f = torch.arange(frames, device=device, dtype=torch.float32).repeat_interleave(patches)
    p = torch.arange(patches, device=device, dtype=torch.float32).repeat(frames)
    row = torch.div(p, side, rounding_mode="floor")
    col = p % side
    return torch.stack(
        [
            f / max(frames - 1, 1),
            row / max(side - 1, 1),
            col / max(side - 1, 1),
        ],
        dim=1,
    )


def positional_distance(coords: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(coords, coords)
    hi = d.max()
    return d / (hi + EPS)


def dpc_knn_scores(dist: torch.Tensor, k: int = DPC_KNN_K) -> torch.Tensor:
    """Appendix B.1: rho_i * delta_i, fully specified by the paper."""
    n = dist.shape[0]
    if n == 0:
        return dist.new_zeros(0)
    k_eff = min(k, max(n - 1, 1))
    neighbours, _ = torch.topk(dist, k=min(k_eff + 1, n), largest=False)
    # drop self-distance at position 0
    neighbours = neighbours[:, 1:] if neighbours.shape[1] > 1 else neighbours
    rho = torch.exp(-(neighbours ** 2).mean(dim=1))

    delta = torch.empty_like(rho)
    rho_max = rho.max()
    for i in range(n):
        higher = rho > rho[i]
        if bool(higher.any()) and rho[i] < rho_max:
            delta[i] = dist[i][higher].min()
        else:
            masked = dist[i].clone()
            masked[i] = -float("inf")
            delta[i] = masked.max()
    return rho * delta


@dataclass
class Stage1Result:
    embeds: torch.Tensor
    kept_index: torch.Tensor
    n_before: int
    n_after: int


def stage1_compress(
    embeds: torch.Tensor,
    attn: torch.Tensor | None,
    *,
    modality: str,
    retention: float,
    frames: int | None = None,
    patches: int | None = None,
) -> Stage1Result:
    """Pre-LLM (post-encoder) compression: dual selection, then merging."""
    n = embeds.shape[0]
    budget = max(int(round(retention * n)), 1)
    if budget >= n:
        return Stage1Result(embeds, torch.arange(n, device=embeds.device), n, n)

    eta = ETA_VISUAL if modality == "visual" else ETA_AUDIO
    tau = TAU_VISUAL if modality == "visual" else TAU_AUDIO

    scores = importance(embeds, attn, modality=modality, frames=frames, patches=patches)
    normed_scores = minmax(scores)

    coords = positional_coords(
        modality=modality, n=n, frames=frames, patches=patches, device=embeds.device
    )
    dpos = positional_distance(coords).to(embeds.dtype)

    k_imp = max(int(round(eta * budget)), 0)
    k_imp = min(k_imp, budget)
    if k_imp > 0:
        top = torch.topk(scores, k=k_imp).indices
    else:
        top = scores.new_zeros(0, dtype=torch.long)

    chosen = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    chosen[top] = True

    remaining_budget = budget - int(chosen.sum().item())
    if remaining_budget > 0:
        rest = (~chosen).nonzero(as_tuple=True)[0]
        if rest.numel() > 0:
            sub = embeds[rest]
            joint = (1.0 - cosine_matrix(sub)) + LAMBDA_POS * dpos[rest][:, rest]
            gamma = dpc_knn_scores(joint, k=DPC_KNN_K)
            take = min(remaining_budget, rest.numel())
            picked = rest[torch.topk(gamma, k=take).indices]
            chosen[picked] = True

    kept_index = chosen.nonzero(as_tuple=True)[0]
    merged = merge_into_representatives(
        embeds=embeds,
        kept_index=kept_index,
        scores=normed_scores,
        dpos=dpos,
        tau=tau,
    )
    return Stage1Result(merged, kept_index, n, int(kept_index.numel()))


def merge_into_representatives(
    *,
    embeds: torch.Tensor,
    kept_index: torch.Tensor,
    scores: torch.Tensor,
    dpos: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Each dropped token folds into its best-matching kept token.

    Phi_ij = cos(x_i, x_j) - tau * d_pos_ij + zeta * s_j, and the update is a
    weighted mean with w_i = (1 + s_i) / 2.
    """
    n = embeds.shape[0]
    dropped = torch.ones(n, dtype=torch.bool, device=embeds.device)
    dropped[kept_index] = False
    dropped_index = dropped.nonzero(as_tuple=True)[0]

    kept = embeds[kept_index].clone()
    if dropped_index.numel() == 0:
        return kept

    cos = cosine_matrix(embeds)[dropped_index][:, kept_index]
    phi = cos - tau * dpos[dropped_index][:, kept_index] + ZETA * scores[kept_index].unsqueeze(0)
    target = phi.argmax(dim=1)

    weights = (1.0 + scores[dropped_index]) / 2.0
    numer = kept.clone()
    denom = torch.ones(kept.shape[0], device=embeds.device, dtype=embeds.dtype)
    numer.index_add_(0, target, embeds[dropped_index] * weights.unsqueeze(1))
    denom.index_add_(0, target, weights)
    return numer / denom.unsqueeze(1)


def stage2_select(
    hidden: torch.Tensor,
    *,
    query_states: torch.Tensor,
    other_modality_states: torch.Tensor | None,
    retention: float = INNER_RETENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Query-conditioned compression inside the LLM, at the insertion layer.

    Returns (compressed_states, kept_index). Pure tensor work: the runner is
    responsible for calling this at the right layer and repairing position ids
    and the KV cache.

    ASSUMPTION A4: `q` and `p` are simple means of their token states.
    """
    n = hidden.shape[0]
    budget = max(int(round(retention * n)), 1)
    if budget >= n:
        return hidden, torch.arange(n, device=hidden.device)

    h = hidden / (hidden.norm(dim=-1, keepdim=True) + EPS)

    q = query_states.mean(dim=0)
    q = q / (q.norm() + EPS)
    text_cos = (h @ (query_states / (query_states.norm(dim=-1, keepdim=True) + EPS)).T)
    textual = torch.maximum(h @ q, text_cos.max(dim=1).values)

    if other_modality_states is not None and other_modality_states.numel() > 0:
        p = other_modality_states.mean(dim=0)
        p = p / (p.norm() + EPS)
        cross = h @ p
    else:
        cross = torch.zeros(n, device=hidden.device, dtype=hidden.dtype)

    within = torch.exp(-(1.0 - (h @ h.T)).mean(dim=1))

    relevance = minmax(textual) + minmax(cross) + minmax(within)
    weight = 1.0 + minmax(relevance)

    selected = [int(relevance.argmax().item())]
    chosen = torch.zeros(n, dtype=torch.bool, device=hidden.device)
    chosen[selected[0]] = True
    diversity = 1.0 - (h @ h[selected[0]])

    while len(selected) < budget:
        masked = torch.where(chosen, torch.full_like(diversity, -float("inf")), weight * diversity)
        nxt = int(masked.argmax().item())
        if masked[nxt] == -float("inf"):
            break
        selected.append(nxt)
        chosen[nxt] = True
        diversity = torch.minimum(diversity, 1.0 - (h @ h[nxt]))

    kept_index = torch.tensor(sorted(selected), device=hidden.device, dtype=torch.long)

    dropped = (~chosen).nonzero(as_tuple=True)[0]
    kept = hidden[kept_index].clone()
    if dropped.numel() > 0:
        sim = (h[dropped] @ h[kept_index].T).clamp(min=0.0)
        target = sim.argmax(dim=1)
        w = sim.max(dim=1).values
        numer = kept.clone()
        denom = torch.ones(kept.shape[0], device=hidden.device, dtype=hidden.dtype)
        numer.index_add_(0, target, hidden[dropped] * w.unsqueeze(1))
        denom.index_add_(0, target, w)
        kept = numer / denom.unsqueeze(1)

    return kept, kept_index
