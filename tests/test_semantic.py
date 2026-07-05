import numpy as np

from gist.core.schemas import Candidate
from gist.core.scoring import lexical_relevance
from gist.core.semantic import SemanticTextScorer


class FakeEmbedder:
    """Deterministic toy embeddings: map known phrases to fixed vectors so we can
    exercise the semantic-vs-lexical contrast without loading a real model."""

    _VECS = {
        "query": [1.0, 0.0, 0.0],
        # true answer — semantically aligned with the query, but shares no
        # content words with it.
        "paraphrase": [0.96, 0.28, 0.0],
        # distractor — shares many query keywords but is off-topic (nearly
        # orthogonal to the query vector).
        "distractor": [0.1, 0.99, 0.0],
    }

    def encode(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            if low.startswith("how do top builders use ai"):
                out.append(self._VECS["query"])
            elif "automation replaces vast teams" in low:
                out.append(self._VECS["paraphrase"])
            else:
                out.append(self._VECS["distractor"])
        return np.array(out, dtype=float)


def test_semantic_ranks_paraphrase_above_keyword_distractor():
    query = "How do top builders use AI to do the work of hundreds of engineers?"
    # true answer: semantically on-point, ~no shared content words
    paraphrase = "automation replaces vast teams of staff"
    # distractor: shares top/builders/use/engineers/work but off-topic
    distractor = "top builders use scaffolding while engineers do site work"

    scorer = SemanticTextScorer(embedder=FakeEmbedder())
    s_para, s_dist = scorer.score_texts(query, [paraphrase, distractor])
    assert s_para > s_dist  # semantic picks the true answer span

    # Confirm this is exactly the case lexical overlap gets WRONG:
    para_c = Candidate(id="p", timestamp_seconds=1.0, text=paraphrase)
    dist_c = Candidate(id="d", timestamp_seconds=2.0, text=distractor)
    assert lexical_relevance(query, dist_c) > lexical_relevance(query, para_c)


def test_score_candidates_sets_saliency():
    scorer = SemanticTextScorer(embedder=FakeEmbedder())
    cands = [
        Candidate(id="p", timestamp_seconds=1.0,
                  text="automation replaces vast teams of staff"),
        Candidate(id="d", timestamp_seconds=2.0, text="unrelated chatter"),
    ]
    scored = scorer.score_candidates("How do top builders use AI to do the work?", cands)
    assert scored[0].saliency_score is not None
    assert scored[0].saliency_score > scored[1].saliency_score


def test_empty_and_blank_handled():
    scorer = SemanticTextScorer(embedder=FakeEmbedder())
    assert scorer.score_texts("q", []) == []
    assert scorer.score_texts("", ["a"]) == [0.0]
    assert scorer.score_texts("q", ["", "  "]) == [0.0, 0.0]
