"""
Phase 5.2 — Retrieval-aware neural re-ranker for EB-NeRD.

This module extends the local DIN-style architecture with retrieval
signals produced by Phase 5.1:

    BM25 score
    BM25 rank
    Semantic score
    Semantic rank
    RRF score

The original Phase 3 DIN implementation is intentionally left unchanged.
This module contains an independent DIN-style implementation so that the
Phase 3 baseline remains reproducible and frozen.
"""

from __future__ import annotations

import torch
from torch import nn


class Phase5Reranker(nn.Module):
    """
    Retrieval-aware DIN-style neural re-ranker.

    The model retains the candidate-conditioned DIN attention mechanism
    and augments the final prediction head with five retrieval features.

    Parameters
    ----------
    num_articles:
        Number of article indices, including reserved index 0.

    embedding_dim:
        Dimension of learned article embeddings.

    hidden_dim:
        Hidden dimension used by attention and prediction MLPs.

    retrieval_dim:
        Number of retrieval features. Defaults to 5.

    dropout:
        Dropout probability.

    padding_idx:
        Reserved article index used for padding/unknown articles.
    """

    def __init__(
        self,
        num_articles: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        retrieval_dim: int = 5,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if num_articles <= 1:
            raise ValueError("num_articles must be greater than 1.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if retrieval_dim <= 0:
            raise ValueError("retrieval_dim must be positive.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.retrieval_dim = retrieval_dim
        self.padding_idx = padding_idx

        # --------------------------------------------------------------
        # Shared article representation
        # --------------------------------------------------------------

        self.article_embedding = nn.Embedding(
            num_embeddings=num_articles,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # --------------------------------------------------------------
        # Candidate-conditioned DIN attention
        # --------------------------------------------------------------

        self.attention = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # --------------------------------------------------------------
        # Retrieval-aware prediction head
        #
        # user interest       = embedding_dim
        # candidate embedding = embedding_dim
        # retrieval features  = retrieval_dim
        #
        # total input:
        #     2 * embedding_dim + retrieval_dim
        # --------------------------------------------------------------

        prediction_input_dim = (
            2 * embedding_dim + retrieval_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                prediction_input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_article_ids: torch.Tensor,
        candidate_article_ids: torch.Tensor,
        retrieval_features: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute click logits.

        Parameters
        ----------
        history_article_ids:
            Tensor of shape [batch_size, history_length].

        candidate_article_ids:
            Tensor of shape [batch_size].

        retrieval_features:
            Tensor of shape [batch_size, retrieval_dim].

            For Phase 5.1, the five columns are:

                [bm25_score,
                 bm25_rank,
                 semantic_score,
                 semantic_rank,
                 rrf_score]

        history_mask:
            Optional boolean tensor of shape
            [batch_size, history_length].

            True = valid history event.
            False = padding.

        Returns
        -------
        torch.Tensor
            Click logits with shape [batch_size].
        """

        # --------------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------------

        if history_article_ids.ndim != 2:
            raise ValueError(
                "history_article_ids must have shape "
                "[batch_size, history_length]."
            )

        if candidate_article_ids.ndim != 1:
            raise ValueError(
                "candidate_article_ids must have shape [batch_size]."
            )

        if retrieval_features.ndim != 2:
            raise ValueError(
                "retrieval_features must have shape "
                "[batch_size, retrieval_dim]."
            )

        batch_size = history_article_ids.shape[0]

        if candidate_article_ids.shape[0] != batch_size:
            raise ValueError(
                "History and candidate batch sizes must match."
            )

        if retrieval_features.shape[0] != batch_size:
            raise ValueError(
                "History and retrieval feature batch sizes must match."
            )

        if retrieval_features.shape[1] != self.retrieval_dim:
            raise ValueError(
                "retrieval_features must have "
                f"{self.retrieval_dim} columns, got "
                f"{retrieval_features.shape[1]}."
            )

        # --------------------------------------------------------------
        # 1. Article embeddings
        # --------------------------------------------------------------

        history_embeddings = self.article_embedding(
            history_article_ids
        )

        candidate_embeddings = self.article_embedding(
            candidate_article_ids
        )

        # history_embeddings:
        #     [B, H, D]
        #
        # candidate_embeddings:
        #     [B, D]

        # --------------------------------------------------------------
        # 2. Candidate-conditioned attention
        # --------------------------------------------------------------

        candidate_expanded = candidate_embeddings.unsqueeze(1)

        candidate_expanded = candidate_expanded.expand(
            -1,
            history_embeddings.shape[1],
            -1,
        )

        attention_input = torch.cat(
            [
                history_embeddings,
                candidate_expanded,
                history_embeddings - candidate_expanded,
                history_embeddings * candidate_expanded,
            ],
            dim=-1,
        )

        attention_scores = self.attention(
            attention_input
        ).squeeze(-1)

        # [B, H]

        # --------------------------------------------------------------
        # 3. History masking
        # --------------------------------------------------------------

        if history_mask is None:
            history_mask = (
                history_article_ids != self.padding_idx
            )
        else:
            if history_mask.shape != history_article_ids.shape:
                raise ValueError(
                    "history_mask must have the same shape as "
                    "history_article_ids."
                )

            history_mask = history_mask.bool()

        attention_scores = attention_scores.masked_fill(
            ~history_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        no_history = ~history_mask.any(dim=1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        attention_weights = attention_weights.masked_fill(
            ~history_mask,
            0.0,
        )

        # --------------------------------------------------------------
        # 4. Candidate-conditioned user interest
        # --------------------------------------------------------------

        user_interest = torch.sum(
            attention_weights.unsqueeze(-1)
            * history_embeddings,
            dim=1,
        )

        # [B, D]

        user_interest = torch.where(
            no_history.unsqueeze(-1),
            torch.zeros_like(user_interest),
            user_interest,
        )

        # --------------------------------------------------------------
        # 5. Retrieval-aware prediction
        # --------------------------------------------------------------

        prediction_input = torch.cat(
            [
                user_interest,
                candidate_embeddings,
                retrieval_features,
            ],
            dim=-1,
        )

        # [B, 2D + retrieval_dim]

        logits = self.mlp(
            prediction_input
        ).squeeze(-1)

        return logits

    def predict_proba(
        self,
        history_article_ids: torch.Tensor,
        candidate_article_ids: torch.Tensor,
        retrieval_features: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return click probabilities instead of logits."""

        logits = self.forward(
            history_article_ids=history_article_ids,
            candidate_article_ids=candidate_article_ids,
            retrieval_features=retrieval_features,
            history_mask=history_mask,
        )

        return torch.sigmoid(logits)