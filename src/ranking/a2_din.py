"""
Phase 3 — Local DIN-style baseline for EB-NeRD.

This module implements a lightweight, CPU-compatible reproduction of the
core idea behind the official RecSys 2024 DIN baseline:

    user history + candidate article
                ↓
        candidate-conditioned
             attention
                ↓
        user interest vector
                ↓
              MLP
                ↓
          click logit

Important:
This is a local reproduction of the DIN architecture, NOT the official
FuxiCTR implementation or the published official score.
"""

from __future__ import annotations

import torch
from torch import nn


class DINModel(nn.Module):
    """
    Lightweight Deep Interest Network for news click prediction.

    Parameters
    ----------
    num_articles:
        Number of distinct articles represented by the embedding table.

    embedding_dim:
        Dimension of each article embedding.

    hidden_dim:
        Hidden dimension used by the prediction MLP.

    dropout:
        Dropout probability.

    padding_idx:
        Reserved article ID used to represent padded history positions.
    """

    def __init__(
        self,
        num_articles: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
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

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Article representation shared by candidate and history articles.
        self.article_embedding = nn.Embedding(
            num_embeddings=num_articles,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # DIN attention network.
        #
        # Input:
        #   history embedding
        #   candidate embedding
        #   element-wise difference
        #   element-wise product
        #
        # Total dimensionality = 4 * embedding_dim.
        self.attention = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Prediction head.
        #
        # We combine:
        #   candidate embedding
        #   attention-weighted user-interest vector
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_article_ids: torch.Tensor,
        candidate_article_ids: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute click logits.

        Parameters
        ----------
        history_article_ids:
            Tensor of shape:

                [batch_size, history_length]

            Padded positions should contain `padding_idx`.

        candidate_article_ids:
            Tensor of shape:

                [batch_size]

        history_mask:
            Optional boolean tensor of shape:

                [batch_size, history_length]

            True = valid history event
            False = padding

            If omitted, padding positions are inferred from
            `padding_idx`.

        Returns
        -------
        torch.Tensor
            Click logits with shape:

                [batch_size]

            Apply `torch.sigmoid()` to obtain click probabilities.
        """

        if history_article_ids.ndim != 2:
            raise ValueError(
                "history_article_ids must have shape "
                "[batch_size, history_length]."
            )

        if candidate_article_ids.ndim != 1:
            raise ValueError(
                "candidate_article_ids must have shape [batch_size]."
            )

        if history_article_ids.shape[0] != candidate_article_ids.shape[0]:
            raise ValueError(
                "History and candidate batch sizes must match."
            )

        # --------------------------------------------------------------
        # 1. Convert article IDs into dense vectors
        # --------------------------------------------------------------

        history_embeddings = self.article_embedding(history_article_ids)
        candidate_embeddings = self.article_embedding(candidate_article_ids)

        # history_embeddings:
        #   [B, H, D]
        #
        # candidate_embeddings:
        #   [B, D]

        # --------------------------------------------------------------
        # 2. Construct candidate-conditioned attention features
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

        # Shape:
        #   [B, H, 4D]

        attention_scores = self.attention(attention_input).squeeze(-1)

        # Shape:
        #   [B, H]

        # --------------------------------------------------------------
        # 3. Mask padded history positions
        # --------------------------------------------------------------

        if history_mask is None:
            history_mask = history_article_ids != self.padding_idx
        else:
            if history_mask.shape != history_article_ids.shape:
                raise ValueError(
                    "history_mask must have the same shape as "
                    "history_article_ids."
                )

            history_mask = history_mask.bool()

        # Padded positions must never receive attention.
        attention_scores = attention_scores.masked_fill(
            ~history_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        # Handle the edge case where an entire history consists of padding.
        no_history = ~history_mask.any(dim=1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        # Explicitly zero out invalid positions.
        attention_weights = attention_weights.masked_fill(
            ~history_mask,
            0.0,
        )

        # --------------------------------------------------------------
        # 4. Build candidate-conditioned user-interest representation
        # --------------------------------------------------------------

        user_interest = torch.sum(
            attention_weights.unsqueeze(-1) * history_embeddings,
            dim=1,
        )

        # Shape:
        #   [B, D]

        # If there is no history, use a zero interest vector.
        user_interest = torch.where(
            no_history.unsqueeze(-1),
            torch.zeros_like(user_interest),
            user_interest,
        )

        # --------------------------------------------------------------
        # 5. Predict click likelihood
        # --------------------------------------------------------------

        prediction_input = torch.cat(
            [
                user_interest,
                candidate_embeddings,
            ],
            dim=-1,
        )

        logits = self.mlp(prediction_input).squeeze(-1)

        return logits

    def predict_proba(
        self,
        history_article_ids: torch.Tensor,
        candidate_article_ids: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return click probabilities instead of logits.
        """

        logits = self.forward(
            history_article_ids=history_article_ids,
            candidate_article_ids=candidate_article_ids,
            history_mask=history_mask,
        )

        return torch.sigmoid(logits)