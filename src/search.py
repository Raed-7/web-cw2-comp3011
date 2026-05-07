"""
Search Engine Module
====================

Query layer over an :class:`~src.indexer.Indexer`.

Two query modes are supported:

* **Single-word**: ``find indifference`` returns every page that
  contains the word, ranked by TF-IDF score.
* **Multi-word**: ``find good friends`` returns the intersection --
  pages that contain *every* query word -- ranked by the sum of
  per-word TF-IDF scores. This matches the boolean-AND semantics
  expected by the assignment brief.

Ranking
-------
We use the classic TF-IDF weighting scheme:

* ``tf(t, d) = freq(t, d) / |d|`` -- term frequency normalised
  by document length, so longer pages do not unfairly dominate.
* ``idf(t) = log(N / df(t))`` -- inverse document frequency, where
  ``N`` is the total number of indexed documents and ``df(t)`` is
  the number of documents containing the term.
* ``score(d, q) = sum over t in q of tf(t, d) * idf(t)``.

This gives rare-but-present terms more weight than common ones
("indifference" beats "the"), which is the standard baseline in
information retrieval.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

from .indexer import Indexer

logger = logging.getLogger(__name__)


class SearchResult(NamedTuple):
    """One ranked page in a result list.

    Using a ``NamedTuple`` keeps the result both immutable and
    tuple-unpackable, which is convenient for callers that just
    want ``url, score = result``.
    """

    url: str
    score: float


class SearchEngine:
    """Query an :class:`Indexer` with boolean AND + TF-IDF ranking.

    The engine deliberately does *not* own the index -- it borrows
    a reference. That separation lets the CLI rebuild or reload
    the index without recreating the search object.

    Attributes:
        indexer: The :class:`Indexer` instance to query.
    """

    def __init__(self, indexer: Indexer) -> None:
        """Initialise with an existing indexer.

        Args:
            indexer: An :class:`Indexer` populated either by
                ``build_from_pages`` or by ``load``.
        """
        self.indexer: Indexer = indexer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, query: str) -> list[SearchResult]:
        """Return ranked pages matching every word in ``query``.

        Tokenisation is delegated to the indexer so it always
        matches the rules used at index time (lowercasing,
        apostrophe handling, etc.).

        Args:
            query: User query, one or more whitespace-separated words.

        Returns:
            A list of :class:`SearchResult` ordered by descending
            TF-IDF score. Empty list if the query is empty or no
            page contains every term.
        """
        terms = self.indexer.tokenize(query)
        if not terms:
            logger.debug("Empty query received.")
            return []

        # Per-term URL sets. Missing terms produce empty sets, which
        # makes the intersection empty -- exactly the AND semantics
        # we want.
        url_sets: list[set[str]] = []
        for term in terms:
            entry = self.indexer.get_entry(term)
            if entry is None:
                logger.debug("Term '%s' not in index.", term)
                # One missing term -> no possible match, short-circuit.
                return []
            url_sets.append(set(entry.keys()))

        # Intersection of all per-term URL sets = pages with every term.
        matching_urls = set.intersection(*url_sets)
        if not matching_urls:
            return []

        # Score each surviving page and sort.
        scored = [
            SearchResult(url, self._score(url, terms))
            for url in matching_urls
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    def print_word(self, word: str) -> str:
        """Format the index entry for ``word`` for CLI display.

        Args:
            word: The word to look up (case-insensitive).

        Returns:
            A multi-line, human-readable string. If the word is
            not in the index, returns a brief explanatory message
            rather than raising -- callers can print it directly.
        """
        entry = self.indexer.get_entry(word)
        if entry is None:
            return f"'{word.lower()}' is not in the index."

        # Build the output line by line so users see structure even
        # when many URLs reference the same word.
        lines = [f"{word.lower()}:"]
        # Sorting URLs gives deterministic, reviewable output.
        for url in sorted(entry):
            stats = entry[url]
            lines.append(
                f"  {url}\n"
                f"    frequency: {stats['frequency']}\n"
                f"    positions: {stats['positions']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    def _score(self, url: str, terms: list[str]) -> float:
        """Compute the summed TF-IDF score for ``terms`` on ``url``.

        This is intentionally simple -- one pass over the query
        terms with O(1) index lookups -- because the index is
        already in memory.

        Args:
            url: Document identifier.
            terms: Tokenised query terms.

        Returns:
            Non-negative TF-IDF total. Higher is more relevant.
        """
        total = 0.0
        for term in terms:
            total += self._tf(term, url) * self._idf(term)
        return total

    def _tf(self, term: str, url: str) -> float:
        """Term frequency normalised by document length.

        Returns ``0.0`` if ``term`` does not occur in ``url`` --
        this should not happen for surviving result pages but the
        guard makes the function safe to call in isolation (and
        keeps unit tests simple).
        """
        entry = self.indexer.get_entry(term)
        if entry is None or url not in entry:
            return 0.0

        doc_length = self.indexer.documents.get(url, {}).get("length", 0)
        if doc_length == 0:
            return 0.0  # Avoid division by zero on empty docs.

        return entry[url]["frequency"] / doc_length

    def _idf(self, term: str) -> float:
        """Inverse document frequency for ``term``.

        We use the smoothed form ``log(N / (df + 1)) + 1`` so the
        score is well-defined even for terms missing from the
        corpus and never collapses to exactly zero (which would
        wipe out any contribution from common words). This is the
        same smoothing used by scikit-learn's :class:`TfidfVectorizer`.
        """
        n = self.indexer.num_documents
        if n == 0:
            return 0.0

        entry = self.indexer.get_entry(term)
        df = len(entry) if entry else 0
        return math.log(n / (df + 1)) + 1