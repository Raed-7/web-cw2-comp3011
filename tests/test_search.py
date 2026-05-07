"""
Tests for the SearchEngine class.

Each test builds a tiny in-memory index, runs queries against it and
asserts on the ranked results. Because everything lives in memory,
the suite runs in milliseconds.
"""

from __future__ import annotations

import pytest

from src.indexer import Indexer
from src.search import SearchEngine, SearchResult


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def small_index() -> Indexer:
    """Reusable 4-page corpus for ranking tests."""
    idx = Indexer()
    idx.build_from_pages({
        "http://a.com": "good friends are good friends forever",
        "http://b.com": "a friend in need is a friend indeed",
        "http://c.com": "indifference is dangerous and silence is deadly",
        "http://d.com": "good morning my good friends and good neighbours",
    })
    return idx


@pytest.fixture
def search(small_index: Indexer) -> SearchEngine:
    """SearchEngine bound to the ``small_index`` corpus."""
    return SearchEngine(small_index)


# ----------------------------------------------------------------------
# Single-word queries
# ----------------------------------------------------------------------


class TestSingleWordSearch:
    def test_returns_pages_containing_word(
        self, search: SearchEngine
    ) -> None:
        results = search.find("indifference")
        # "indifference" only appears on page c.
        assert len(results) == 1
        assert results[0].url == "http://c.com"

    def test_returns_multiple_pages(self, search: SearchEngine) -> None:
        # "good" appears on pages a and d.
        results = search.find("good")
        urls = [r.url for r in results]
        assert "http://a.com" in urls
        assert "http://d.com" in urls
        assert "http://b.com" not in urls

    def test_word_not_in_index_returns_empty(
        self, search: SearchEngine
    ) -> None:
        # Critical for the CLI -- empty list, not an exception.
        assert search.find("quantum") == []

    def test_results_ordered_by_score_desc(
        self, search: SearchEngine
    ) -> None:
        # Whatever the absolute scores, they must be monotonically
        # non-increasing in the result list.
        results = search.find("good")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------------
# Multi-word queries (boolean AND)
# ----------------------------------------------------------------------


class TestMultiWordSearch:
    def test_intersection_of_pages(self, search: SearchEngine) -> None:
        # Only pages with BOTH "good" AND "friends" should appear.
        results = search.find("good friends")
        urls = {r.url for r in results}
        assert urls == {"http://a.com", "http://d.com"}
        # b has "friend" but not "good"; c has neither.
        assert "http://b.com" not in urls
        assert "http://c.com" not in urls

    def test_one_term_missing_returns_empty(
        self, search: SearchEngine
    ) -> None:
        # If even one query word is absent from the corpus, AND
        # semantics demand zero results.
        assert search.find("good zebra") == []

    def test_three_word_query(self, search: SearchEngine) -> None:
        # Pages must contain *all three* tokens to qualify.
        results = search.find("good morning friends")
        # Only page d has all three words.
        urls = {r.url for r in results}
        assert urls == {"http://d.com"}

    def test_repeated_terms_in_query(self, search: SearchEngine) -> None:
        # "good good" -> tokens are ["good", "good"]; the result set
        # is identical to "good", but the score is doubled because
        # we sum over both occurrences.
        single = search.find("good")
        double = search.find("good good")
        assert {r.url for r in single} == {r.url for r in double}
        for s, d in zip(
            sorted(single, key=lambda r: r.url),
            sorted(double, key=lambda r: r.url),
        ):
            assert d.score == pytest.approx(s.score * 2)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_query(self, search: SearchEngine) -> None:
        # Whitespace-only and empty queries must not raise.
        assert search.find("") == []

    def test_whitespace_only_query(self, search: SearchEngine) -> None:
        assert search.find("   \t  \n  ") == []

    def test_punctuation_only_query(self, search: SearchEngine) -> None:
        # Tokeniser drops everything; nothing left to search.
        assert search.find("!!!???") == []

    def test_case_insensitive_query(self, search: SearchEngine) -> None:
        upper = search.find("INDIFFERENCE")
        lower = search.find("indifference")
        # Same query, different case -> identical results.
        assert {r.url for r in upper} == {r.url for r in lower}

    def test_query_with_punctuation_tokenised_correctly(
        self, search: SearchEngine
    ) -> None:
        # The tokeniser strips ", " before lookup; query treated like
        # plain "good friends".
        results = search.find("good, friends!")
        urls = {r.url for r in results}
        assert urls == {"http://a.com", "http://d.com"}

    def test_search_on_empty_index(self) -> None:
        # An empty corpus should never crash -- just return [].
        empty_idx = Indexer()
        empty_search = SearchEngine(empty_idx)
        assert empty_search.find("anything") == []


# ----------------------------------------------------------------------
# print_word
# ----------------------------------------------------------------------


class TestPrintWord:
    def test_returns_word_entry(self, search: SearchEngine) -> None:
        # The output is the user-visible string for the ``print`` cmd.
        output = search.print_word("indifference")
        assert "indifference" in output
        assert "http://c.com" in output
        assert "frequency" in output
        assert "positions" in output

    def test_missing_word_message(self, search: SearchEngine) -> None:
        output = search.print_word("missingword")
        assert "not in the index" in output
        # The lowercased form should appear -- consistent with how
        # the index stores keys.
        assert "missingword" in output

    def test_case_insensitive(self, search: SearchEngine) -> None:
        # The CLI passes raw user input straight through, so we must
        # tolerate any case.
        upper = search.print_word("INDIFFERENCE")
        lower = search.print_word("indifference")
        assert upper == lower


# ----------------------------------------------------------------------
# TF-IDF ranking
# ----------------------------------------------------------------------


class TestTfIdfRanking:
    """Targeted tests for TF-IDF behaviour, not just empirical scores."""

    def test_denser_pages_rank_higher(self) -> None:
        # Both pages contain "good" twice, but page A is shorter, so
        # its TF is higher and it should rank above page B.
        idx = Indexer()
        idx.build_from_pages({
            "http://short.com": "good good filler",
            "http://long.com": "good good " + "filler " * 50,
        })
        results = SearchEngine(idx).find("good")
        assert results[0].url == "http://short.com"

    def test_rare_terms_have_higher_idf(self) -> None:
        # IDF should boost rare terms over common ones. With 3 docs
        # where "common" appears in all and "rare" in only one, the
        # rare term must yield a strictly higher IDF.
        idx = Indexer()
        idx.build_from_pages({
            "http://a.com": "common rare",
            "http://b.com": "common only",
            "http://c.com": "common only",
        })
        engine = SearchEngine(idx)
        assert engine._idf("rare") > engine._idf("common")

    def test_score_is_zero_for_term_not_in_doc(
        self, search: SearchEngine
    ) -> None:
        # Defensive guard inside _tf -- callers should still get a
        # well-defined number, not a KeyError.
        assert search._tf("missingword", "http://a.com") == 0.0

    def test_score_with_zero_length_doc(self) -> None:
        # An empty document must not cause division-by-zero.
        idx = Indexer()
        idx.add_document("http://empty.com", "")
        engine = SearchEngine(idx)
        # Behaviour is "no division-by-zero crash"; zero is fine.
        assert engine._tf("anything", "http://empty.com") == 0.0


# ----------------------------------------------------------------------
# SearchResult NamedTuple
# ----------------------------------------------------------------------


class TestSearchResult:
    """The result type is part of the public API; verify its shape."""

    def test_is_unpackable(self) -> None:
        # ``url, score = result`` is a documented usage pattern.
        result = SearchResult("http://x.com", 0.5)
        url, score = result
        assert url == "http://x.com"
        assert score == 0.5

    def test_field_access(self) -> None:
        result = SearchResult("http://x.com", 0.5)
        assert result.url == "http://x.com"
        assert result.score == 0.5