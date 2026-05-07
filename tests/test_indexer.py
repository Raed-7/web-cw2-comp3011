"""
Tests for the Indexer class.

The indexer is pure Python with no I/O outside ``save`` / ``load``,
so most tests just construct an Indexer, feed it strings and assert
on the resulting data structures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.indexer import Indexer


# ----------------------------------------------------------------------
# Tokenisation
# ----------------------------------------------------------------------


class TestTokenize:
    """Tokenisation rules drive everything downstream -- test thoroughly."""

    def test_lowercases(self) -> None:
        # Case-insensitivity is required by the brief.
        assert Indexer.tokenize("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert Indexer.tokenize("Hello, world!") == ["hello", "world"]

    def test_keeps_apostrophes_inside_words(self) -> None:
        # Splitting "don't" into "don" + "t" would harm search quality.
        assert Indexer.tokenize("Don't can't won't") == [
            "don't", "can't", "won't"
        ]

    def test_handles_multiple_spaces(self) -> None:
        # Stray whitespace must not produce empty tokens.
        assert Indexer.tokenize("   hello   world   ") == ["hello", "world"]

    def test_handles_numbers(self) -> None:
        # Year and price tokens are useful search terms; keep them.
        assert Indexer.tokenize("Year 2024 has 365 days") == [
            "year", "2024", "has", "365", "days"
        ]

    def test_empty_string(self) -> None:
        assert Indexer.tokenize("") == []

    def test_only_punctuation(self) -> None:
        # Nothing matches the [a-z0-9...] character class.
        assert Indexer.tokenize("!!!???...") == []

    def test_preserves_token_order(self) -> None:
        # Order matters for positional indexing.
        text = "the quick brown fox"
        assert Indexer.tokenize(text) == ["the", "quick", "brown", "fox"]


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestIndexerInit:
    def test_starts_empty(self) -> None:
        idx = Indexer()
        assert idx.index == {}
        assert idx.documents == {}
        assert idx.num_documents == 0
        assert idx.vocabulary_size == 0


# ----------------------------------------------------------------------
# Adding documents
# ----------------------------------------------------------------------


class TestAddDocument:
    """Single-document indexing must produce correct frequency & positions."""

    def test_single_document_basic(self) -> None:
        idx = Indexer()
        idx.add_document("http://example.com", "hello world hello")

        assert idx.num_documents == 1
        # "hello" twice, "world" once.
        assert idx.vocabulary_size == 2
        assert "hello" in idx.index
        assert "world" in idx.index

    def test_frequency_counted_correctly(self) -> None:
        idx = Indexer()
        idx.add_document("http://example.com", "good good good bad")
        good_entry = idx.get_entry("good")
        assert good_entry["http://example.com"]["frequency"] == 3

    def test_positions_recorded_in_order(self) -> None:
        idx = Indexer()
        idx.add_document("http://example.com", "a b a c a")
        # "a" is at indices 0, 2, 4 in the token list.
        assert idx.get_entry("a")["http://example.com"]["positions"] == [0, 2, 4]

    def test_document_length_recorded(self) -> None:
        # Length is required for TF normalisation in TF-IDF.
        idx = Indexer()
        idx.add_document("http://example.com", "one two three four five")
        assert idx.documents["http://example.com"]["length"] == 5

    def test_case_insensitive_indexing(self) -> None:
        idx = Indexer()
        idx.add_document("http://example.com", "Apple APPLE apple")
        # All three forms collapse to the same key.
        assert idx.get_entry("apple")["http://example.com"]["frequency"] == 3
        # Lookup by uppercase should still find it.
        assert idx.get_entry("APPLE") is not None

    def test_reindexing_same_url_does_not_double_count(self) -> None:
        # Re-running ``build`` must not corrupt counts. We reset the
        # entry in-place rather than appending.
        idx = Indexer()
        idx.add_document("http://example.com", "hello world")
        idx.add_document("http://example.com", "hello world")
        assert idx.get_entry("hello")["http://example.com"]["frequency"] == 1


# ----------------------------------------------------------------------
# Bulk indexing
# ----------------------------------------------------------------------


class TestBuildFromPages:
    def test_indexes_multiple_pages(self) -> None:
        idx = Indexer()
        idx.build_from_pages({
            "http://a.com": "good morning",
            "http://b.com": "good evening",
        })
        # "good" appears once in each of two pages.
        good_entry = idx.get_entry("good")
        assert len(good_entry) == 2
        assert "http://a.com" in good_entry
        assert "http://b.com" in good_entry

    def test_empty_pages_dict(self) -> None:
        # Nothing to index -- not an error, just a no-op.
        idx = Indexer()
        idx.build_from_pages({})
        assert idx.num_documents == 0


# ----------------------------------------------------------------------
# Lookup helpers
# ----------------------------------------------------------------------


class TestGetEntry:
    def test_returns_none_for_missing_word(self) -> None:
        # Callers (the search engine) rely on None to short-circuit.
        idx = Indexer()
        idx.add_document("http://x.com", "hello")
        assert idx.get_entry("missing") is None

    def test_lookup_is_case_insensitive(self) -> None:
        # Users should be able to pass raw input without lowercasing.
        idx = Indexer()
        idx.add_document("http://x.com", "Hello")
        assert idx.get_entry("HELLO") is not None
        assert idx.get_entry("hello") is not None


# ----------------------------------------------------------------------
# Persistence (save / load)
# ----------------------------------------------------------------------


class TestSaveAndLoad:
    """Round-trip the index through JSON to catch serialisation bugs."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        idx = Indexer()
        idx.add_document("http://x.com", "hello world")

        filepath = tmp_path / "index.json"
        idx.save(filepath)
        assert filepath.exists()

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        # The first ``build`` ever runs against a missing data/ dir.
        idx = Indexer()
        idx.add_document("http://x.com", "hello")

        filepath = tmp_path / "deep" / "nested" / "index.json"
        idx.save(filepath)
        assert filepath.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        # Anyone can open the file and inspect the index -- one of
        # the reasons we chose JSON over pickle.
        idx = Indexer()
        idx.add_document("http://x.com", "hello world")

        filepath = tmp_path / "index.json"
        idx.save(filepath)

        with filepath.open() as fh:
            data = json.load(fh)
        assert "index" in data
        assert "documents" in data

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        idx = Indexer()
        idx.add_document("http://x.com", "hello world hello")
        idx.add_document("http://y.com", "foo bar")

        filepath = tmp_path / "index.json"
        idx.save(filepath)

        loaded = Indexer()
        loaded.load(filepath)

        # The dicts must be exactly equal post-roundtrip.
        assert loaded.index == idx.index
        assert loaded.documents == idx.documents

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        # The CLI relies on this exception to print a friendly message
        # rather than displaying an empty result.
        with pytest.raises(FileNotFoundError):
            Indexer().load(tmp_path / "nope.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("this is not json {{{")
        with pytest.raises(json.JSONDecodeError):
            Indexer().load(bad_file)

    def test_load_wrong_shape_raises(self, tmp_path: Path) -> None:
        # A valid JSON file that is not an index payload should fail
        # loudly rather than yield silently empty searches.
        bad_file = tmp_path / "wrong.json"
        bad_file.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError):
            Indexer().load(bad_file)

    def test_save_load_with_unicode_url_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        # ASCII tokens are stored, but the URL contains non-ASCII --
        # ``ensure_ascii=False`` in save() must not raise on this.
        idx = Indexer()
        idx.add_document("http://example.com/café", "hello world")

        filepath = tmp_path / "index.json"
        idx.save(filepath)

        loaded = Indexer()
        loaded.load(filepath)

        # The non-ASCII URL survives the JSON round-trip intact.
        assert "http://example.com/café" in loaded.documents