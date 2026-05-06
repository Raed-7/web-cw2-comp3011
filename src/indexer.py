"""
Indexer Module
==============

Builds and persists an inverted index over a collection of web pages.

The index is the heart of the search engine. For every distinct word
encountered we store, per page, both how many times it appeared
(``frequency``) and the token positions at which it appeared
(``positions``). Frequency drives ranking; positions enable future
phrase- and proximity-based queries.

Index schema
------------
``index`` (dict)::

    {
        "good": {
            "http://quotes.toscrape.com/page/1/": {
                "frequency": 3,
                "positions": [12, 47, 89]
            },
            ...
        },
        ...
    }

``documents`` (dict) -- per-page metadata used by ranking::

    {
        "http://quotes.toscrape.com/page/1/": {"length": 124},
        ...
    }

The whole structure is serialised as JSON so the file can be
inspected, diffed and version-controlled.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Indexer:
    """An inverted-index builder and persistence layer.

    The indexer is intentionally agnostic of *where* the page text
    came from -- it accepts any ``(url, text)`` pair, which keeps it
    easy to unit-test without a real network.

    Attributes:
        index: The inverted index (see module docstring).
        documents: Per-document metadata (length in tokens).
    """

    # Pre-compiled token pattern: a "word" is a run of letters/digits
    # optionally containing apostrophes (so ``don't`` stays one token
    # rather than splitting into ``don`` + ``t``). Compiling once at
    # class load time avoids re-compilation on every call.
    _TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")

    def __init__(self) -> None:
        """Initialise an empty indexer."""
        self.index: dict[str, dict[str, dict]] = {}
        self.documents: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Split raw text into normalised, lowercase tokens.

        We lowercase before matching so the regex only has to consider
        a single case. Returning a list (not a set) preserves order,
        which is required for positional indexing.

        Args:
            text: Raw text to tokenise.

        Returns:
            Ordered list of tokens.

        Examples:
            >>> Indexer.tokenize("Don't think -- act!")
            ["don't", 'think', 'act']
        """
        return cls._TOKEN_RE.findall(text.lower())

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def add_document(self, url: str, text: str) -> None:
        """Tokenise ``text`` and merge its statistics into the index.

        If the same URL is added twice, the previous entry is
        overwritten so re-indexing produces a consistent result.

        Args:
            url: Page URL -- used as the document identifier.
            text: Raw page text.
        """
        # Remove any previous entries for this URL so we never end up
        # with stale positional data after a re-crawl.
        self._remove_url_from_index(url)

        tokens = self.tokenize(text)
        self.documents[url] = {"length": len(tokens)}

        # Walk the token list once, accumulating positions per word.
        # ``enumerate`` gives us the position needed by the index.
        for position, token in enumerate(tokens):
            word_entry = self.index.setdefault(token, {})
            doc_entry = word_entry.setdefault(
                url, {"frequency": 0, "positions": []}
            )
            doc_entry["frequency"] += 1
            doc_entry["positions"].append(position)

        logger.debug("Indexed %s (%d tokens)", url, len(tokens))

    def build_from_pages(self, pages: dict[str, str]) -> None:
        """Bulk-index a mapping of ``url -> text``.

        This is the typical entry point after a crawl: the crawler
        returns ``{url: text}`` and we feed the whole mapping in.

        Args:
            pages: Mapping returned by ``Crawler.crawl()``.
        """
        for url, text in pages.items():
            self.add_document(url, text)

        logger.info(
            "Index built: %d documents, %d unique words.",
            len(self.documents),
            len(self.index),
        )

    # ------------------------------------------------------------------
    # Querying helpers (used by SearchEngine)
    # ------------------------------------------------------------------

    def get_entry(self, word: str) -> Optional[dict[str, dict]]:
        """Return the index entry for ``word``, or ``None`` if absent.

        The lookup is case-insensitive: callers can pass user input
        without lowercasing it themselves.

        Args:
            word: The word to look up.

        Returns:
            The per-URL stats dict for ``word``, or ``None``.
        """
        return self.index.get(word.lower())

    @property
    def num_documents(self) -> int:
        """Total number of indexed documents (for IDF calculations)."""
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct words currently in the index."""
        return len(self.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str | Path) -> None:
        """Serialise the index to a single JSON file.

        Both ``index`` and ``documents`` are written so a subsequent
        ``load`` reconstitutes the indexer exactly.

        Args:
            filepath: Destination path. Parent directories are
                created automatically if missing.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"index": self.index, "documents": self.documents}

        # ``indent=2`` keeps the file diffable in Git -- worth the
        # small size hit for a coursework-scale index.
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("Index saved to %s", path)

    def load(self, filepath: str | Path) -> None:
        """Restore the index from a previously saved JSON file.

        Args:
            filepath: Path to a JSON file produced by ``save``.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
            ValueError: If the file is not a valid index payload.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"No index file found at {path}. "
                "Run the 'build' command first."
            )

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        # Defensive shape check -- a corrupt file should fail loudly
        # rather than yielding silently empty searches later.
        if not isinstance(payload, dict) or "index" not in payload:
            raise ValueError(f"{path} is not a valid index file.")

        self.index = payload.get("index", {})
        self.documents = payload.get("documents", {})

        logger.info(
            "Index loaded from %s (%d documents, %d unique words).",
            path,
            self.num_documents,
            self.vocabulary_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_url_from_index(self, url: str) -> None:
        """Strip any existing entries for ``url`` from the index.

        Used by ``add_document`` so re-indexing the same page does
        not double-count its tokens.
        """
        if url not in self.documents:
            return  # Nothing to clean up.

        # Iterate over a copy of keys because we may mutate the dict.
        for word in list(self.index.keys()):
            if url in self.index[word]:
                del self.index[word][url]
                # Drop empty word buckets so vocabulary_size stays
                # accurate.
                if not self.index[word]:
                    del self.index[word]

        del self.documents[url]