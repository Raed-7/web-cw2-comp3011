"""
Command-Line Interface
======================

Interactive shell for the COMP3011 search engine.

Supported commands
------------------
=========  ==================================================
build      Crawl the target website, build the inverted index
           and persist it to the data directory.
load       Load a previously built index from disk.
print W    Pretty-print the inverted-index entry for word *W*.
find Q...  Find every page that contains *all* query words,
           ranked by TF-IDF.
help       Show this help text.
exit       Quit the program (Ctrl+D / Ctrl+C also work).
=========  ==================================================

Usage
-----
From the project root::

    python -m src.main

The shell loop is intentionally simple: parse the first whitespace
token as the command, treat the rest as arguments. A small dispatch
table maps commands to handler methods.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .crawler import Crawler
from .indexer import Indexer
from .search import SearchEngine

# Default location of the persisted index. The data/ directory is
# checked into the repository (with a ``.gitkeep``) so save() can
# always write here without surprises.
DEFAULT_INDEX_PATH: Path = Path("data") / "index.json"

# Friendly text shown by the ``help`` command.
HELP_TEXT: str = (
    "Commands:\n"
    "  build         Crawl the website and build the index\n"
    "  load          Load a previously built index from disk\n"
    "  print <word>  Show the index entry for a single word\n"
    "  find <words>  Find pages containing all given words\n"
    "  help          Show this help message\n"
    "  exit          Quit"
)


class SearchCLI:
    """Stateful REPL wrapping the crawler, indexer and search engine.

    The CLI owns the long-lived :class:`Indexer` so that ``build``
    and ``load`` mutate it in place, and ``print``/``find`` always
    see the latest state.

    Attributes:
        index_path: Where the JSON index is read from / written to.
        indexer: The shared inverted-index instance.
        search: Search engine bound to ``indexer``.
        index_ready: Becomes ``True`` after a successful build/load,
            used to gate ``print`` and ``find``.
    """

    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        """Initialise the CLI with empty state.

        Args:
            index_path: File system location for the index JSON.
        """
        self.index_path: Path = index_path
        self.indexer: Indexer = Indexer()
        self.search: SearchEngine = SearchEngine(self.indexer)
        self.index_ready: bool = False

    # ------------------------------------------------------------------
    # REPL loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the interactive command loop until the user exits.

        The method swallows ``EOFError`` (Ctrl+D) and
        ``KeyboardInterrupt`` (Ctrl+C) so quitting the shell never
        looks like a crash.
        """
        print("COMP3011 Search Engine. Type 'help' for commands.")

        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Newline keeps the next shell prompt tidy.
                print()
                break

            if not raw:
                continue  # Ignore empty input -- avoids an error message.

            # Split on the first whitespace run: command + remainder.
            parts = raw.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command in {"exit", "quit"}:
                break

            try:
                self._dispatch(command, args)
            except Exception as exc:
                # Defensive: never let a handler crash the shell.
                # The full traceback is logged for debugging while
                # the user sees a concise message.
                logger.exception("Command '%s' failed", command)
                print(f"[error] {exc}")

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, command: str, args: str) -> None:
        """Route a parsed command to the appropriate handler."""
        handlers = {
            "build": lambda _: self._cmd_build(),
            "load": lambda _: self._cmd_load(),
            "print": self._cmd_print,
            "find": self._cmd_find,
            "help": lambda _: print(HELP_TEXT),
        }

        handler = handlers.get(command)
        if handler is None:
            print(
                f"Unknown command: '{command}'. Type 'help' for the "
                "command list."
            )
            return

        handler(args)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_build(self) -> None:
        """Run the crawler, build the index and persist it."""
        # The site has ~50-80 pages once tag/author pages are
        # included. With the mandatory 6s politeness window, this
        # comes out to several minutes of wall-clock time.
        print(
            "Crawling website. This will take roughly 5-10 minutes "
            "due to the mandatory 6-second politeness window between "
            "requests. Progress will be shown below."
        )
        print()
 
        crawler = Crawler()
        pages = crawler.crawl()
 
        if not pages:
            # The crawler returns an empty dict if every fetch failed.
            print("[error] No pages were crawled. Check your network "
                  "connection and try again.")
            return
 
        print()
        print(f"Crawled {len(pages)} pages. Building index...")

        # Reset the in-place indexer so a re-build never carries over
        # stale entries from a previous run.
        self.indexer.index.clear()
        self.indexer.documents.clear()
        self.indexer.build_from_pages(pages)

        self.indexer.save(self.index_path)
        self.index_ready = True

        print(
            f"Done. Indexed {self.indexer.vocabulary_size} unique words "
            f"across {self.indexer.num_documents} pages."
        )
        print(f"Index saved to {self.index_path}.")

    def _cmd_load(self) -> None:
        """Load an existing index from disk."""
        try:
            self.indexer.load(self.index_path)
        except FileNotFoundError:
            # Convert to a friendly message -- the user does not need
            # the full Python traceback for an expected failure.
            print(
                f"[error] No index found at {self.index_path}. "
                "Run 'build' first."
            )
            return
        except ValueError as exc:
            print(f"[error] Index file is corrupt: {exc}")
            return

        self.index_ready = True
        print(
            f"Index loaded: {self.indexer.num_documents} pages, "
            f"{self.indexer.vocabulary_size} unique words."
        )

    def _cmd_print(self, args: str) -> None:
        """Pretty-print the index entry for a single word."""
        if not self._require_index():
            return

        word = args.strip()
        if not word:
            print("Usage: print <word>")
            return

        # Multi-word ``print`` is meaningless -- the index keys are
        # single tokens. Catch this early to avoid confusion.
        if len(word.split()) > 1:
            print("'print' takes exactly one word.")
            return

        print(self.search.print_word(word))

    def _cmd_find(self, args: str) -> None:
        """Search the index for pages containing all query words."""
        if not self._require_index():
            return

        query = args.strip()
        if not query:
            print("Usage: find <word> [more words...]")
            return

        results = self.search.find(query)
        if not results:
            print(f"No pages found matching '{query}'.")
            return

        print(f"Found {len(results)} page(s) matching '{query}':")
        # Enumerate from 1 -- humans count from 1 in result lists.
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. {result.url}  (score: {result.score:.4f})")

    # ------------------------------------------------------------------
    # Shared validation
    # ------------------------------------------------------------------

    def _require_index(self) -> bool:
        """Print a hint and return ``False`` when the index is empty.

        Returns:
            ``True`` if the index is ready to query, else ``False``.
        """
        if not self.index_ready:
            print(
                "[error] No index loaded. Run 'build' or 'load' first."
            )
            return False
        return True


# ----------------------------------------------------------------------
# Module entry point
# ----------------------------------------------------------------------

# Logger is configured here -- once -- when the module is run as a
# program. Importing the module elsewhere does not change logging.
logger = logging.getLogger(__name__)


def main() -> int:
    """Entry point used by ``python -m src.main``.

    Returns:
        Process exit code (0 on normal exit).
    """
    # WARNING level keeps everyday output clean; the crawler/indexer
    # log at INFO/DEBUG which would otherwise spam the shell.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cli = SearchCLI()
    cli.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
