"""
Tests for the CLI in ``src.main``.

The CLI is exercised end-to-end by patching ``input`` to feed a
scripted sequence of commands and capturing what gets printed.
This style of test catches regressions in command parsing, error
messages and the dispatch table that pure unit tests miss.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.indexer import Indexer
from src.main import SearchCLI


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _run_cli(cli: SearchCLI, commands: list[str]) -> str:
    """Drive the CLI with a scripted command list and return output.

    A custom ``input`` raises ``EOFError`` once the script is
    exhausted, mimicking the user pressing Ctrl+D and giving the
    REPL a clean exit path.
    """
    iterator = iter(commands)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    with patch("builtins.input", side_effect=fake_input):
        # Capture stdout via pytest's capsys -- but capsys is a
        # fixture so callers must use ``_run_cli_with_capsys``.
        cli.run()
    return ""


# A pre-populated index file we can ``load`` from in tests.
@pytest.fixture
def index_file(tmp_path: Path) -> Path:
    """Write a small index to disk and return its path."""
    idx = Indexer()
    idx.build_from_pages({
        "http://test.com/a": "good friends are good",
        "http://test.com/b": "indifference is dangerous",
    })
    path = tmp_path / "index.json"
    idx.save(path)
    return path


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestSearchCLIInit:
    def test_starts_with_empty_index(self) -> None:
        cli = SearchCLI(index_path=Path("/tmp/nonexistent.json"))
        # Brand-new CLI -- no index loaded yet.
        assert cli.indexer.num_documents == 0
        assert cli.index_ready is False


# ----------------------------------------------------------------------
# Help / exit / unknown
# ----------------------------------------------------------------------


class TestBasicCommands:
    def test_help_shows_all_commands(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["help", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        # Spot-check that every command is documented.
        for cmd in ["build", "load", "print", "find", "exit"]:
            assert cmd in out

    def test_unknown_command_shows_hint(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["nonsense", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        assert "Unknown command" in out

    def test_empty_input_does_not_error(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["", "  ", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        # Empty input should be silently ignored, not produce errors.
        assert "error" not in out.lower()
        assert "unknown" not in out.lower()

    def test_quit_alias_exits(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        # ``quit`` should behave like ``exit``.
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["quit"]):
            cli.run()  # Should return cleanly.

    def test_eof_exits_cleanly(self, tmp_path: Path) -> None:
        # Ctrl+D from the user must terminate the loop without raising.
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=EOFError):
            cli.run()

    def test_keyboard_interrupt_exits_cleanly(self, tmp_path: Path) -> None:
        # Ctrl+C must also terminate cleanly, not crash.
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            cli.run()


# ----------------------------------------------------------------------
# Commands requiring an index
# ----------------------------------------------------------------------


class TestIndexGating:
    """``find`` and ``print`` must refuse to run before build/load."""

    def test_find_without_index(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["find good", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        assert "No index loaded" in out

    def test_print_without_index(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["print word", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        assert "No index loaded" in out


# ----------------------------------------------------------------------
# load command
# ----------------------------------------------------------------------


class TestLoadCommand:
    def test_load_existing_index(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch("builtins.input", side_effect=["load", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        assert "Index loaded" in out
        # The CLI must enable the index_ready flag on success.
        assert cli.index_ready is True

    def test_load_missing_file_shows_error(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        # Friendly error message rather than a raw traceback.
        cli = SearchCLI(index_path=tmp_path / "missing.json")
        with patch("builtins.input", side_effect=["load", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        assert "No index found" in out
        assert cli.index_ready is False

    def test_load_corrupt_file_shows_error(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        # A truncated/garbage file must not crash the shell.
        bad = tmp_path / "bad.json"
        bad.write_text("definitely not json {{")
        cli = SearchCLI(index_path=bad)
        with patch("builtins.input", side_effect=["load", "exit"]):
            cli.run()
        out = capsys.readouterr().out
        # The dispatch wrapper catches generic exceptions and prefixes
        # the message with "[error]".
        assert "[error]" in out


# ----------------------------------------------------------------------
# find command
# ----------------------------------------------------------------------


class TestFindCommand:
    def test_find_returns_results(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input", side_effect=["load", "find good", "exit"]
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "http://test.com/a" in out
        assert "score:" in out

    def test_find_no_results(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input",
            side_effect=["load", "find nonexistent", "exit"],
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "No pages found" in out

    def test_find_without_arguments_shows_usage(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input", side_effect=["load", "find", "exit"]
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "Usage: find" in out

    def test_find_multi_word(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input",
            side_effect=["load", "find good friends", "exit"],
        ):
            cli.run()
        out = capsys.readouterr().out
        # Page A has both words; page B has neither and must be excluded.
        assert "http://test.com/a" in out
        assert "http://test.com/b" not in out


# ----------------------------------------------------------------------
# print command
# ----------------------------------------------------------------------


class TestPrintCommand:
    def test_print_existing_word(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input",
            side_effect=["load", "print indifference", "exit"],
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "indifference" in out
        assert "http://test.com/b" in out
        assert "frequency" in out

    def test_print_missing_word(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input",
            side_effect=["load", "print nonexistent", "exit"],
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "not in the index" in out

    def test_print_without_arguments(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input", side_effect=["load", "print", "exit"]
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "Usage: print" in out

    def test_print_multi_word_rejected(
        self,
        capsys: pytest.CaptureFixture,
        index_file: Path,
    ) -> None:
        # ``print`` is a single-word command -- the index keys are
        # tokens, not phrases. Reject early to avoid confusing output.
        cli = SearchCLI(index_path=index_file)
        with patch(
            "builtins.input",
            side_effect=["load", "print good friends", "exit"],
        ):
            cli.run()
        out = capsys.readouterr().out
        assert "exactly one word" in out


# ----------------------------------------------------------------------
# build command (mocked crawler -- no network)
# ----------------------------------------------------------------------


class TestBuildCommand:
    """Integration-test ``build`` with a stubbed Crawler.crawl()."""

    @patch("src.main.Crawler")
    def test_build_creates_index(
        self,
        mock_crawler_cls,
        capsys: pytest.CaptureFixture,
        tmp_path: Path,
    ) -> None:
        # ``Crawler()`` returns our stub, whose .crawl() returns
        # a known {url: text} mapping.
        mock_crawler = mock_crawler_cls.return_value
        mock_crawler.crawl.return_value = {
            "http://fake.com/a": "test content for indexing",
            "http://fake.com/b": "more test content here",
        }

        index_path = tmp_path / "index.json"
        cli = SearchCLI(index_path=index_path)
        with patch("builtins.input", side_effect=["build", "exit"]):
            cli.run()

        # The CLI ran the crawler, indexed the results, and saved.
        assert index_path.exists()
        assert cli.index_ready is True
        assert cli.indexer.num_documents == 2

        out = capsys.readouterr().out
        assert "Crawled 2 pages" in out
        assert "Index saved" in out

    @patch("src.main.Crawler")
    def test_build_with_failed_crawl(
        self,
        mock_crawler_cls,
        capsys: pytest.CaptureFixture,
        tmp_path: Path,
    ) -> None:
        # Empty crawl result simulates total network failure.
        mock_crawler = mock_crawler_cls.return_value
        mock_crawler.crawl.return_value = {}

        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch("builtins.input", side_effect=["build", "exit"]):
            cli.run()

        out = capsys.readouterr().out
        assert "No pages were crawled" in out
        # Index never became ready because the build aborted.
        assert cli.index_ready is False


# ----------------------------------------------------------------------
# Full integration scenario
# ----------------------------------------------------------------------


class TestFullScenario:
    """Replay a realistic user session start to finish."""

    @patch("src.main.Crawler")
    def test_build_then_find_then_print(
        self,
        mock_crawler_cls,
        capsys: pytest.CaptureFixture,
        tmp_path: Path,
    ) -> None:
        mock_crawler = mock_crawler_cls.return_value
        mock_crawler.crawl.return_value = {
            "http://x.com/1": "good morning friends",
            "http://x.com/2": "good evening world",
        }

        cli = SearchCLI(index_path=tmp_path / "index.json")
        with patch(
            "builtins.input",
            side_effect=[
                "build",
                "find good",
                "find good friends",
                "print morning",
                "exit",
            ],
        ):
            cli.run()

        out = capsys.readouterr().out
        # "good" matches both pages.
        assert "Found 2 page(s)" in out
        # "good friends" only matches page 1.
        assert "Found 1 page(s)" in out
        # ``print morning`` shows its inverted-index entry.
        assert "morning" in out
        assert "frequency" in out