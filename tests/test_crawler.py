"""
Tests for the Crawler class.

Strategy
--------
We never hit the real network. ``requests.get`` is patched with
``unittest.mock`` so each test deterministically controls the
HTML, status code and timing. This makes the suite fast (no 6s
politeness wait) and reliable (works offline).

The politeness logic is exercised separately by mocking ``time.sleep``
so we can assert *that* the wait happened without actually waiting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.crawler import Crawler


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    """Build a minimal stand-in for a ``requests.Response``."""
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    return response


SAMPLE_HTML = """
<html>
  <head><title>Quotes</title></head>
  <body>
    <h1>Hello World</h1>
    <p>Be yourself; everyone else is taken.</p>
    <a href="/page/2/">Next</a>
    <a href="https://twitter.com/external">External</a>
    <a href="mailto:test@example.com">Email</a>
    <a href="#section">Anchor</a>
    <script>console.log("polluting JS")</script>
    <style>.cls{color:red}</style>
  </body>
</html>
"""


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestCrawlerConstruction:
    """The crawler should initialise with sensible defaults."""

    def test_default_values(self) -> None:
        crawler = Crawler()
        assert crawler.base_url == "https://quotes.toscrape.com/"
        assert crawler.delay == 6.0
        assert crawler.timeout == 10.0
        assert crawler.visited == set()
        assert crawler.pages == {}

    def test_allowed_netloc_extracted_from_base_url(self) -> None:
        crawler = Crawler(base_url="https://example.com/start")
        assert crawler._allowed_netloc == "example.com"

    def test_warns_when_delay_below_brief_minimum(self, caplog) -> None:
        # Anything < 6s violates the assignment brief; the crawler
        # should still construct but log a warning.
        Crawler(delay=2.0)
        assert any("politeness window" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# Link extraction
# ----------------------------------------------------------------------


class TestLinkExtraction:
    """Internal-link filtering must reject every external scheme."""

    def test_extracts_internal_links_only(self) -> None:
        crawler = Crawler()
        links = crawler.extract_links(
            SAMPLE_HTML, "https://quotes.toscrape.com/"
        )
        # /page/2/ resolves to the internal domain, the others do not.
        assert "https://quotes.toscrape.com/page/2/" in links
        assert all("twitter.com" not in link for link in links)
        assert all(not link.startswith("mailto:") for link in links)

    def test_resolves_relative_urls(self) -> None:
        html = '<a href="page/3/">Three</a>'
        crawler = Crawler()
        links = crawler.extract_links(
            html, "https://quotes.toscrape.com/page/2/"
        )
        # urljoin should produce an absolute URL relative to the
        # current page, not the base.
        assert "https://quotes.toscrape.com/page/2/page/3/" in links

    def test_strips_url_fragments(self) -> None:
        html = '<a href="/page/1/#top">Top</a>'
        crawler = Crawler()
        links = crawler.extract_links(
            html, "https://quotes.toscrape.com/"
        )
        # Fragments must be stripped to avoid re-crawling the same
        # page once per anchor.
        assert "https://quotes.toscrape.com/page/1/" in links
        assert all("#" not in link for link in links)

    def test_returns_empty_set_for_no_links(self) -> None:
        crawler = Crawler()
        assert crawler.extract_links("<p>no links here</p>", "https://x/") == set()


# ----------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------


class TestTextExtraction:
    """``extract_text`` must drop noise that would pollute the index."""

    def test_strips_script_and_style(self) -> None:
        crawler = Crawler()
        text = crawler.extract_text(SAMPLE_HTML)
        # Visible content survives.
        assert "Hello World" in text
        # Non-content elements are removed entirely.
        assert "console.log" not in text
        assert "color:red" not in text

    def test_collapses_whitespace(self) -> None:
        html = "<p>hello\n\n\n   world\t\ttest</p>"
        crawler = Crawler()
        # Multiple whitespace runs should fold into single spaces so
        # later tokenisation does not produce empty tokens.
        assert crawler.extract_text(html) == "hello world test"

    def test_empty_html(self) -> None:
        crawler = Crawler()
        assert crawler.extract_text("") == ""


# ----------------------------------------------------------------------
# Fetching (with HTTP mocked)
# ----------------------------------------------------------------------


class TestFetch:
    """``fetch`` should be defensive against every network failure mode."""

    @patch("src.crawler.requests.get")
    def test_successful_fetch(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response("<html>ok</html>")
        crawler = Crawler(delay=0.0)  # No wait in tests.
        assert crawler.fetch("https://example.com") == "<html>ok</html>"

    @patch("src.crawler.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        # The crawler must not raise on 4xx/5xx, just skip the page.
        mock_get.return_value = _mock_response("Not Found", status_code=404)
        crawler = Crawler(delay=0.0)
        assert crawler.fetch("https://example.com") is None

    @patch("src.crawler.requests.get")
    def test_request_exception_returns_none(self, mock_get: MagicMock) -> None:
        # ConnectionError / Timeout / SSLError all derive from
        # RequestException -- one handler covers them all.
        mock_get.side_effect = requests.exceptions.ConnectionError("boom")
        crawler = Crawler(delay=0.0)
        assert crawler.fetch("https://example.com") is None

    @patch("src.crawler.requests.get")
    def test_timeout_returns_none(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        crawler = Crawler(delay=0.0)
        assert crawler.fetch("https://example.com") is None

    @patch("src.crawler.requests.get")
    def test_user_agent_header_sent(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response("<html></html>")
        crawler = Crawler(delay=0.0, user_agent="TestBot/1.0")
        crawler.fetch("https://example.com")

        # The crawler must identify itself -- some sites block
        # default ``python-requests`` UAs.
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["User-Agent"] == "TestBot/1.0"


# ----------------------------------------------------------------------
# Politeness window
# ----------------------------------------------------------------------


class TestPoliteness:
    """The 6-second window is a hard requirement of the brief."""

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.time.time")
    def test_no_sleep_on_first_request(
        self, mock_time: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_time.return_value = 1000.0
        crawler = Crawler(delay=6.0)
        crawler._respect_politeness()
        # First request fires immediately -- no prior request to wait for.
        mock_sleep.assert_not_called()

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.time.time")
    def test_sleeps_when_called_too_soon(
        self, mock_time: MagicMock, mock_sleep: MagicMock
    ) -> None:
        # Pretend the previous request finished 1 second ago.
        mock_time.return_value = 1001.0
        crawler = Crawler(delay=6.0)
        crawler._last_request_time = 1000.0

        crawler._respect_politeness()

        # We must wait the *remaining* 5 seconds, not a full 6.
        mock_sleep.assert_called_once()
        wait_time = mock_sleep.call_args[0][0]
        assert wait_time == pytest.approx(5.0)

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.time.time")
    def test_no_sleep_when_window_already_elapsed(
        self, mock_time: MagicMock, mock_sleep: MagicMock
    ) -> None:
        # Pretend it's been 10 seconds -- well past the 6s window.
        mock_time.return_value = 1010.0
        crawler = Crawler(delay=6.0)
        crawler._last_request_time = 1000.0

        crawler._respect_politeness()
        # No need to wait -- the window has already passed.
        mock_sleep.assert_not_called()


# ----------------------------------------------------------------------
# Domain filtering
# ----------------------------------------------------------------------


class TestDomainFiltering:
    """Only same-domain http(s) URLs should be considered internal."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://quotes.toscrape.com/page/1/", True),
            ("http://quotes.toscrape.com/", True),
            ("https://twitter.com/share", False),
            ("mailto:test@test.com", False),
            ("javascript:void(0)", False),
            ("tel:+447777777777", False),
            ("ftp://quotes.toscrape.com/file", False),
        ],
    )
    def test_is_internal(self, url: str, expected: bool) -> None:
        crawler = Crawler()
        assert crawler._is_internal(url) is expected


# ----------------------------------------------------------------------
# End-to-end crawl (still mocked)
# ----------------------------------------------------------------------


class TestCrawlIntegration:
    """End-to-end test of crawl() with a mocked two-page site."""

    @patch("src.crawler.time.sleep")  # Skip real waits.
    @patch("src.crawler.requests.get")
    def test_crawls_multiple_pages(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        page1 = """
        <html><body>
          <p>Page one content</p>
          <a href="/page/2/">Next</a>
        </body></html>
        """
        page2 = """
        <html><body>
          <p>Page two content</p>
          <a href="/">Home</a>
        </body></html>
        """

        # First call returns page1; subsequent calls return page2.
        # ``side_effect`` as a list pops responses in order.
        mock_get.side_effect = [
            _mock_response(page1),
            _mock_response(page2),
        ]

        crawler = Crawler(
            base_url="https://quotes.toscrape.com/", delay=0.0
        )
        pages = crawler.crawl()

        assert len(pages) == 2
        # Both URLs were visited and we got their plain text back.
        assert any("Page one content" in v for v in pages.values())
        assert any("Page two content" in v for v in pages.values())

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_handles_failed_fetch_gracefully(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        # Failures must not abort the crawl -- skip and continue.
        mock_get.return_value = _mock_response("error", status_code=500)
        crawler = Crawler(delay=0.0)
        pages = crawler.crawl()
        # Failed pages are not added to the result.
        assert pages == {}