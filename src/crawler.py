"""
Web Crawler Module
==================

Implements a polite, breadth-first web crawler for the COMP3011 search
engine coursework. Enforces a configurable delay (default 6 seconds)
between successive HTTP requests and gracefully handles network
failures.

The crawler restricts itself to the domain of the seed URL to avoid
runaway crawls into external sites.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Module-level logger so users of the library can configure handlers
# externally without us forcing a configuration on import.
logger = logging.getLogger(__name__)


class Crawler:
    """A polite breadth-first web crawler.

    The crawler starts at ``base_url`` and follows internal links
    (links pointing to the same network location) until no new pages
    remain. A configurable politeness delay ensures we do not flood
    the target server with requests.

    Attributes:
        base_url: The seed URL the crawl starts from.
        delay: Minimum seconds between successive HTTP requests.
        timeout: Per-request socket timeout (seconds).
        user_agent: Identifier sent in the ``User-Agent`` header.
        visited: URLs that have been successfully fetched.
        pages: Mapping of URL -> extracted plain text for each page.
    """

    def __init__(
        self,
        base_url: str = "https://quotes.toscrape.com/",
        delay: float = 6.0,
        timeout: float = 10.0,
        user_agent: str = "COMP3011-SearchEngine/1.0 (Educational Crawler)",
    ) -> None:
        """Initialise the crawler.

        Args:
            base_url: Seed URL to start crawling from.
            delay: Minimum seconds between successive requests
                (must be >= 6 to satisfy the assignment brief).
            timeout: HTTP socket timeout in seconds.
            user_agent: Value sent as the ``User-Agent`` request header.
        """
        if delay < 6.0:
            # Defensive check -- the brief mandates a 6-second window.
            logger.warning(
                "Delay %.1fs is below the required 6s politeness window.",
                delay,
            )

        self.base_url: str = base_url
        self.delay: float = delay
        self.timeout: float = timeout
        self.user_agent: str = user_agent

        # Domain we are allowed to crawl. Setting this once means we
        # do not have to re-parse ``base_url`` for every link check.
        self._allowed_netloc: str = urlparse(base_url).netloc

        # State accumulated during a crawl.
        self.visited: set[str] = set()
        self.pages: dict[str, str] = {}

        # Used to enforce the politeness window. ``0.0`` means no
        # request has been made yet, so the first request fires
        # immediately.
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> dict[str, str]:
        """Run the full crawl and return all discovered pages.

        Returns:
            A dict mapping each successfully-fetched URL to its
            extracted plain text.
        """
        logger.info("Starting crawl at %s", self.base_url)

        # Use a list as a FIFO queue. ``collections.deque`` would be
        # marginally faster, but pages-per-crawl is small here so
        # readability wins.
        queue: list[str] = [self._normalise(self.base_url)]

        while queue:
            url = queue.pop(0)

            # Skip already-seen URLs. We add to ``visited`` only
            # *after* a successful fetch, but we also short-circuit
            # here to avoid enqueuing duplicates.
            if url in self.visited:
                continue

            html = self.fetch(url)
            if html is None:
                # Mark as visited so we do not retry indefinitely.
                self.visited.add(url)
                continue

            self.visited.add(url)
            self.pages[url] = self.extract_text(html)

            # Enqueue any links we have not seen yet.
            for link in self.extract_links(html, url):
                if link not in self.visited and link not in queue:
                    queue.append(link)

        logger.info("Crawl complete: %d pages fetched.", len(self.pages))
        return self.pages

    def fetch(self, url: str) -> Optional[str]:
        """Fetch a single URL, respecting the politeness window.

        Args:
            url: Absolute URL to fetch.

        Returns:
            The response body as text, or ``None`` if the request
            failed (network error, non-200 status, etc.).
        """
        self._respect_politeness()

        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            self._last_request_time = time.time()
        except requests.exceptions.RequestException as exc:
            # Catches Timeout, ConnectionError, TooManyRedirects, etc.
            logger.warning("Request failed for %s: %s", url, exc)
            self._last_request_time = time.time()
            return None

        if response.status_code != 200:
            logger.warning(
                "Non-200 status %d for %s", response.status_code, url
            )
            return None

        return response.text

    def extract_links(self, html: str, current_url: str) -> set[str]:
        """Extract all internal links from a page.

        Args:
            html: Raw HTML of the page.
            current_url: The URL the HTML was fetched from. Used to
                resolve relative links.

        Returns:
            A set of absolute URLs that point to the same domain.
        """
        soup = BeautifulSoup(html, "html.parser")
        links: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]

            # Resolve relative URLs (e.g. "/page/2/") against the
            # current page.
            absolute = urljoin(current_url, href)
            normalised = self._normalise(absolute)

            if self._is_internal(normalised):
                links.add(normalised)

        return links

    def extract_text(self, html: str) -> str:
        """Extract human-readable text from a page.

        Strips ``<script>`` and ``<style>`` content because their
        contents would otherwise pollute the inverted index with
        JavaScript identifiers and CSS class names.

        Args:
            html: Raw HTML of the page.

        Returns:
            Whitespace-collapsed plain text from the page body.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements outright.
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # ``get_text`` joins every text node with the given separator;
        # collapsing whitespace prevents accidental token merges.
        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _respect_politeness(self) -> None:
        """Sleep just long enough to honour the politeness window."""
        if self._last_request_time == 0.0:
            return  # First request -- no need to wait.

        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            wait = self.delay - elapsed
            logger.debug("Sleeping %.2fs to honour politeness window.", wait)
            time.sleep(wait)

    def _is_internal(self, url: str) -> bool:
        """Return ``True`` iff ``url`` belongs to the crawl domain."""
        parsed = urlparse(url)

        # Reject anything that is not http(s) -- skips ``mailto:``,
        # ``javascript:``, ``tel:`` and similar schemes.
        if parsed.scheme not in {"http", "https"}:
            return False

        return parsed.netloc == self._allowed_netloc

    @staticmethod
    def _normalise(url: str) -> str:
        """Drop the URL fragment so ``/page#a`` and ``/page#b`` match."""
        parsed = urlparse(url)
        # Reconstruct without the fragment component.
        return parsed._replace(fragment="").geturl()