# 🔍 COMP3011 Search Engine
 
A Python-based web search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index of word occurrences, and provides ranked search results via a command-line interface.
 
## 📋 Project Overview
 
This search engine implements the three core stages of information retrieval:
 
1. **Web Crawling** — Politely crawls all pages of the target website (6-second delay between requests)
2. **Indexing** — Builds an inverted index storing word frequency and positional information for each page
3. **Searching** — Supports single-word and multi-word queries with TF-IDF ranking for relevance
A typical full crawl of `quotes.toscrape.com` collects **~210 pages** and indexes **~4,650 unique words** in roughly 22 minutes (the wall-clock time is dominated by the mandatory 6-second politeness window between requests).
 
### Key Features
 
- ✅ Polite web crawler with configurable delay (default: 6 seconds)
- ✅ Robust error handling for network failures
- ✅ Inverted index with frequency + position tracking
- ✅ Case-insensitive search
- ✅ Multi-word queries with intersection logic (AND)
- ✅ TF-IDF ranking for result relevance
- ✅ Persistent index storage (JSON)
- ✅ Comprehensive test suite with mocked network calls
- ✅ Type hints and docstrings throughout
## 🏗️ Architecture
 
```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                         │
│              [build] [load] [print] [find]                   │
└────────────┬───────────────┬────────────┬───────────────────┘
             │               │            │
             ▼               ▼            ▼
      ┌──────────┐    ┌──────────┐  ┌──────────┐
      │ crawler  │───▶│ indexer  │  │  search  │
      │   .py    │    │   .py    │  │   .py    │
      └──────────┘    └──────────┘  └──────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │ data/       │
                       │ index.json  │
                       └─────────────┘
```
 
### Module Responsibilities
 
| Module | Class | Responsibility |
|--------|-------|----------------|
| `crawler.py` | `Crawler` | Fetch HTML, extract links, enforce politeness |
| `indexer.py` | `Indexer` | Tokenize text, build/save/load inverted index |
| `search.py` | `SearchEngine` | Query the index, rank results with TF-IDF |
| `main.py` | — | CLI loop and command dispatch |
 
### Inverted Index Structure
 
```python
{
    "good": {
        "http://quotes.toscrape.com/page/1/": {
            "frequency": 3,
            "positions": [12, 47, 89]
        },
        "http://quotes.toscrape.com/page/2/": {
            "frequency": 1,
            "positions": [22]
        }
    },
    ...
}
```
 
**Why this structure?**
- Hash-map lookups give **O(1)** word retrieval
- Frequency enables TF-IDF ranking
- Positions enable phrase and proximity searches
- JSON-serializable for simple persistence
## 🚀 Installation
 
### Prerequisites
- Python 3.9 or higher
- pip
### Setup
 
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd comp3011-search-engine
 
# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
## 💻 Usage
 
Launch the search tool from the project root:
 
```bash
python -m src.main
```
 
You'll see the prompt:
```
> 
```
 
### Available Commands
 
#### `build`
Crawls the target website and builds the inverted index, saving it to `data/index.json`.
 
```
> build
[INFO] Starting crawl of https://quotes.toscrape.com/
[INFO] Crawled 10 pages
[INFO] Indexed 1234 unique words
[INFO] Index saved to data/index.json
```
 
⚠️ Note: Build takes ~60+ seconds due to the 6-second politeness delay between requests.
 
#### `load`
Loads a previously built index from disk.
 
```
> load
[INFO] Index loaded from data/index.json
```
 
#### `print <word>`
Displays the inverted index entry for a single word.
 
```
> print nonsense
nonsense:
  http://quotes.toscrape.com/page/3/: frequency=1, positions=[42]
```
 
#### `find <word(s)>`
Searches the index for pages containing all given words. Results are ranked by TF-IDF.
 
```
> find indifference
Found 1 page(s):
  1. http://quotes.toscrape.com/page/5/  (score: 0.0234)
 
> find good friends
Found 2 page(s):
  1. http://quotes.toscrape.com/page/1/  (score: 0.0456)
  2. http://quotes.toscrape.com/page/4/  (score: 0.0189)
```
 
#### `exit`
Exits the program.
 
## 🧪 Testing
 
Run the full test suite with coverage:
 
```bash
pytest
```
 
Generate an HTML coverage report:
 
```bash
pytest --cov-report=html
# Open htmlcov/index.html in your browser
```
 
### Test Strategy
 
- **Unit tests** for each class with isolated dependencies
- **Mocking** of HTTP requests (no live network calls during testing)
- **Edge case coverage**: empty queries, missing words, network errors, malformed HTML, corrupt index files
- **Integration tests** for end-to-end CLI flow (`build` → `find` → `print`)
- **Mocked timing**: `time.sleep` is patched so politeness logic is verified without slowing tests
### Current Test Statistics
 
| Metric | Value |
|--------|-------|
| Total tests | **100** |
| Coverage | **96%** |
| Suite runtime | ~3 seconds (no network I/O) |
| Test files | 4 (`test_crawler.py`, `test_indexer.py`, `test_search.py`, `test_main.py`) |
 
## 📂 Project Structure
 
```
comp3011-search-engine/
├── src/
│   ├── __init__.py
│   ├── crawler.py        # Crawler class
│   ├── indexer.py        # Indexer class
│   ├── search.py         # SearchEngine class
│   └── main.py           # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_crawler.py
│   ├── test_indexer.py
│   └── test_search.py
├── data/
│   └── index.json        # Generated by `build`
├── requirements.txt
├── pytest.ini
├── README.md
└── .gitignore
```
 
## 🔬 Design Decisions
 
### Why BFS over DFS for crawling?
A queue-based BFS avoids recursion depth limits and gives more predictable memory usage on websites with deep link structures.
 
### Why JSON over a database?
The brief specifies a single index file. JSON is human-readable (helpful for debugging and the video walkthrough), portable, and natively supported by Python without extra dependencies.
 
### Why store positions if we don't use them yet?
Positions enable future enhancements like phrase search ("good friends" as a phrase) and proximity ranking, with negligible storage overhead.
 
### Why TF-IDF for ranking?
TF-IDF balances local frequency (TF) against global commonness (IDF), so rare-but-relevant words boost a page's score more than common words. It's a classic, well-understood baseline for information retrieval.
 
## 📚 Dependencies
 
| Package | Purpose |
|---------|---------|
| `requests` | HTTP client for fetching pages |
| `beautifulsoup4` | HTML parsing |
| `pytest` | Test framework |
| `pytest-cov` | Coverage reporting |
 
## 📄 License
 
This project is submitted as coursework for COMP3011 Web Services and Web Data at the University of Leeds.
 
## 🙏 Acknowledgements
 
- Target website: [quotes.toscrape.com](https://quotes.toscrape.com/)
- [Requests](https://docs.python-requests.org/) and [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) documentation
 