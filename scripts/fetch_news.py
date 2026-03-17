"""
CLI script to fetch Cointelegraph news for the backtest period.

Usage:
    python scripts/fetch_news.py
    python scripts/fetch_news.py --start 2025-01-01 --end 2026-01-01
    python scripts/fetch_news.py --start 2025-01-01 --end 2025-02-01 --max 50
"""

import argparse
import logging

from environ.data.cointelegraph import CointelegraphFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Fetch Cointelegraph news articles.")
    parser.add_argument("--start", default="2025-01-01", help="Start date (inclusive, ISO format)")
    parser.add_argument("--end", default="2026-01-01", help="End date (exclusive, ISO format)")
    parser.add_argument("--output-dir", default="data/news", help="Directory to save results")
    parser.add_argument("--max", type=int, default=None, dest="max_articles",
                        help="Max articles to fetch (for testing)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between requests (default: 1.5)")
    args = parser.parse_args()

    fetcher = CointelegraphFetcher(output_dir=args.output_dir, request_delay=args.delay)
    articles = fetcher.fetch_range(
        start=args.start,
        end=args.end,
        max_articles=args.max_articles,
    )
    print(f"\nFetched {len(articles)} articles → {args.output_dir}")


if __name__ == "__main__":
    main()
