# src/data/scrapers/review_scraper.py

import scrapy
from scrapy.crawler import CrawlerProcess
from typing import List, Dict, Any
import logging
from urllib.parse import urljoin


class TrustpilotScraper(scrapy.Spider):
    """
    Scraper Trustpilot respectueux du robots.txt.
    Limite volontaire : max 10 pages par entreprise.
    Extrait : texte, note, titre, date, URL source.
    """

    name = 'trustpilot_scraper'

    custom_settings = {
        'DOWNLOAD_DELAY': 3,
        'CONCURRENT_REQUESTS': 1,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'Mozilla/5.0 (Educational Purpose Bot)',
        'LOG_LEVEL': 'INFO',
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    def __init__(self, company_urls: List[str]):
        super().__init__()
        self.start_urls = company_urls
        self.page_count = 0
        self.max_pages = 10

    def parse(self, response):
        self.page_count += 1
        self.logger.info(f"[TrustpilotScraper]  Page {self.page_count} — {response.url}")

        reviews = response.css('div.review-card')
        if not reviews:
            self.logger.warning(f"[TrustpilotScraper]  Aucun avis trouvé sur cette page.")

        for review in reviews:
            item = self.parse_review(review, response.url)
            if item:
                yield item

        next_page = response.css('a.next-page::attr(href)').get()
        if next_page and self.page_count < self.max_pages:
            next_url = urljoin(response.url, next_page)
            self.logger.info(f"[TrustpilotScraper]  Prochaine page : {next_url}")
            yield response.follow(next_url, self.parse)
        else:
            self.logger.info(f"[TrustpilotScraper]  Fin du scraping pour {response.url}")

    def parse_review(self, review, base_url: str) -> Dict[str, Any]:
        """
        Parse propre d'un bloc d'avis.
        """
        text = review.css('p.review-text::text').get(default="").strip()
        rating = review.css('div.star-rating::attr(data-rating)').get(default="").strip()
        date = review.css('time::attr(datetime)').get(default="").strip()
        title = review.css('h3.review-title::text').get(default="").strip()

        if not text:
            self.logger.debug("[TrustpilotScraper]  Avis ignoré car vide.")
            return None

        return {
            'text': text,
            'rating': rating,
            'date': date,
            'title': title,
            'source_url': base_url
        }


def scrape_reviews(company_urls: List[str], output_file: str):
    """
    Lance le scraping pour une liste d'URL Trustpilot et sauvegarde en CSV.
    Exemple :
        scrape_reviews(['https://fr.trustpilot.com/review/example.com'], 'output/trustpilot_reviews.csv')
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    process = CrawlerProcess({
        'FEEDS': {
            output_file: {
                'format': 'csv',
                'encoding': 'utf-8',
                'overwrite': True
            }
        },
        'LOG_LEVEL': 'INFO'
    })

    process.crawl(TrustpilotScraper, company_urls=company_urls)
    process.start()
    print(f"[INFO] Scraping terminé → Données sauvegardées : {output_file}")


if __name__ == "__main__":
    # Exemple d'appel direct
    example_urls = ["https://fr.trustpilot.com/review/example.com"]
    scrape_reviews(example_urls, "output/trustpilot_reviews.csv")
