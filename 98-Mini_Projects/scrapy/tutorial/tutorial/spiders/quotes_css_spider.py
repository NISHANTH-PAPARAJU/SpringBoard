import scrapy


class quotes_css_spider(scrapy.Spider):
    name = 'toscrape-css'

    start_urls = [
        'http://quotes.toscrape.com/'
    ]

    def parse(self, response):
        for quote in response.css('div.quote'):
            text = quote.css('span.text::text').get()
            author = quote.css('small.author::text').get()
            tags = quote.css('div.tags a.tag::text').getall()
            yield dict({'text': text, 'author': author, 'tags': tags})

        next_page = response.css('li.next a::attr("href")').get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
