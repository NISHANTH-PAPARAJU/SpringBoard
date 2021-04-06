import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    # using the default implementation of start_requests()
    start_urls = [
        'http://quotes.toscrape.com/page/1/',
        'http://quotes.toscrape.com/page/2/'
    ]

    # def start_requests(self):
    #     urls = [
    #         'http://quotes.toscrape.com/page/1/',
    #         'http://quotes.toscrape.com/page/2/'
    #     ]
    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)

    # def parse(self, response):
    #     page = response.url.split("/")[-2]
    #     filename = 'quotes-OOTB-%s.html' % page
    #     with open(filename, "wb") as f:
    #         f.write(response.body)
    #     self.log('Saved file %s' % filename)

    def parse(self, response):
        for quote in response.css('div.quote'):
            text = quote.css('span.text::text').get()
            author = quote.css('small.author::text').get()
            tags = quote.css('div.tags a.tag::text').getall()
            yield dict(text=text, author=author, tags=tags)

        # next_page = response.css("li.next a::attr('href')").get()
        # if next_page is not None:
        #     # next_page = response.urljoin(next_page)
        #     # yield scrapy.Request(url=next_page, callback=self.parse)
        #
        #     # short cut for passing the next page to spider
        #     yield response.follow(next_page, callback=

        # another method2
        # for href in response.css('ul.pager a::attr(href)'):
        #     yield response.follow(href, callback=self.parse)

        # another method3
        # for a in response.css('ul.pager a'):
        #     yield response.follow(a, callback=self.parse)

        # another method4
        anchors = response.css('ul.pager a')
        yield from response.follow_all(anchors, callback=self.parse)
