import scrapy

class BookSpiderSpider(scrapy.Spider):
    name = "book_spider"
    allowed_domains = ["fourminutebooks.com"]
    start_urls = ["https://fourminutebooks.com/book-summaries/"]

    def parse(self, response):
        
        links = response.css("a::attr(href)").getall()
        filtered_links = set([link for link in links if link.startswith("https://fourminutebooks.com/") and link.endswith("-summary/")])
        
        self.logger.info(f"Total filtered links: {len(filtered_links)}")  
        
        for link in filtered_links:
            yield response.follow(link, callback=self.parse_book)

    def parse_book(self, response):
        title = response.css("h1::text").get().strip().removesuffix(" Summary").strip()

        
        content_elements = response.xpath('//div[@class="su-note"]/following-sibling::*')
        end_index = next((i for i, el in enumerate(content_elements) if el.xpath('./@align').get() == 'center'), len(content_elements))
        content = '\n'.join(
            [''.join(el.xpath('.//text()').getall()).strip()
             for el in content_elements[:end_index]
             if not el.xpath('self::div[@class="formkit-background"]') and el.xpath('.//text()')]
        )

        yield {
            "title": title,
            "url": response.url,
            "content": content
        }
