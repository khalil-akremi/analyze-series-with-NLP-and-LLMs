import scrapy
from bs4 import BeautifulSoup


class blogSpider(scrapy.Spider):
    name = "companyspider"
    start_urls = [
        'https://breakingbad.fandom.com/wiki/Category:Companies',
    ]

    def parse(self, response):
        for href in response.css('.category-page__members a::attr(href)').extract():
            yield scrapy.Request(
                url=response.urljoin(href),
                callback=self.parse_company
            )

        next_page = response.css('a.category-page__pagination-next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_company(self, response):
        company_name = response.css('span.mw-page-title-main::text').get().strip()

        div_selector = response.css("div.mw-content-ltr.mw-parser-output")
        if not div_selector:
            return

        div_html = div_selector.get()
        soup = BeautifulSoup(div_html, "html.parser")

        company_type = ""
        aside = soup.find('aside')
        if aside:
            for cell in aside.find_all('div', class_='pi-data'):
                header = cell.find('h3')
                if header and header.text.strip() == 'Type':
                    company_type = cell.find('div').text.strip()

            aside.decompose()  # remove infobox

        content_div = soup.find("div", class_="mw-parser-output")
        paragraphs = content_div.find_all("p")

        company_description = " ".join(
            p.get_text(strip=True) for p in paragraphs[:2]
        )

        return {
            "company_name": company_name,
            "company_type": company_type,
            "company_description": company_description
        }

