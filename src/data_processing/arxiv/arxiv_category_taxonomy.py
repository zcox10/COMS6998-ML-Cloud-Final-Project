import requests
import bs4
import re


class ArxivCategoryTaxonomy:
    def __init__(self, taxonomy_url: str = "https://arxiv.org/category_taxonomy"):
        """
        Scrapes `taxonomy_url` for categories of research papers hosted on arXiv to enrich metadata.
        """
        self.taxonomy_url = taxonomy_url

    def retrieve_taxonomy(self) -> dict:
        """
        Returns flat mapping like:
            {"cs.AI": "Computer Science - Artificial Intelligence", ...}
        """
        html = requests.get(self.taxonomy_url, timeout=10).text
        soup = bs4.BeautifulSoup(html, "html.parser")

        category_map = {}
        current_primary_label = None

        for tag in soup.find_all(["h2", "h4"]):
            # PRIMARY (e.g. <h2 id="cs">Computer Science</h2>)
            if tag.name == "h2":
                current_primary_label = tag.text.strip()
                continue

            # SECONDARY (e.g. <h4>cs.AI (Artificial Intelligence)</h4>)
            if tag.name == "h4":
                match = re.match(r"^([a-z]{2})\.([A-Z]{2})", tag.text)
                if match and current_primary_label:
                    prim_code, sec_code = match.groups()
                    full_code = f"{prim_code}.{sec_code}"
                    sec_label = tag.text.replace(full_code, "").strip(" ()\n")
                    category_map[full_code] = f"{current_primary_label} - {sec_label}"

        return category_map
