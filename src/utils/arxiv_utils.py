class ArxivUtils:
    def extract_formatted_entry_id_from_url(self, entry_id: str):
        """
        Parses entry id (e.g., `https://arxiv.org/pdf/2504.17782v1` or `http://arxiv.org/abs/2504.17782v1`)
        to output `2504-17782v1`
        """
        return entry_id.split("/")[-1].replace(".", "-")
