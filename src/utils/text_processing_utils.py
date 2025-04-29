import re


class TextProcessingUtils:
    def __init__(self):
        pass

    def clean_and_wrap_markdown(self, doc_md: str) -> str:
        # Define undesired sections
        undesired_sections = [
            "ACKNOWLEDGEMENTS",
            "ACKNOWLEDGEMENT",
            "ACKNOWLEDGMENT",
            "REFERENCE",
            "REFERENCES",
            "APPENDIX",
            "APPENDICES",
            "BIBLIOGRAPHY",
            "BIBLIOGRAPHIES",
        ]

        # Regex components:
        # - ^#{1,6}\s+: match any markdown heading level
        # - (?:[\w\d.]+\s+)? optional enumerator (e.g., "1. ", "I. ", "A. ")
        # - (ACKNOWLEDGEMENTS|...): one of the undesired sections
        pattern = re.compile(
            r"^#{1,6}\s+(?:[\w\d.]+\s+)?(" + "|".join(undesired_sections) + r")\b.*",
            re.IGNORECASE | re.MULTILINE,
        )

        # Find the start of the first undesired section
        match = pattern.search(doc_md)
        if match:
            doc_md = doc_md[: match.start()].rstrip()

        # Wrap with delimiters
        return f"<|startofpaper|>\n{doc_md}\n<|endofpaper|>"
