import yaml
import pathlib
from typing import Any, Optional


class YamlParser:
    """
    Load a YAML config file, expand ${placeholders} that refer to other keys,
    and push everything into os.environ.
    """

    def __init__(self, config_file_path: str):
        self.config_file_path = pathlib.Path(config_file_path).expanduser().resolve()

        # Load as config
        with self.config_file_path.open("r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

    def get_field(self, dotted_key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a nested value from a dict using a dotted path.

        Example: get_field("gcp.storage.paths.data")  ->  "data/papers"
        """
        node = self.config
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node
