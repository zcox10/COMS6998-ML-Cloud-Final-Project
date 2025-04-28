import yaml
import pathlib
from typing import Any, List


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

    def get_field(self, dotted_key: str) -> Any:
        """
        Retrieve a nested value from a dict using a dotted path.
        Example: get_field("gcp.storage.paths.data")  ->  "data/papers"
        """
        node = self.config
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise KeyError(f"Key path '{dotted_key}' not found in config (failed at '{part}').")
        return node

    def export_to_tfvars(self, whitelist: List[str], output_path: str) -> None:
        """
        Export selected keys to a .tfvars file in proper HCL format.
        """
        output_path = pathlib.Path(output_path).expanduser().resolve()
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Auto-generated from {self.config_file_path.name}\n")

            for dotted_key in whitelist:
                value = self.get_field(dotted_key)
                if value is None:
                    print(f"Warning: {dotted_key} not found.")
                    continue

                tf_var_name = dotted_key.replace(".", "_")
                hcl_value = self._to_hcl(value)
                fh.write(f"{tf_var_name} = {hcl_value}\n")

        print(f"\n========== Terraform tfvars generated at {output_path.name} ==========\n")

    def _to_hcl(self, value: Any) -> str:
        """
        Convert Python value to Terraform HCL syntax.
        """
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return f"[{', '.join(self._to_hcl(item) for item in value)}]"
        elif isinstance(value, dict):
            return f"{{{', '.join(f'{k} = {self._to_hcl(v)}' for k, v in value.items())}}}"
        else:
            raise ValueError(f"Unsupported type for HCL serialization: {type(value)}")
