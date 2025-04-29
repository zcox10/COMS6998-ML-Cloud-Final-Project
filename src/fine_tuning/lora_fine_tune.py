from src.utils.yaml_parser import YamlParser
from src.utils.gcs_file_handler import GcsFileHandler
from src.utils.local_file_handler import LocalFileHandler


class LoraFineTune:
    def __init__(self):
        # Configure file handlers and utils
        self._config = YamlParser("./config.yaml")
        self._gcs_bucket_name = self._config.get_field("gcp.gcs.buckets")[0]["name"]
        self._gcs_data_directory = self._config.get_field("gcp.gcs.buckets")[0]["paths"]["data"]
        self._gcs_file_handler = GcsFileHandler(bucket_name=self._gcs_bucket_name)
        self._local_file_handler = LocalFileHandler()
