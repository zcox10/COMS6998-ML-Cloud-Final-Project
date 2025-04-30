from google.cloud import secretmanager
from src.utils.yaml_parser import YamlParser


class SecretsUtils:

    def __init__(self):
        self._config = YamlParser("./config.yaml")
        self._gcp_project_id = self._config.get_field("gcp.project_id")

    def get_secret(self, secret_id: str):
        """
        Retrieve a secret from Google Cloud Secrets Manager.

        Args:
            project_id (str): The ID of the Google Cloud project.
            secret_id (str): The ID of the secret.
            version_id (str): The ID of the secret version.
        Returns:
            str: The plain text value of the secret.
        """
        name = f"projects/{self._gcp_project_id}/secrets/{secret_id}/versions/latest"
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
