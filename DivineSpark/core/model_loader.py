# DivineSpark/core/model_loader.py

class ModelLoader:
    def __init__(self, api_client, model_registry):
        """
        Initializes the ModelLoader with the given API client and model registry.

        Args:
            api_client (BaseAPIClient): The API client to interact with.
            model_registry (ModelRegistry): The registry containing model configurations.
        """
        self.api_client = api_client
        self.model_registry = model_registry

    def load_model(self, api_name, model_name):
        """
        Loads a model from the registry.

        Args:
            api_name (str): The name of the API (e.g., "openai").
            model_name (str): The name of the model to load.

        Returns:
            dict: The model details.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        return self.model_registry.get_model(api_name, model_name)

    def is_multimodal(self, api_name, model_name):
        """
        Checks if a model is multimodal based on its capabilities.

        Args:
            api_name (str): The name of the API.
            model_name (str): The name of the model.

        Returns:
            bool: True if the model is multimodal, False otherwise.
        """
        model = self.load_model(api_name, model_name)
        capabilities = model.get("capabilities", {})
        inputs = capabilities.get("input", [])
        return "image" in inputs or "audio" in inputs

    def get_endpoint(self, api_name, model_name):
        """
        Retrieves the API endpoint associated with a model.

        Args:
            api_name (str): The name of the API.
            model_name (str): The name of the model.

        Returns:
            str: The API endpoint.

        Raises:
            ValueError: If the endpoint is not defined for the model.
        """
        model = self.load_model(api_name, model_name)
        endpoint = model.get("endpoint")
        if not endpoint:
            raise ValueError(f"Endpoint not defined for model '{model_name}'.")
        return endpoint
