class ModelLoader:
    def __init__(self, api_client, model_registry):
        self.api_client = api_client
        self.model_registry = model_registry

    def load_model(self, api_name, model_name):
        model_details = self.model_registry.get_model(api_name, model_name)
        return self.api_client.load_model(model_name, **model_details)
