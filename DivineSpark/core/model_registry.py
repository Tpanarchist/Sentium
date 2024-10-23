import json

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, api_name, model_name, model_details):
        if api_name not in self.models:
            self.models[api_name] = {}
        self.models[api_name][model_name] = model_details

    def get_model(self, api_name, model_name):
        if api_name in self.models and model_name in self.models[api_name]:
            return self.models[api_name][model_name]
        else:
            raise ValueError(f"Model {model_name} for API {api_name} not found.")

    def list_models(self, api_name=None):
        if api_name:
            return self.models.get(api_name, {})
        else:
            return self.models
