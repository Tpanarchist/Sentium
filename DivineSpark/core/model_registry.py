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

    def list_models(self, api_name=None, filter_multimodal=None, filter_audio=None):
        models = self.models.get(api_name, {}) if api_name else self.models
        
        # Apply filters if specified
        if filter_multimodal is not None:
            models = {
                api: {
                    name: details for name, details in api_models.items()
                    if details.get("multimodal") == filter_multimodal
                }
                for api, api_models in models.items()
            }
        
        if filter_audio is not None:
            models = {
                api: {
                    name: details for name, details in api_models.items()
                    if details.get("audio_input") == filter_audio or details.get("audio_output") == filter_audio
                }
                for api, api_models in models.items()
            }

        return models

    def load_from_file(self, filepath):
        try:
            with open(filepath, 'r') as file:
                config_data = json.load(file)
                for api_name, api_models in config_data.items():
                    for model_name, model_details in api_models.items():
                        self.register_model(api_name, model_name, model_details)
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file at {filepath} was not found.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in the configuration file.")

    def save_to_file(self, filepath):
        try:
            with open(filepath, 'w') as file:
                json.dump(self.models, file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to save configuration to {filepath}: {e}")
