# DivineSpark/core/model_registry.py

import json

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, api_name, model_name, model_details):
        if api_name not in self.models:
            self.models[api_name] = {}
        self.models[api_name][model_name] = model_details
        print(f"Registered model '{model_name}' under API '{api_name}'.")

    def get_model(self, api_name, model_name):
        if api_name in self.models and model_name in self.models[api_name]:
            print(f"Retrieving model: '{model_name}' for API: '{api_name}'.")
            return self.models[api_name][model_name]
        else:
            raise ValueError(f"Model '{model_name}' for API '{api_name}' not found.")

    def list_models(self, api_name=None, filter_multimodal=None, filter_audio=None):
        models = self.models.get(api_name, {}) if api_name else self.models
        print(f"Listing models. API: '{api_name}', Filter Multimodal: '{filter_multimodal}', Filter Audio: '{filter_audio}'.")

        # Apply filters if specified
        if filter_multimodal is not None:
            models = {
                api: {
                    name: details for name, details in api_models.items()
                    if isinstance(details, dict) and details.get("capabilities", {}).get("input") and any(input_type in ["text", "image"] for input_type in details["capabilities"].get("input", [])) and details.get("capabilities", {}).get("output") and "text" in details["capabilities"]["output"] and details.get("capabilities", {}).get("output") != "audio"
                }
                for api, api_models in models.items()
            }

        if filter_audio is not None:
            models = {
                api: {
                    name: details for name, details in api_models.items()
                    if isinstance(details, dict) and details.get("capabilities", {}).get("input") and "audio" in details["capabilities"]["input"]
                }
                for api, api_models in models.items()
            }

        print(f"Filtered models: {models}")
        return models

    def load_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                config_data = json.load(file)
                for api_name, categories in config_data.items():
                    for category_name, models in categories.items():
                        for model_name, model_details in models.items():
                            self.register_model(api_name, model_name, model_details)
            print(f"All models loaded successfully from '{filepath}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file at '{filepath}' was not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in the configuration file: {e}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Unicode decoding error in the configuration file: {e}")

    def save_to_file(self, filepath):
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(self.models, file, indent=4)
            print(f"Model configurations saved successfully to '{filepath}'.")
        except IOError as e:
            raise IOError(f"Failed to save configuration to '{filepath}': {e}")
