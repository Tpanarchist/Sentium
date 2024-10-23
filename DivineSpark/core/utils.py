import json

class Utils:
    @staticmethod
    def save_to_json(data, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    @staticmethod
    def load_from_json(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
