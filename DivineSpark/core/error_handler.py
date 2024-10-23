class ErrorHandler:
    @staticmethod
    def handle_api_error(error):
        # Logic for handling different types of API errors
        print(f"API Error: {error}")
        # Here you could include retry mechanisms or more detailed logging

    @staticmethod
    def handle_model_error(error):
        # Logic for handling model-related errors
        print(f"Model Error: {error}")
