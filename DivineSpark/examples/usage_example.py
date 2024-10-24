import os
from DivineSpark.core.divine_spark import DivineSpark
from DivineSpark.core.utils import Utils

# Load the configuration for OpenAI models
CONFIG_FILE_PATH = 'DivineSpark/config/models_config.json'

def main():
    # Ensure API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API Key. Please set the OPENAI_API_KEY environment variable.")
    
    api_type = "openai"
    
    # Initialize DivineSpark
    spark = DivineSpark(api_key, api_type)
    
    # Load OpenAI model configuration from file
    if Utils.file_exists(CONFIG_FILE_PATH):
        try:
            openai_models = Utils.load_from_json(CONFIG_FILE_PATH).get("openai", {})
            # Register each model from the configuration
            for model_name, model_details in openai_models.items():
                spark.register_model(model_name, model_details)
                print(f"Model registered: {model_name}")
        except Exception as e:
            print(f"Error loading model configuration: {e}")
            return
    else:
        print(f"Configuration file not found at {CONFIG_FILE_PATH}. Please provide a valid configuration.")
        return
    
    # Example usage of a registered model
    try:
        model_name = "gpt-4o"
        model = spark.use_model(model_name)
        print(f"Model loaded: {model}")
        
        # Example text generation
        prompt = "What is the future of AI?"
        result = spark.generate_text_completion(prompt, model_name=model_name, max_tokens=50)
        print(f"Generated response: {result}")
        
    except Exception as e:
        print(f"Error using the model: {e}")

if __name__ == "__main__":
    main()
