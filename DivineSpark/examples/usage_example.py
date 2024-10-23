from ..core.divine_spark import DivineSpark
from ..config import openai

# Example of using the Divine Spark interface
if __name__ == "__main__":
    api_key = "your_openai_api_key"
    api_type = "openai"
    
    spark = DivineSpark(api_key, api_type)
    
    # Register a model from configuration
    for model_name, model_details in openai.items():
        spark.register_model(model_name, model_details)
    
    # Use a registered model
    model = spark.use_model("gpt-4o")
    print(f"Model loaded: {model}")
