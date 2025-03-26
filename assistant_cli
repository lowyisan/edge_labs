import argparse
import os
from openai import OpenAI

def get_response(model: str, text_input: str, api_key: str) -> str:
    """
    Creates a response using the OpenAI client with the specified model and text input.

    Parameters:
        model (str): The model to use (e.g., "gpt-4o").
        text_input (str): The text input from the user.
        api_key (str): Your OpenAI API key.

    Returns:
        str: The text content of the response.
    """
    # Initialize the OpenAI client with your API key
    client = OpenAI(api_key=api_key)
    
    # Create the response using the client
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text_input
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )
    
    if model == "gpt-4o":
        return response.output[0].content[0].text
    if model == "o3-mini-2025-01-31":
        return response.output[1].content[0].text
    return "Model not recognized."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a response using the OpenAI API from the terminal."
    )
    parser.add_argument(
        "text",
        help="The text input to send to the model."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="The model to use (default: 'gpt-4o')."
    )
    
    args = parser.parse_args()
    
    # Retrieve the API key from the environment variable
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    result = get_response(args.model, args.text, API_KEY)
    print(result)
