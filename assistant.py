# pip install openai

from openai import OpenAI

def get_response(model: str, text_input: str) -> str:
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
    client = OpenAI()
    
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

# Example usage:
if __name__ == "__main__":
    MODEL = "gpt-4o"
    # MODEL = "o3-mini-2025-01-31"
    TEXT_INPUT = """ # Import MediaPipe's Python task wrappers for vision tasks (object detection, etc.)
from mediapipe.tasks import python    # Base API for task configuration
from mediapipe.tasks.python import vision  # Vision-specific modules for object detection
    """
    
    result = get_response(MODEL, TEXT_INPUT)
    print(result)
