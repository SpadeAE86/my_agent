from openai import AsyncOpenAI
import asyncio
from prompts.system_prompt.neuro_role import neuro_role

async def process_stream_response(response):
    """
    Processes a streaming response from the OpenAI API.

    Args:
        response: The streaming response object from the OpenAI API.
    Yields:
        delta: The content of each message chunk as it is received.
    """
    async for chunk in response:
        if not hasattr(chunk, "choices") or not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        yield delta

async def chat(client: AsyncOpenAI, messages,
               model="gemini-3-pro", temperature=0.7,
               max_tokens=150, stream=False):
    """
    Sends a chat message to the OpenAI API and returns the response.

    Args:
        client (AsyncOpenAI): An instance of the AsyncOpenAI client.
        messages (list): A list of message dictionaries.
        model (str): The model to use for the chat.
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        stream (bool): Whether to stream the response or not.
    Returns:
        str: The response from the OpenAI API.

    Parameters
    ----------

    """


    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream,
        max_tokens=max_tokens
    )
    message_content = ""
    if stream:
        async for delta in process_stream_response(response):
            print(delta.model_dump())
            text = delta.content or delta.reasoning_content or ""
            message_content += text
    else:
        message_content = response.choices[0].message.content
    return message_content

if __name__ == "__main__":
    client = AsyncOpenAI(
        base_url="https://z.apiyihe.org/v1",
        api_key="sk-TMd7SbPPbVw1JMx0GYKflkWkv8Mzi1tb0B64Y9HqBQ53TaqW",
    )
    messages = [
        {"role": "system", "content": f"{neuro_role}"},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    result = asyncio.run(chat(client, messages))
    print(f"receive: {result}")