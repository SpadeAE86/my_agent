from openai import AsyncOpenAI
import asyncio

SYSTEM_PROMPT = """你是一个人工智能助手，协助用户解答问题和提供信息。请根据用户的提问，尽可能准确和详细地回答。如果你不确定答案，可以说你不知道，但不要编造信息。"""

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
               max_tokens=150, stream=False, tools=None):
    """
    Sends a chat message to the OpenAI API and returns the response message object.
    
    Args:
        client (AsyncOpenAI): An instance of the AsyncOpenAI client.
        messages (list): A list of message dictionaries.
        model (str): The model to use for the chat.
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        stream (bool): Whether to stream the response or not.
        tools (list[dict]): A list of tool JSON schemas for function calling.
    Returns:
        The raw response message object (with .content and .tool_calls).
    """
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "max_tokens": max_tokens
    }
    
    if tools:
        kwargs["tools"] = tools
        
    response = await client.chat.completions.create(**kwargs)
    
    if stream:
        # TODO: Handle streams properly with tool calls if needed
        message_content = ""
        async for delta in process_stream_response(response):
            print(delta.model_dump())
            text = delta.content or delta.reasoning_content or ""
            message_content += text
        class DummyMsg:
            content = message_content
            tool_calls = None
        return DummyMsg()
    else:
        return response.choices[0].message

if __name__ == "__main__":
    client = AsyncOpenAI(
        base_url="https://z.apiyihe.org/v1",
        api_key="sk-TMd7SbPPbVw1JMx0GYKflkWkv8Mzi1tb0B64Y9HqBQ53TaqW",
    )
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    result = asyncio.run(chat(client, messages,stream=True))
    print(f"receive: {result}")