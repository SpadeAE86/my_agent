# core/memory/summarizer.py — LLM-based summarization helper
from openai import AsyncOpenAI
from utils.llm_utils import chat
import os

COMPACT_INSTRUCTIONS = (
    "You are summarizing the conversation history to save context tokens. "
    "Please generate a concise but comprehensive summary. "
    "MUST PRESERVE:\n"
    "- Active tasks and their current status (in-progress, blocked, pending)\n"
    "- The last thing the user requested and what was being done about it\n"
    "- Decisions made and their rationale\n"
    "- TODOs, open questions, and constraints\n"
    "- Any commitments or follow-ups promised\n"
    "PRIORITIZE recent context over older history. Keep the summary under 800 tokens."
)

SESSION_SUMMARY_INSTRUCTIONS = (
    "You are summarizing a completed chat session to write a daily log entry.\n"
    "Based on the provided session diary (if any) and raw chat transcript, extract:\n"
    "- Important facts about the user or their preferences\n"
    "- Key project decisions and their rationale\n"
    "- Completed tasks and notable achievements\n"
    "- Open issues, bugs, and pending tasks\n"
    "Output a concise list of bullet points with date-agnostic references."
)

DREAM_INSTRUCTIONS = (
    "You are performing a memory consolidation dream. "
    "Review the existing MEMORY.md index content (if any) and the recent daily log entries. "
    "Your goal is to synthesize what you've learned recently into durable, well-organized memories so that future sessions can orient quickly. "
    "Focus on:\n"
    "- Merging new information into existing topic sections or creating new sections\n"
    "- Updating or removing outdated/wrong memories\n"
    "- Converting relative dates (e.g., 'yesterday') to absolute dates\n"
    "Structure the output with clear headings and a concise index. "
    "Keep index entries to one line, under 150 characters, pointing to the key facts."
)

def get_default_client_and_model():
    """Fallback to default OpenAI client and model config."""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://z.apiyihe.org/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-TMd7SbPPbVw1JMx0GYKflkWkv8Mzi1tb0B64Y9HqBQ53TaqW")
    model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return client, model

async def generate_compaction_summary(
    client: AsyncOpenAI | None,
    model: str,
    messages: list[dict],
    custom_instructions: str = ""
) -> str:
    """Generates a summary of historical messages for session compaction."""
    if not client or not model:
        d_client, d_model = get_default_client_and_model()
        client = client or d_client
        model = model or d_model

    # Convert messages to a readable format
    formatted_chat = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        # Handle list/dict contents
        if isinstance(content, list):
            text = " ".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
        else:
            text = str(content)
        formatted_chat.append(f"{role.upper()}: {text[:1000]}") # truncate huge texts

    chat_text = "\n".join(formatted_chat)
    user_prompt = f"Please summarize the following conversation history:\n\n{chat_text}"
    system_prompt = COMPACT_INSTRUCTIONS
    if custom_instructions:
        system_prompt += f"\n\nAdditional instructions:\n{custom_instructions}"

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await chat(client, prompt_messages, model=model, max_tokens=1024)
    return response.content or ""

async def generate_session_summary(
    client: AsyncOpenAI | None,
    model: str,
    session_diary: str,
    jsonl_content: str
) -> str:
    """Summarizes a session for daily log consolidation."""
    if not client or not model:
        d_client, d_model = get_default_client_and_model()
        client = client or d_client
        model = model or d_model

    user_prompt = ""
    if session_diary:
        user_prompt += f"Session Diary (session.md):\n{session_diary}\n\n"
    user_prompt += f"Raw Chat Transcript (JSONL):\n{jsonl_content}"

    prompt_messages = [
        {"role": "system", "content": SESSION_SUMMARY_INSTRUCTIONS},
        {"role": "user", "content": user_prompt}
    ]
    response = await chat(client, prompt_messages, model=model, max_tokens=1024)
    return response.content or ""

async def dream_and_merge(
    client: AsyncOpenAI | None,
    model: str,
    old_memory: str,
    daily_logs: list[str]
) -> str:
    """Consolidates daily logs into the MEMORY.md file."""
    if not client or not model:
        d_client, d_model = get_default_client_and_model()
        client = client or d_client
        model = model or d_model

    user_prompt = f"Existing MEMORY.md:\n{old_memory}\n\n"
    user_prompt += "Recent Daily Logs to merge:\n"
    for i, log_content in enumerate(daily_logs):
        user_prompt += f"--- Log {i+1} ---\n{log_content}\n"

    prompt_messages = [
        {"role": "system", "content": DREAM_INSTRUCTIONS},
        {"role": "user", "content": user_prompt}
    ]
    response = await chat(client, prompt_messages, model=model, max_tokens=2048)
    return response.content or ""
