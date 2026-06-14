import re
import json
from typing import Any, Dict, List

# 用于匹配图片 URL 的正则 (png, jpg, jpeg, webp, gif)
IMG_URL_PATTERN = re.compile(
    r'https?://[^\s/$.?#].[^\s]*?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s]*)?', 
    re.IGNORECASE
)

def process_messages_vision(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析消息历史，实现动态视觉上下文挂载与直传图片 URL 的提取。
    确保将图片 URL 原生转为大模型支持的多模态格式，同时避免不必要的重复。
    
    设计考量（Token 优化与缓存友好）：
    1. 动态绑定的图片只挂载在最新的（最后一个）user 消息上。这样做既能保证当前轮次有足够的视觉上下文，
       又能避免历史交互中已过期的图片 URL (如火山引擎具有时效性的 TOS 签名 URL) 导致大模型 API 返回 500 报错挂死会话。
    2. 挂载时保留 user 消息中原本包含的所有 content parts (例如用户最初上传的参考图 reference_image_list)，
       防止原有多模态字段被覆盖丢失。
    """
    transformed_messages = []
    user_image_urls = {}  # {user_msg_index: list_of_urls}

    # 找出整个消息历史中最后一个 user 消息的索引
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    # 1. 扫描所有 user 消息中的文本链接
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            text_content = ""
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content = part.get("text", "")
                        break
            if text_content:
                urls = IMG_URL_PATTERN.findall(text_content)
                if urls:
                    if idx not in user_image_urls:
                        user_image_urls[idx] = []
                    for url in urls:
                        if url not in user_image_urls[idx]:
                            user_image_urls[idx].append(url)

    # 2. 扫描所有成功的 tool 消息输出并向前绑定到发起 user 消息
    for idx, msg in enumerate(messages):
        if msg.get("role") == "tool":
            content = msg.get("content")
            if not content:
                continue
                
            img_urls = []
            
            # 获取该 tool 消息对应的工具名称
            tool_name = "unknown"
            tool_call_id = msg.get("tool_call_id")
            for prev_msg in reversed(messages[:idx]):
                if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                    for tc in prev_msg["tool_calls"]:
                        if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                            tool_name = tc.get("function", {}).get("name", "unknown")
                            break
            
            # 如果是 read_image，由于其 output 可能是普通文本，优先用正则匹配其中的图片 URL
            if tool_name == "read_image":
                img_urls = IMG_URL_PATTERN.findall(content)
            
            # 否则，尝试作为 JSON 进行解析
            if not img_urls:
                try:
                    json_str = content
                    if json_str.startswith("Error: "):
                        json_str = json_str[7:]
                    data = json.loads(json_str)
                    if isinstance(data, dict) and data.get("url"):
                        img_urls.append(data["url"])
                except Exception:
                    pass
            
            # 如果依然没有提取出来，且工具执行成功，用正则作为兜底提取单个 URL
            if not img_urls and not content.startswith("Error:"):
                urls = IMG_URL_PATTERN.findall(content)
                if urls:
                    # 仅限 generate_image, read_image 等图像生成/查看工具，避免在 get_canvas_graph 等工具中提取过多无用图片导致 prompt 膨胀
                    if tool_name in ["generate_image", "read_image"]:
                        img_urls.extend(urls)

            if img_urls:
                # 向前寻找发起该互动的 user 消息
                initiator_idx = None
                for j in range(idx - 1, -1, -1):
                    if messages[j].get("role") == "user":
                        initiator_idx = j
                        break
                # 优化关键点：只将动态提取的图片绑定到最新的那个 user 消息上，避免历史过期图片导致 500 报错并节省视觉 token
                if initiator_idx is not None and initiator_idx == last_user_idx:
                    if initiator_idx not in user_image_urls:
                        user_image_urls[initiator_idx] = []
                    for img_url in img_urls:
                        if img_url not in user_image_urls[initiator_idx]:
                            user_image_urls[initiator_idx].append(img_url)

    # 3. 构建多模态的 transformed_messages
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "user":
            # 如果不是最后一个 user 消息，则过滤掉所有的 image_url，只保留 text，防止历史过期或不可达的图片 URL 导致大模型 API 返回 500 报错
            if idx != last_user_idx:
                if isinstance(content, list):
                    content = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
                    if len(content) == 1:
                        content = content[0].get("text", "")
                    elif len(content) == 0:
                        content = ""
                new_msg = dict(msg)
                new_msg["content"] = content
                transformed_messages.append(new_msg)
                continue
                
            # 如果是最后一个 user 消息，挂载可能存在的动态/参考图片 URL
            urls = user_image_urls.get(idx, [])
            if urls or isinstance(content, list):
                content_parts = []
                if isinstance(content, str):
                    content_parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    content_parts.extend(content)
                
                # 收集现有 content_parts 中已有的 image_url，避免重复添加
                existing_urls = set()
                for part in content_parts:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        u = part.get("image_url", {}).get("url")
                        if u:
                            existing_urls.add(u)
                            
                for url in urls:
                    if url not in existing_urls:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                        existing_urls.add(url)
                        
                new_msg = dict(msg)
                new_msg["content"] = content_parts
                transformed_messages.append(new_msg)
                continue
                
        transformed_messages.append(msg)
        
    return transformed_messages

def merge_consecutive_user_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并对话历史中相邻的连续 user 消息，将其 text 字段以双换行合并，并保留所有的 image_url，
    以保证大模型 API 的 role 严格交替规范，并防止连续追问导致模型滑入纯口头回复闲聊模式。
    """
    if not messages:
        return []
        
    merged = []
    for msg in messages:
        if not merged:
            merged.append(dict(msg))
            continue
            
        last = merged[-1]
        if last.get("role") == "user" and msg.get("role") == "user":
            # 提取 last 的 content parts
            def to_parts(c):
                if isinstance(c, str):
                    return [{"type": "text", "text": c}]
                elif isinstance(c, list):
                    return list(c)
                return []
                
            last_parts = to_parts(last.get("content"))
            curr_parts = to_parts(msg.get("content"))
            
            combined_parts = []
            combined_parts.extend(last_parts)
            combined_parts.extend(curr_parts)
            
            # 合并连续 of text parts
            optimized_parts = []
            for part in combined_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_val = part.get("text", "")
                    if optimized_parts and optimized_parts[-1].get("type") == "text":
                        optimized_parts[-1]["text"] += "\n\n" + text_val
                    else:
                        optimized_parts.append(dict(part))
                else:
                    if isinstance(part, dict):
                        optimized_parts.append(dict(part))
                    else:
                        optimized_parts.append(part)
                        
            last["content"] = optimized_parts
        else:
            merged.append(dict(msg))
            
    return merged
