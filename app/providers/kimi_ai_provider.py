import json
import time
import uuid
import random
import string
import re
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional, List

import cloudscraper
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from cachetools import TTLCache
from loguru import logger

from app.core.config import settings
from app.providers.base_provider import BaseProvider
from app.utils.sse_utils import create_sse_data, create_chat_completion_chunk, DONE_CHUNK

class KimiAIProvider(BaseProvider):
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        # 缓存结构: { "user_key": {"kimi_session_id": "...", "messages": [...] } }
        self.session_cache = TTLCache(maxsize=1024, ttl=settings.SESSION_CACHE_TTL)
        self._nonce: Optional[str] = None
        self._nonce_lock = asyncio.Lock()

    async def initialize(self):
        """在服务启动时预取一次 nonce。"""
        logger.info("正在初始化 KimiAIProvider，首次获取 nonce...")
        await self._get_nonce()

    async def _fetch_nonce(self) -> str:
        """从聊天页面抓取动态的 nonce 值。"""
        try:
            logger.info("正在从上游页面抓取新的 nonce...")
            response = self.scraper.get(settings.CHAT_PAGE_URL, timeout=20)
            response.raise_for_status()
            html_content = response.text

            match = re.search(r'var kimi_ajax = ({.*?});', html_content)
            if not match:
                raise ValueError("在页面 HTML 中未找到 'kimi_ajax' JS 变量。")

            ajax_data = json.loads(match.group(1))
            nonce = ajax_data.get("nonce")
            if not nonce:
                raise ValueError("'kimi_ajax' 对象中缺少 'nonce' 字段。")
            
            logger.success(f"成功抓取到新的 nonce: {nonce}")
            return nonce
        except Exception as e:
            logger.error(f"抓取 nonce 失败: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"无法从上游服务获取必要的动态参数: {e}")

    async def _get_nonce(self, force_refresh: bool = False) -> str:
        """获取并缓存 nonce，处理并发请求和刷新逻辑。"""
        async with self._nonce_lock:
            if self._nonce is None or force_refresh:
                self._nonce = await self._fetch_nonce()
            return self._nonce

    def _get_or_create_session(self, user_key: str) -> Dict[str, Any]:
        """为用户获取或创建一个新的会话对象。"""
        if user_key in self.session_cache:
            return self.session_cache[user_key]
        
        timestamp = int(time.time() * 1000)
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
        new_session_id = f"session_{timestamp}_{random_str}"
        
        new_session = {
            "kimi_session_id": new_session_id,
            "messages": []
        }
        self.session_cache[user_key] = new_session
        logger.info(f"为用户 '{user_key}' 创建了新的会话: {new_session_id}")
        return new_session

    def _build_contextual_prompt(self, history: List[Dict[str, str]], new_message: str) -> str:
        """
        将历史记录和新消息拼接成一个单一的字符串，并根据需要截断以满足1000字符的限制。
        """
        # 格式化历史记录为 "角色: 内容" 的形式
        history_lines = []
        for msg in history:
            role = "用户" if msg.get("role") == "user" else "模型"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        
        history_str = "\n".join(history_lines)
        
        # 初始完整上下文
        full_prompt = f"{history_str}\n用户: {new_message}".strip()

        # 智能截断逻辑：如果超出长度，从头开始移除一轮对话（用户+模型）
        while len(full_prompt) > settings.CONTEXT_MAX_LENGTH and history:
            logger.warning(f"上下文超长 ({len(full_prompt)} > {settings.CONTEXT_MAX_LENGTH})，正在从头部截断...")
            # 移除最早的一条用户消息
            history.pop(0) 
            # 如果还有消息，再移除一条对应的模型消息
            if history:
                history.pop(0)
            
            history_lines = []
            for msg in history:
                role = "用户" if msg.get("role") == "user" else "模型"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(history_lines)
            full_prompt = f"{history_str}\n用户: {new_message}".strip()
        
        return full_prompt

    async def chat_completion(self, request_data: Dict[str, Any]) -> StreamingResponse:
        user_key = request_data.get("user")
        
        messages = request_data.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            raise HTTPException(status_code=400, detail="'messages' 列表不能为空，且最后一条必须是 user 角色。")

        current_user_message = messages[-1]
        
        # 根据是否存在 user_key 决定工作模式
        if user_key:
            # --- 有状态模式 ---
            logger.info(f"检测到 'user' 字段，进入有状态模式。用户: {user_key}")
            session_data = self._get_or_create_session(user_key)
            prompt_to_send = self._build_contextual_prompt(session_data["messages"], current_user_message["content"])
            kimi_session_id = session_data["kimi_session_id"]
        else:
            # --- 无状态模式 ---
            logger.info("未检测到 'user' 字段，进入无状态模式。")
            session_data = None # 无状态模式下没有会话数据
            prompt_to_send = current_user_message["content"]
            # 为无状态请求生成一次性的 session_id
            timestamp = int(time.time() * 1000)
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
            kimi_session_id = f"session_{timestamp}_{random_str}"

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            request_id = f"chatcmpl-{uuid.uuid4()}"
            model = request_data.get("model", settings.DEFAULT_MODEL)
            
            try:
                nonce = await self._get_nonce()
                payload = self._prepare_payload(prompt_to_send, model, kimi_session_id, nonce)
                
                logger.info(f"向上游发送请求, Session ID: {kimi_session_id}, 模型: {payload['model']}")
                logger.debug(f"发送的完整 Prompt: {prompt_to_send}")
                
                response = self.scraper.post(settings.UPSTREAM_URL, data=payload, timeout=settings.API_REQUEST_TIMEOUT)
                response.raise_for_status()
                
                response_data = response.json()
                if not response_data.get("success"):
                    error_message = response_data.get("data", "未知错误")
                    logger.warning(f"上游请求失败: {error_message}。可能 nonce 失效，正在尝试刷新并重试...")
                    
                    nonce = await self._get_nonce(force_refresh=True)
                    payload['nonce'] = nonce
                    
                    response = self.scraper.post(settings.UPSTREAM_URL, data=payload, timeout=settings.API_REQUEST_TIMEOUT)
                    response.raise_for_status()
                    response_data = response.json()

                    if not response_data.get("success"):
                         raise HTTPException(status_code=502, detail=f"重试后上游请求依然失败: {response_data.get('data', '未知错误')}")

                assistant_response_content = response_data.get("data", {}).get("message", "")
                
                # 如果是有状态模式，则更新服务端会话历史
                if session_data:
                    session_data["messages"].append(current_user_message)
                    session_data["messages"].append({"role": "assistant", "content": assistant_response_content})
                    self.session_cache[user_key] = session_data
                    logger.info(f"会话 '{user_key}' 上下文已更新。")

                # 应用【模式：伪流式生成】
                for char in assistant_response_content:
                    chunk = create_chat_completion_chunk(request_id, model, char)
                    yield create_sse_data(chunk)
                    await asyncio.sleep(0.02)

                final_chunk = create_chat_completion_chunk(request_id, model, "", "stop")
                yield create_sse_data(final_chunk)
                yield DONE_CHUNK

            except Exception as e:
                logger.error(f"处理流时发生错误: {e}", exc_info=True)
                error_message = f"内部服务器错误: {str(e)}"
                error_chunk = create_chat_completion_chunk(request_id, model, error_message, "stop")
                yield create_sse_data(error_chunk)
                yield DONE_CHUNK

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    def _prepare_payload(self, prompt: str, model: str, session_id: str, nonce: str) -> Dict[str, Any]:
        # 映射到上游接受的模型名称
        if model == "kimi-k2-instruct-0905":
            upstream_model = "moonshotai/Kimi-K2-Instruct-0905"
        elif model == "kimi-k2-instruct":
            upstream_model = "moonshotai/Kimi-K2-Instruct"
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型: {model}")

        return {
            "action": "kimi_send_message",
            "nonce": nonce,
            "message": prompt,
            "model": upstream_model,
            "session_id": session_id
        }

    async def get_models(self) -> JSONResponse:
        model_data = {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": int(time.time()), "owned_by": "lzA6"}
                for name in settings.KNOWN_MODELS
            ]
        }
        return JSONResponse(content=model_data)
