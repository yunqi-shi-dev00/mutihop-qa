"""
半导体QA生成系统 - LLM客户端模块
支持vLLM和SGLang推理框架
"""

import asyncio
import aiohttp
import requests
from typing import Dict


class LLMAPIClient:
    """统一的LLM API客户端，支持vLLM和SGLang（原版完整保留）"""
    
    def __init__(self, model_path: str, server_type: str = "vllm", 
                 host: str = "localhost", port: int = 8000, max_retries: int = 3):
        self.model_path = model_path
        self.server_type = server_type.lower()
        self.base_url = f"http://{host}:{port}"
        self.session = None
        self.max_retries = max_retries
        self.is_connected = False
        
        print(f"\n{'='*60}")
        print(f"[LLM] 初始化 {server_type.upper()} 客户端")
        print(f"[LLM] 模型: {model_path}")
        print(f"[LLM] 服务器: {self.base_url}")
        print(f"{'='*60}\n")
        
        self._check_server()
    
    def _check_server(self):
        """检查服务器是否可用"""
        test_url = f"{self.base_url}/v1/models"
        
        try:
            res = requests.get(test_url, timeout=10)
            if res.status_code == 200:
                print(f"[LLM] ✓ 服务器连接成功")
                self.is_connected = True
                models_info = res.json()
                print(f"[LLM] 可用模型: {models_info}")
                return
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] ✗ 无法连接到服务器 {self.base_url}: {e}")
        
        print(f"[WARNING] 请确保已启动 {self.server_type} 服务")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def async_generate(self, prompt: str, sampling_kwargs: Dict):
        """异步生成，使用聊天格式"""
        for attempt in range(self.max_retries):
            try:
                return await self._vllm_chat_generate(prompt, sampling_kwargs)
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"[RETRY] 第{attempt + 1}次重试，等待{wait_time}秒...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                raise
    
    async def _vllm_chat_generate(self, prompt: str, sampling_kwargs: Dict):
        """vLLM聊天格式生成"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        n = sampling_kwargs.get('n', 1)
        
        # 使用聊天格式
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": sampling_kwargs.get('max_new_tokens', 8192),
            "temperature": sampling_kwargs.get('temperature', 0.8),
            "top_p": sampling_kwargs.get('top_p', 0.95),
            "top_k": sampling_kwargs.get('top_k', -1),
            "n": n,
            "stream": False
        }
        
        # 移除None值
        payload = {k: v for k, v in payload.items() if v is not None}
        
        async with self.session.post(
            url=f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"[ERROR] 服务器返回错误: {response.status}, {error_text}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"HTTP {response.status}: {error_text}"
                )
            
            result = await response.json()
            
            if n == 1:
                return {"text": result['choices'][0]['message']['content']}
            else:
                texts = [choice['message']['content'] for choice in result['choices']]
                return {"text": texts}
