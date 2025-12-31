"""
异步API客户端 - 支持GPU绑定和异步并发
"""
import asyncio
import aiohttp
import base64
import io
import json
import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from PIL import Image
import config
import os


class AsyncGeminiClient:
    """异步API客户端，支持高并发和GPU绑定"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, 
                 base_url: Optional[str] = None, gpu_id: Optional[int] = None,
                 max_concurrent: int = 10):
        """
        初始化异步客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础URL
            gpu_id: 绑定的GPU ID（用于进程隔离，不影响API调用）
            max_concurrent: 最大并发请求数
        """
        self.api_key = api_key or config.API_KEY
        self.model_name = model_name or config.MODEL_NAME
        self.base_url = base_url or config.BASE_URL
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent
        
        if not self.api_key:
            raise ValueError("API Key未设置")
        
        if not self.base_url:
            raise ValueError("Base URL未设置")
        
        # 设置GPU可见性（用于进程隔离）
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[INFO] GPU绑定: GPU {gpu_id}")
        
        # 创建会话
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """编码图片为base64"""
        try:
            image = self._load_image(image_input)
            
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image.close()
                image = rgb_image
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            
            return image_base64
        finally:
            if 'image' in locals() and image is not None:
                try:
                    image.close()
                except:
                    pass
    
    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """加载图片"""
        if isinstance(image_input, Image.Image):
            return image_input
        
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        
        if isinstance(image_input, (str, Path)):
            image_str = str(image_input)
            
            if image_str.startswith(('http://', 'https://')):
                # URL需要异步下载，这里先不支持
                raise ValueError("URL图片需要异步下载，请使用load_image_async")
            
            if image_str.startswith('data:image'):
                base64_data = image_str.split(',', 1)[1]
            elif len(image_str) > 100:
                try:
                    clean_str = re.sub(r'\s', '', image_str)
                    base64.b64decode(clean_str)
                    base64_data = clean_str
                except:
                    base64_data = None
            else:
                base64_data = None
            
            if base64_data:
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes))
            else:
                return Image.open(image_input)
        
        raise ValueError(f"不支持的图片类型: {type(image_input)}")
    
    async def analyze_image_async(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        prompt: str,
        temperature: float = 0.7
    ) -> str:
        """
        异步分析图片
        
        Args:
            image_input: 图片输入
            prompt: 提示词
            temperature: 温度参数
            
        Returns:
            响应文本
        """
        async with self.semaphore:  # 限制并发数
            if not self.session:
                raise RuntimeError("Session not initialized. Use async with statement.")
            
            # 编码图片
            image_base64 = self._encode_image(image_input)
            
            # 构建请求
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "stream": False,
                "max_tokens": 4096,
                "temperature": temperature
            }
            
            # 发送请求
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def filter_image_async(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        criteria_description: str,
        question: str,
        temperature: float = 0.3
    ) -> dict:
        """
        异步筛选图片
        
        Args:
            image_input: 图片输入
            criteria_description: 筛选标准描述
            question: 问题描述
            
        Returns:
            筛选结果字典
        """
        prompt = """
You are a professional image filtering and quality evaluation expert.

Your task is to evaluate whether an image meets the required criteria, and to assign a quality score based on how well it satisfies both required and optional standards.

Question Type:
{question}

Evaluation Criteria:
{criteria_description}

Please follow the rules below strictly.


1. Required (non-optional) Criteria:
- These are mandatory criteria.
- ALL required criteria must be satisfied for the image to pass.
- If ANY required criterion is not satisfied:
  - "passed" must be false
  - "score" must be 0.0
  - Optional criteria must NOT be considered
- If ALL required criteria are satisfied:
  - "passed" is true
  - The score starts from a base score of 0.1
  - The quality score for required criteria ranges from 0.1 to 0.6
  - The closer the image matches the required criteria (clarity, correctness, completeness, alignment),
    the closer the score should be to 0.6

2. Optional Criteria:
- Optional criteria are marked explicitly as "optional" in the criteria.
- Optional criteria are considered ONLY IF all required criteria are satisfied.
- Optional criteria represent higher difficulty and higher value.

Scoring with Optional Criteria:
- If there are NO optional criteria:
  - Automatically add 0.4 to the score (i.e., total score = required score + 0.4)
- If there ARE optional criteria:
  - A bonus score up to 0.4 is available
  - The more optional criteria are satisfied, and the better they are satisfied,
    the higher the bonus score (from 0.0 to 0.4)

3. Final Score:
- Total score = required score (max 0.6) + optional bonus (max 0.4)
- Final score must be in the range [0.0, 1.0]
- Images satisfying optional criteria well should score higher than those that only satisfy required criteria.

4. Confidence:
- "confidence" represents how confident you are in your judgment
- It must be a float between 0.0 and 1.0
- Confidence should be lower if the image is ambiguous or borderline


Return the result in JSON format ONLY, with no extra text:

{{
  "passed": true/false,
  "basic_score": float,
  "bonus_score": float (0.0-0.4),
  "total_score": float (0.0–1.0),
  "reason": "Detailed explanation of which required and optional criteria are satisfied or violated, and how the score is determined",
  "confidence": float (0.0–1.0)
}}""".format(question=question, criteria_description=criteria_description)
        
        try:
            response_text = await self.analyze_image_async(image_input, prompt, temperature)
            
            # 解析JSON响应
            json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_block_match:
                response_text = json_block_match.group(1)
            else:
                start_idx = response_text.find('{')
                if start_idx != -1:
                    brace_count = 0
                    for i in range(start_idx, len(response_text)):
                        if response_text[i] == '{':
                            brace_count += 1
                        elif response_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0 and '"passed"' in response_text[start_idx:i+1]:
                                response_text = response_text[start_idx:i+1]
                                break
            
            result = json.loads(response_text)
            
            # 验证结果格式
            if "passed" not in result:
                result["passed"] = False
            if "reason" not in result:
                result["reason"] = "无法解析筛选结果"
            if "confidence" not in result:
                result["confidence"] = 0.5
            
            return result
            
        except Exception as e:
            return {
                "passed": False,
                "reason": f"筛选过程出错: {str(e)}",
                "confidence": 0.0
            }


async def process_batch_async(
    items: List[Dict[str, Any]],
    num_gpus: int = 8,
    max_concurrent_per_gpu: int = 10
) -> List[Dict[str, Any]]:
    """
    使用多GPU异步处理批量数据
    
    Args:
        items: 待处理的数据项列表
        num_gpus: GPU数量
        max_concurrent_per_gpu: 每个GPU的最大并发数
        
    Returns:
        处理结果列表
    """
    # 将任务分配到不同的GPU
    tasks_per_gpu = len(items) // num_gpus
    gpu_tasks = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * tasks_per_gpu
        if gpu_id == num_gpus - 1:
            end_idx = len(items)  # 最后一个GPU处理剩余所有任务
        else:
            end_idx = (gpu_id + 1) * tasks_per_gpu
        
        gpu_tasks.append((gpu_id, items[start_idx:end_idx]))
    
    # 为每个GPU创建处理任务
    async def process_gpu_tasks(gpu_id: int, tasks: List[Dict]):
        """处理单个GPU的任务"""
        results = []
        async with AsyncGeminiClient(
            gpu_id=gpu_id,
            max_concurrent=max_concurrent_per_gpu
        ) as client:
            # 创建所有异步任务
            async_tasks = []
            for item in tasks:
                # 这里需要根据实际需求调用相应的异步方法
                # 示例：假设item包含image_input, criteria_description, question
                task = client.filter_image_async(
                    image_input=item.get("image_input"),
                    criteria_description=item.get("criteria_description", ""),
                    question=item.get("question", ""),
                    temperature=0.3
                )
                async_tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "error": str(result),
                        "item": tasks[i]
                    })
                else:
                    processed_results.append(result)
        
        return processed_results
    
    # 并发处理所有GPU的任务
    all_results = await asyncio.gather(*[
        process_gpu_tasks(gpu_id, tasks)
        for gpu_id, tasks in gpu_tasks
    ])
    
    # 合并结果
    final_results = []
    for gpu_results in all_results:
        final_results.extend(gpu_results)
    
    return final_results

