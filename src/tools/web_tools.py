"""网络搜索和信息获取工具。"""

import json
import os
from typing import Any, Dict, List, Optional

from langchain_tavily import TavilySearch

from .base import BaseTool, ToolError


class WebSearchTool(BaseTool):
    """网络搜索工具，基于 Tavily Search 实现。"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True,
        include_raw_content: bool = False
    ):
        super().__init__(
            name="web_search",
            description="使用 AI 优化的搜索引擎进行网络信息搜索",
            required_permissions=["internet_access"],
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content
        )
        
        # 获取 API Key
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ToolError("Tavily API key is required. Set TAVILY_API_KEY environment variable or pass api_key parameter.")
        
        # 初始化 Tavily 搜索
        try:
            self.tavily = TavilySearch(
                tavily_api_key=self.api_key,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
            )
        except Exception as e:
            raise ToolError(f"Failed to initialize Tavily Search: {str(e)}")
    
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行网络搜索。
        
        Args:
            query: 搜索查询
            **kwargs: 额外的搜索参数
            
        Returns:
            搜索结果字典，包含：
            - query: 搜索查询
            - answer: AI 生成的答案摘要
            - results: 详细搜索结果列表
            - response_time: 响应时间
            
        Raises:
            ToolError: 搜索失败
        """
        if not query or not query.strip():
            raise ToolError("Search query cannot be empty")
        
        try:
            # 执行搜索
            result = self.tavily.run(query)
            
            # 确保结果是字典格式
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    # 如果不是 JSON，包装成字典
                    result = {
                        "query": query,
                        "answer": result,
                        "results": [],
                        "response_time": 0
                    }
            
            # 验证和标准化结果格式
            if not isinstance(result, dict):
                raise ToolError(f"Unexpected result format: {type(result)}")
            
            # 确保必要字段存在
            standardized_result = {
                "query": result.get("query", query),
                "answer": result.get("answer", ""),
                "results": result.get("results", []),
                "response_time": result.get("response_time", 0),
                "images": result.get("images", []),
                "follow_up_questions": result.get("follow_up_questions"),
                "request_id": result.get("request_id")
            }
            
            return standardized_result
            
        except Exception as e:
            raise ToolError(f"Web search failed for query '{query}': {str(e)}")
    
    def execute_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """同步版本的网络搜索。"""
        import asyncio
        return asyncio.run(self.execute(query, **kwargs))
    
    def format_results(self, results: Dict[str, Any], max_length: int = 1000) -> str:
        """
        格式化搜索结果为可读字符串。
        
        Args:
            results: 搜索结果字典
            max_length: 最大长度限制
            
        Returns:
            格式化的搜索结果字符串
        """
        formatted = []
        
        # 添加查询和答案
        formatted.append(f"🔍 查询: {results.get('query', '')}")
        
        if results.get('answer'):
            answer = results['answer']
            if len(answer) > max_length // 2:
                answer = answer[:max_length//2] + "..."
            formatted.append(f"💡 答案摘要: {answer}")
        
        # 添加搜索结果
        search_results = results.get('results', [])
        if search_results:
            formatted.append(f"\n📋 搜索结果 ({len(search_results)} 条):")
            
            for i, result in enumerate(search_results[:3], 1):  # 只显示前3条
                title = result.get('title', '无标题')
                url = result.get('url', '')
                content = result.get('content', '')
                
                # 截断内容
                if content and len(content) > 150:
                    content = content[:150] + "..."
                
                formatted.append(f"\n{i}. {title}")
                if url:
                    formatted.append(f"   🔗 {url}")
                if content:
                    formatted.append(f"   📄 {content}")
        
        # 添加响应时间
        if results.get('response_time'):
            formatted.append(f"\n⏱️ 响应时间: {results['response_time']:.2f}秒")
        
        result_text = "\n".join(formatted)
        
        # 确保不超过最大长度
        if len(result_text) > max_length:
            result_text = result_text[:max_length] + "...\n[结果已截断]"
        
        return result_text


class WebContentExtractorTool(BaseTool):
    """网页内容提取工具。"""
    
    def __init__(self):
        super().__init__(
            name="extract_web_content",
            description="从指定URL提取网页内容",
            required_permissions=["internet_access"]
        )
    
    async def execute(self, url: str, max_length: int = 5000) -> str:
        """
        提取网页内容。
        
        Args:
            url: 网页URL
            max_length: 最大内容长度
            
        Returns:
            提取的网页内容
            
        Raises:
            ToolError: 内容提取失败
        """
        if not url or not url.strip():
            raise ToolError("URL cannot be empty")
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # 发送HTTP请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # 解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取文本内容
            text = soup.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # 限制长度
            if len(text) > max_length:
                text = text[:max_length] + "...[内容已截断]"
            
            return text
            
        except requests.RequestException as e:
            raise ToolError(f"Failed to fetch URL {url}: {str(e)}")
        except Exception as e:
            raise ToolError(f"Failed to extract content from {url}: {str(e)}")
    
    def execute_sync(self, url: str, max_length: int = 5000) -> str:
        """同步版本的网页内容提取。"""
        import asyncio
        return asyncio.run(self.execute(url, max_length))
