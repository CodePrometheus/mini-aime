"""网络搜索和信息获取工具。"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import requests
from langchain_tavily import TavilySearch

from .base import BaseTool, ToolError
from .rate_limiter import ExponentialBackoff, GlobalRateLimiter


logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """网络搜索工具，基于 Tavily Search 实现。"""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True,
        include_raw_content: bool = False,
    ):
        super().__init__(
            name="web_search",
            description="使用 AI 优化的搜索引擎进行网络信息搜索",
            required_permissions=["internet_access"],
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )

        # 获取 API Key
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ToolError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable or pass api_key parameter."
            )

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
            raise ToolError(f"Failed to initialize Tavily Search: {e!s}")

    async def execute(self, query: str, **kwargs) -> dict[str, Any]:
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
                    result = {"query": query, "answer": result, "results": [], "response_time": 0}

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
                "request_id": result.get("request_id"),
            }

            return standardized_result

        except Exception as e:
            raise ToolError(f"Web search failed for query '{query}': {e!s}")

    def execute_sync(self, query: str, **kwargs) -> dict[str, Any]:
        """同步版本的网络搜索。"""
        import asyncio

        return asyncio.run(self.execute(query, **kwargs))

    def format_results(self, results: dict[str, Any], max_length: int = 1000) -> str:
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

        if results.get("answer"):
            answer = results["answer"]
            if len(answer) > max_length // 2:
                answer = answer[: max_length // 2] + "..."
            formatted.append(f"💡 答案摘要: {answer}")

        # 添加搜索结果
        search_results = results.get("results", [])
        if search_results:
            formatted.append(f"\n📋 搜索结果 ({len(search_results)} 条):")

            for i, result in enumerate(search_results[:3], 1):  # 只显示前3条
                title = result.get("title", "无标题")
                url = result.get("url", "")
                content = result.get("content", "")

                # 截断内容
                if content and len(content) > 150:
                    content = content[:150] + "..."

                formatted.append(f"\n{i}. {title}")
                if url:
                    formatted.append(f"   🔗 {url}")
                if content:
                    formatted.append(f"   📄 {content}")

        # 添加响应时间
        if results.get("response_time"):
            formatted.append(f"\n⏱️ 响应时间: {results['response_time']:.2f}秒")

        result_text = "\n".join(formatted)

        # 确保不超过最大长度
        if len(result_text) > max_length:
            result_text = result_text[:max_length] + "...\n[结果已截断]"

        return result_text


class BraveSearchTool(BaseTool):
    """基于 Brave Search API 的网络搜索工具，支持智能速率限制和重试。"""

    # 类级别配置（可通过环境变量覆盖）
    _rate_limit = None  # 每秒请求数
    _rate_limiter_name = "brave_search_api"
    _max_retries = None  # 最大重试次数
    _retry_base_delay = None  # 重试基础延迟

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 3,
        country: str = "US",
        search_lang: str = "en",
        ui_lang: str = "en-US",
        safe_search: str = "moderate",
        freshness: str | None = None,
        text_decorations: bool = True,
        spellcheck: bool = True,
        rate_limit: float | None = None,
        max_retries: int | None = None,
    ):
        super().__init__(
            name="brave_search",
            description="使用 Brave Search API 进行网络信息搜索",
            required_permissions=["internet_access"],
            max_results=max_results,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
            safe_search=safe_search,
            freshness=freshness,
            text_decorations=text_decorations,
            spellcheck=spellcheck,
        )

        # 获取 API Key
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            raise ToolError(
                "Brave Search API key is required. Set BRAVE_SEARCH_API_KEY environment variable or pass api_key parameter."
            )

        # API 配置
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        # 搜索参数
        self.max_results = max_results
        self.country = country
        self.search_lang = search_lang
        self.ui_lang = ui_lang
        self.safe_search = safe_search
        self.freshness = freshness
        self.text_decorations = text_decorations
        self.spellcheck = spellcheck

        # 速率限制配置（优先级：参数 > 环境变量 > 默认值）
        if rate_limit is not None:
            self.rate_limit = rate_limit
        elif BraveSearchTool._rate_limit is not None:
            self.rate_limit = BraveSearchTool._rate_limit
        else:
            # 从环境变量读取，默认0.95 QPS（略低于1.0以留出安全边际）
            self.rate_limit = float(os.getenv("BRAVE_SEARCH_RATE_LIMIT", "0.95"))

        # 重试配置
        if max_retries is not None:
            self.max_retries = max_retries
        elif BraveSearchTool._max_retries is not None:
            self.max_retries = BraveSearchTool._max_retries
        else:
            self.max_retries = int(os.getenv("BRAVE_SEARCH_MAX_RETRIES", "3"))

        if BraveSearchTool._retry_base_delay is not None:
            self.retry_base_delay = BraveSearchTool._retry_base_delay
        else:
            self.retry_base_delay = float(os.getenv("BRAVE_SEARCH_RETRY_BASE_DELAY", "2.0"))

        logger.debug(
            f"BraveSearchTool initialized: rate_limit={self.rate_limit} QPS, "
            f"max_retries={self.max_retries}, retry_base_delay={self.retry_base_delay}s"
        )

    async def execute(self, query: str, **kwargs) -> dict[str, Any]:
        """
        执行网络搜索（带速率限制和智能重试）。

        Args:
            query: 搜索查询
            **kwargs: 额外的搜索参数，可以覆盖默认配置

        Returns:
            搜索结果字典，包含：
            - query: 搜索查询
            - web: 网页搜索结果
            - infobox: 信息框结果（如果有）
            - news: 新闻结果（如果有）
            - videos: 视频结果（如果有）
            - locations: 位置结果（如果有）
            - response_time: 响应时间
            - retry_count: 重试次数（如果有）

        Raises:
            ToolError: 搜索失败
        """
        if not query or not query.strip():
            raise ToolError("Search query cannot be empty")

        # 构建搜索参数
        params = {
            "q": query.strip(),
            "count": kwargs.get("count", self.max_results),
            "offset": kwargs.get("offset", 0),
            "country": kwargs.get("country", self.country),
            "search_lang": kwargs.get("search_lang", self.search_lang),
            "ui_lang": kwargs.get("ui_lang", self.ui_lang),
            "safesearch": kwargs.get("safe_search", self.safe_search),
            "text_decorations": kwargs.get("text_decorations", self.text_decorations),
            "spellcheck": kwargs.get("spellcheck", self.spellcheck),
        }

        # 添加可选参数
        if self.freshness or kwargs.get("freshness"):
            params["freshness"] = kwargs.get("freshness", self.freshness)

        # 初始化重试机制
        backoff = ExponentialBackoff(
            base=self.retry_base_delay, max_delay=60.0, multiplier=2.0, jitter=True
        )

        last_error = None
        retry_count = 0

        # 重试循环
        for attempt in range(self.max_retries + 1):
            try:
                # 获取全局速率限制器
                limiter = await GlobalRateLimiter.get_limiter(
                    name=self._rate_limiter_name,
                    rate=self.rate_limit,
                    capacity=max(1, int(self.rate_limit * 2)),  # 允许短时突发
                )

                # 等待速率限制
                await limiter.acquire(tokens=1, timeout=30.0)
                logger.debug(f"Brave Search rate limit acquired for query: {query[:50]}...")

                start_time = time.time()

                # 在线程池中执行同步请求（避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        self.base_url, headers=self.headers, params=params, timeout=30
                    ),
                )

                response_time = time.time() - start_time

                # 检查响应状态
                if response.status_code == 429:
                    # 速率限制错误，触发重试
                    retry_count = attempt
                    logger.warning(
                        f"Brave Search rate limit hit (429) on attempt {attempt + 1}/{self.max_retries + 1}, "
                        f"query: {query[:50]}..."
                    )

                    if attempt < self.max_retries:
                        delay = backoff.get_delay()
                        logger.info(f"Retrying after {delay:.2f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise ToolError(
                            f"Rate limit exceeded after {self.max_retries} retries. "
                            "Please reduce request frequency or upgrade your Brave Search plan."
                        )

                elif response.status_code == 401:
                    raise ToolError("Invalid API key or unauthorized access.")

                elif response.status_code != 200:
                    raise ToolError(
                        f"Brave Search API error: {response.status_code} - {response.text}"
                    )

                # 解析响应
                result = response.json()

                # 标准化结果格式
                standardized_result = {
                    "query": query,
                    "web": result.get("web", {}),
                    "infobox": result.get("infobox"),
                    "news": result.get("news"),
                    "videos": result.get("videos"),
                    "locations": result.get("locations"),
                    "response_time": response_time,
                    "total_results": result.get("web", {}).get("total", 0),
                    "retry_count": retry_count,
                }

                if retry_count > 0:
                    logger.info(
                        f"Brave Search succeeded after {retry_count} retries, query: {query[:50]}..."
                    )

                return standardized_result

            except ToolError:
                # ToolError 直接抛出，不重试
                raise

            except requests.RequestException as e:
                last_error = e
                retry_count = attempt
                logger.warning(
                    f"Network error on attempt {attempt + 1}/{self.max_retries + 1}: {e!s}"
                )

                if attempt < self.max_retries:
                    delay = backoff.get_delay()
                    logger.info(f"Retrying after {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ToolError(
                        f"Network error during Brave search after {self.max_retries} retries "
                        f"for query '{query}': {e!s}"
                    )

            except json.JSONDecodeError as e:
                raise ToolError(f"Failed to parse Brave search response: {e!s}")

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e!s}")

                if attempt < self.max_retries:
                    delay = backoff.get_delay()
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ToolError(f"Brave search failed for query '{query}': {e!s}")

        # 理论上不会到达这里，但为了类型安全
        raise ToolError(f"Brave search failed after all retries: {last_error!s}")

    def execute_sync(self, query: str, **kwargs) -> dict[str, Any]:
        """同步版本的网络搜索。"""
        import asyncio

        try:
            # 尝试获取当前事件循环
            asyncio.get_running_loop()
            # 如果已经在事件循环中，创建新任务
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.execute(query, **kwargs))
                return future.result()
        except RuntimeError:
            # 没有运行的事件循环，可以直接使用 asyncio.run
            return asyncio.run(self.execute(query, **kwargs))

    def format_results(self, results: dict[str, Any], max_length: int = 1000) -> str:
        """
        格式化搜索结果为可读字符串。

        Args:
            results: 搜索结果字典
            max_length: 最大长度限制

        Returns:
            格式化的搜索结果字符串
        """
        formatted = []

        # 添加查询信息
        formatted.append(f"🔍 查询: {results.get('query', '')}")

        # 添加总结果数
        total_results = results.get("total_results", 0)
        if total_results > 0:
            formatted.append(f"📊 找到约 {total_results:,} 条结果")

        # 添加信息框（如果有）
        infobox = results.get("infobox")
        if infobox:
            formatted.append("\n📋 信息框:")
            if infobox.get("title"):
                formatted.append(f"   标题: {infobox['title']}")
            if infobox.get("description"):
                desc = infobox["description"]
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                formatted.append(f"   描述: {desc}")

        # 添加网页搜索结果
        web_results = results.get("web", {}).get("results", [])
        if web_results:
            formatted.append("\n🌐 网页结果:")

            for i, result in enumerate(web_results[:3], 1):  # 显示前3条
                title = result.get("title", "无标题")
                url = result.get("url", "")
                description = result.get("description", "")

                # 截断描述
                if description and len(description) > 150:
                    description = description[:150] + "..."

                formatted.append(f"\n{i}. {title}")
                if url:
                    formatted.append(f"   🔗 {url}")
                if description:
                    formatted.append(f"   📄 {description}")

        # 添加新闻结果（如果有）
        news_data = results.get("news")
        if news_data and isinstance(news_data, dict):
            news_results = news_data.get("results", [])
            if news_results:
                formatted.append("\n📰 新闻结果:")
                for i, news in enumerate(news_results[:3], 1):  # 显示前3条新闻
                    title = news.get("title", "无标题")
                    url = news.get("url", "")
                    age = news.get("age", "")

                    formatted.append(f"\n{i}. {title}")
                    if url:
                        formatted.append(f"   🔗 {url}")
                    if age:
                        formatted.append(f"   📅 {age}")

        # 添加响应时间和重试信息
        response_time = results.get("response_time")
        retry_count = results.get("retry_count", 0)

        if response_time or retry_count > 0:
            info_parts = []
            if response_time:
                info_parts.append(f"响应时间: {response_time:.2f}秒")
            if retry_count > 0:
                info_parts.append(f"重试次数: {retry_count}")
            formatted.append(f"\n⏱️ {', '.join(info_parts)}")

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
            required_permissions=["internet_access"],
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
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # 解析HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.decompose()

            # 提取文本内容
            text = soup.get_text()

            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # 限制长度
            if len(text) > max_length:
                text = text[:max_length] + "...[内容已截断]"

            return text

        except requests.RequestException as e:
            raise ToolError(f"Failed to fetch URL {url}: {e!s}")
        except Exception as e:
            raise ToolError(f"Failed to extract content from {url}: {e!s}")

    def execute_sync(self, url: str, max_length: int = 5000) -> str:
        """同步版本的网页内容提取。"""
        import asyncio

        return asyncio.run(self.execute(url, max_length))


# 导出的类
__all__ = ["BraveSearchTool", "WebContentExtractorTool", "WebSearchTool"]
