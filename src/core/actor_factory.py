"""基于 LangChain 工具的动态智能体工厂。"""

import inspect
import json
import os
import uuid
from datetime import datetime
from typing import Any, Union

import requests
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

from ..llm.base import BaseLLMClient
from ..tools.web_tools import BraveSearchTool
from src.tools.file_tools import _find_project_root


class TaskSpecification(BaseModel):
    """用于创建智能体的任务规格。"""

    task_id: str = Field(..., description="任务唯一标识")
    description: str = Field(..., description="任务描述")
    parent_goal: str | None = Field(None, description="父级目标")
    context: dict[str, Any] = Field(default_factory=dict, description="执行上下文")
    constraints: list[str] = Field(default_factory=list, description="约束条件")
    priority: str = Field(default="medium", description="优先级")
    estimated_duration: int | None = Field(None, description="预估时长(分钟)")


class TaskAnalysis(BaseModel):
    """LLM 对任务需求分析的结果。"""

    task_type: str = Field(..., description="任务类型")
    domain: str = Field(..., description="领域")
    required_capabilities: list[str] = Field(..., description="所需能力")
    complexity: str = Field(..., description="复杂度")
    key_challenges: list[str] = Field(..., description="关键挑战")
    recommended_tools: list[str] = Field(..., description="推荐工具包")
    knowledge_areas: list[str] = Field(..., description="知识领域")


class ActorConfiguration(BaseModel):
    """智能体配置。"""

    actor_id: str = Field(..., description="智能体ID")
    task_id: str = Field(..., description="关联任务ID")
    persona: str = Field(..., description="人格设定")
    tools: list[str] = Field(..., description="工具名称列表")
    knowledge: str = Field(..., description="注入的知识")
    system_prompt: str = Field(..., description="完整系统提示")
    execution_config: dict[str, Any] = Field(default_factory=dict, description="执行配置")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class ActorFactory:
    """按需创建具备动态配置的专用智能体的工厂。"""

    def __init__(self, llm_client: BaseLLMClient, initialize_tools: bool = True):
        self.llm = llm_client
        # 工具使用统计和建议记录
        self.tool_usage_stats = {}
        self.tool_recommendations_history = []

        if initialize_tools:
            self._initialize_tool_bundles()
        else:
            # 测试模式：使用空的工具包
            self.tool_bundles = {
                "web_research": [],
                "file_operations": [],
                "weather_services": [],
                "data_processing": []
            }

    def _initialize_tool_bundles(self):
        """初始化工具包。"""

        # 网络搜索工具 - 优先使用 Brave Search，回退到 Tavily
        web_search_tools = []

        # 尝试初始化 Brave Search
        brave_api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if brave_api_key:
            try:
                brave_search_tool = BraveSearchTool(api_key=brave_api_key, max_results=2)

                # 创建 LangChain 工具包装器
                def brave_search_func(query: str) -> str:
                    """使用 Brave Search 进行网络搜索。"""
                    try:
                        result = brave_search_tool.execute_sync(query)
                        return brave_search_tool.format_results(result, max_length=1000)
                    except Exception as e:
                        return f"Brave 搜索失败: {e!s}"

                brave_search_langchain = Tool(
                    name="brave_search",
                    func=brave_search_func,
                    description="使用 Brave Search API 进行网络搜索，返回格式化的搜索结果"
                )
                web_search_tools.append(brave_search_langchain)

            except Exception as e:
                print(f"警告: Brave Search 初始化失败: {e!s}")

        # 暂时注释掉 Tavily Search，优先使用 Brave Search
        # tavily_api_key = os.getenv("TAVILY_API_KEY")
        # if tavily_api_key:
        #     try:
        #         tavily_search = TavilySearch(
        #             tavily_api_key=tavily_api_key,
        #             max_results=3,
        #             search_depth="advanced",
        #             include_answer=True,
        #             include_raw_content=True,
        #         )
        #         web_search_tools.append(tavily_search)
        #     except Exception as e:
        #         print(f"警告: Tavily Search 初始化失败: {str(e)}")

        # 如果没有可用的搜索工具，抛出错误
        if not web_search_tools:
            raise ValueError("需要设置 BRAVE_SEARCH_API_KEY 环境变量以启用网络搜索功能")

        # 统一的 docs 基准目录（用于文件相关工具的相对路径解析）
        project_root = _find_project_root()
        docs_base_dir = os.path.join(project_root, "docs")
        os.makedirs(docs_base_dir, exist_ok=True)

        def _resolve_in_docs(path: str) -> str:
            """Resolve relative paths under the project's docs directory.

            Absolute paths are returned unchanged. Empty or "." paths resolve
            to the docs base directory.
            """
            if not path or path == ".":
                return docs_base_dir
            if os.path.isabs(path):
                return path
            return os.path.join(docs_base_dir, path)

        # 文件操作工具（默认相对路径均解析到 docs/ 下）
        def read_file_func(file_path: str) -> str:
            """读取文件内容。"""
            try:
                file_path = _resolve_in_docs(file_path)
                with open(file_path, encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"读取文件失败: {e!s}"

        def write_file_func(file_path: str, content: str) -> str:
            """写入文件内容。"""
            try:
                file_path = _resolve_in_docs(file_path)
                dir_path = os.path.dirname(file_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"成功写入文件: {file_path}"
            except Exception as e:
                return f"写入文件失败: {e!s}"

        class WriteFileInput(BaseModel):
            file_path: str = Field(..., description="目标文件路径")
            content: str = Field(..., description="要写入的文本内容")

        def list_directory_func(directory_path: str) -> str:
            """列出目录内容（相对路径解析到 docs/）。"""
            try:
                directory_path = _resolve_in_docs(directory_path)
                if not os.path.exists(directory_path):
                    return f"目录不存在: {directory_path}"
                if not os.path.isdir(directory_path):
                    return f"路径不是目录: {directory_path}"
                items = sorted([e for e in os.listdir(directory_path)])
                return f"目录 {directory_path} 内容:\n" + "\n".join(items)
            except Exception as e:
                return f"列出目录失败: {e!s}"

        # 天气查询工具
        def get_weather_func(city: str) -> str:
            """获取指定城市的当前天气信息。"""
            # 使用 WeatherAPI 的 API 来获取天气信息
            api_key = os.getenv("WEATHER_API_KEY")
            if not api_key:
                return "错误：未设置 WEATHER_API_KEY 环境变量，请在 .env 文件中配置"

            base_url = "http://api.weatherapi.com/v1/current.json"
            params = {
                'key': api_key,
                'q': city,
                'aqi': 'no'  # 不需要空气质量数据
            }

            try:
                # 调用天气 API
                response = requests.get(base_url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    # 拿到天气和温度
                    weather = data['current']['condition']['text']
                    temperature = data['current']['temp_c']
                    humidity = data['current']['humidity']
                    wind_kph = data['current']['wind_kph']
                    return f"城市 {city} 当前天气：{weather}，温度 {temperature}°C，湿度 {humidity}%，风速 {wind_kph} km/h"
                else:
                    return f"无法检索 {city} 的天气信息，API返回状态码: {response.status_code}"

            except requests.exceptions.RequestException as e:
                return f"获取 {city} 天气信息时网络请求失败: {e!s}"
            except Exception as e:
                return f"获取 {city} 天气信息时发生错误: {e!s}"

        # 货币转换（ExchangeRate-API）
        def currency_convert_func(from_currency: str, to_currency: str, amount: float) -> str:
            """使用 ExchangeRate-API 将金额从一种货币转换为另一种货币。"""
            api_key = os.getenv("EXCHANGE_RATE_API_KEY")
            if not api_key:
                return "错误：未设置 EXCHANGE_RATE_API_KEY 环境变量，请在 .env 配置。"

            base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency.upper()}/{to_currency.upper()}/{amount}"
            try:
                resp = requests.get(base_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("result") == "success":
                    rate = data.get("conversion_rate")
                    converted = data.get("conversion_result")
                    return (
                        f"{amount} {from_currency.upper()} -> {converted} {to_currency.upper()} (rate: {rate})"
                    )
                else:
                    return f"货币转换失败：{data.get('error-type') or data}"
            except requests.RequestException as e:
                return f"货币转换网络错误：{e!s}"
            except Exception as e:
                return f"货币转换发生错误：{e!s}"

        class CurrencyConvertInput(BaseModel):
            from_currency: str = Field(..., description="源货币代码，如 CNY")
            to_currency: str = Field(..., description="目标货币代码，如 JPY")
            amount: float = Field(..., description="金额")

        # 时区查询（TimeZoneDB）
        def get_timezone_func(zone: str | None = None, lat: float | None = None, lng: float | None = None) -> str:
            """查询时区信息，支持按 IANA 区域名或经纬度。"""
            api_key = os.getenv("TIMEZONEDB_API_KEY")
            if not api_key:
                return "错误：未设置 TIMEZONEDB_API_KEY 环境变量，请在 .env 配置。"

            if zone:
                params = {
                    "key": api_key,
                    "format": "json",
                    "by": "zone",
                    "zone": zone,
                }
            elif lat is not None and lng is not None:
                params = {
                    "key": api_key,
                    "format": "json",
                    "by": "position",
                    "lat": lat,
                    "lng": lng,
                }
            else:
                return "错误：请提供 zone 或 (lat, lng)。"

            try:
                resp = requests.get("https://api.timezonedb.com/v2.1/get-time-zone", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "OK":
                    zone_name = data.get("zoneName")
                    gmt_offset = data.get("gmtOffset")
                    abbrev = data.get("abbreviation")
                    local_time = data.get("formatted") or data.get("timestamp")
                    return f"时区: {zone_name}, 偏移: {gmt_offset}, 简写: {abbrev}, 本地时间: {local_time}"
                return f"时区查询失败：{data.get('message') or data}"
            except requests.RequestException as e:
                return f"时区查询网络错误：{e!s}"
            except Exception as e:
                return f"时区查询发生错误：{e!s}"

        class TimezoneInput(BaseModel):
            zone: str | None = Field(None, description="IANA 区域名，如 Asia/Tokyo")
            lat: float | None = Field(None, description="纬度，与 lng 联合使用")
            lng: float | None = Field(None, description="经度，与 lat 联合使用")

        # 公共假期（Nager.Date）
        def get_public_holidays_func(country_code: str, year: int) -> str:
            """查询指定国家与年份的公共假期（Nager.Date）。"""
            base_url = os.getenv("NAGER_DATE_BASE_URL", "https://date.nager.at")
            url = f"{base_url}/api/v3/PublicHolidays/{year}/{country_code.upper()}"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    return f"返回格式异常：{data}"
                # 简要汇总
                items = [f"{item.get('date')} - {item.get('localName')} ({item.get('name')})" for item in data[:10]]
                more = "" if len(data) <= 10 else f"，其余 {len(data) - 10} 条省略"
                return "公共假期：\n" + "\n".join(items) + more
            except requests.RequestException as e:
                return f"公共假期查询网络错误：{e!s}"
            except Exception as e:
                return f"公共假期查询发生错误：{e!s}"

        class PublicHolidaysInput(BaseModel):
            country_code: str = Field(..., description="国家代码，如 JP/CN")
            year: int = Field(..., description="年份，如 2025")

        # 数据处理工具
        def parse_json_func(json_string: str) -> str:
            """解析 JSON 字符串。"""
            try:
                parsed = json.loads(json_string)
                return f"JSON解析成功: {json.dumps(parsed, ensure_ascii=False, indent=2)}"
            except Exception as e:
                return f"JSON解析失败: {e!s}"

        # 定义工具包
        self.tool_bundles = {
            "web_research": {
                "tools": web_search_tools,
                "description": "通过 Brave Search 或 Tavily 搜索网络并返回结构化摘要",
                "use_cases": ["实时信息搜索", "背景调研", "事实验证", "资料收集"]
            },

            "file_operations": {
            "tools": [
                    Tool(
                        name="read_file",
                        func=read_file_func,
                        description="读取指定路径的文件内容",
                    ),
                    StructuredTool.from_function(
                        name="write_file",
                        func=write_file_func,
                        args_schema=WriteFileInput,
                        description="将内容写入指定路径的文件"
                    ),
                    Tool(
                        name="list_directory",
                        func=list_directory_func,
                        description="列出指定目录的所有文件和子目录",
                    )
                ],
                "description": "在工作目录内读取/写入文件并管理目录",
                "use_cases": ["查看/保存文档", "生成报告", "整理工作区", "辅助代码/笔记编辑"]
            },

            "weather_services": {
                "tools": [
                    Tool(
                        name="get_weather",
                        func=get_weather_func,
                        description="获取指定城市的实时天气信息，包括温度、湿度、风速等",
                    )
                ],
                "description": "使用 WeatherAPI 查询指定城市的实时天气",
                "use_cases": ["旅行/活动规划", "行程备忘", "简单气候分析"]
            },

            "data_processing": {
                "tools": [
                    Tool(
                        name="parse_json",
                        func=parse_json_func,
                        description="解析和格式化JSON格式的字符串数据",
                    )
                ],
                "description": "解析 JSON 字符串并返回格式化结果",
                "use_cases": ["解析 API 响应", "数据清洗", "快速格式校验"]
            }
            ,
            "travel_services": {
                "tools": [
                    StructuredTool.from_function(
                        name="currency_convert",
                        func=currency_convert_func,
                        args_schema=CurrencyConvertInput,
                        description="货币转换：from_currency, to_currency, amount。优先使用此工具进行汇率换算，避免通过网络搜索获取汇率数据。"
                    ),
                    StructuredTool.from_function(
                        name="get_timezone",
                        func=get_timezone_func,
                        args_schema=TimezoneInput,
                        description="查询时区：提供 zone 或 (lat,lng)"
                    ),
                    StructuredTool.from_function(
                        name="get_public_holidays",
                        func=get_public_holidays_func,
                        args_schema=PublicHolidaysInput,
                        description="查询公共假期：country_code, year"
                    ),
                ],
                "description": "旅行辅助：汇率换算、时区确认、节假日查询",
                "use_cases": ["旅行行程制定", "跨国会议安排", "预算换算", "节假日提醒"]
            }
        }

    async def create_agent(self, task_spec: TaskSpecification) -> "DynamicActor":
        """
        创建针对特定任务需求定制的智能体。

        Args:
            task_spec: 包含需求与上下文的任务规格

        Returns:
            配置完整的 DynamicActor 实例
        """
        try:
            # 1. 分析任务需求（LLM 驱动）
            task_analysis = await self._analyze_task_requirements(task_spec)

            # 2. 选择工具包（LangChain Tools）
            selected_tools = self._select_tool_bundles(task_analysis)

            # 3. 转换为 Function Calling 格式
            functions = self._tools_to_functions(selected_tools)

            # 4. 生成相关知识
            knowledge = await self._retrieve_knowledge(task_spec, task_analysis)

            # 5. 生成人格设定
            persona = await self._generate_persona(task_spec, task_analysis)

            # 6. 组装系统提示
            system_prompt = self._compose_prompt_for_functions(
                persona=persona,
                knowledge=knowledge,
                environment=self._get_environment(),
                selected_tools=selected_tools
            )

            # 7. 记录工具推荐（工具闭环）
            planner_suggested_tools = task_spec.context.get("required_tools", [])
            self.record_tool_recommendation(
                task_id=task_spec.task_id,
                recommended_tools=task_analysis.recommended_tools,
                planner_suggested=planner_suggested_tools
            )

            # 8. 构建配置
            config = ActorConfiguration(
                actor_id=f"actor_{uuid.uuid4().hex[:8]}",
                task_id=task_spec.task_id,
                persona=persona,
                tools=[tool.name for tool in selected_tools],
                knowledge=knowledge,
                system_prompt=system_prompt,
                execution_config=self._build_execution_config(task_analysis),
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "task_analysis": task_analysis.dict(),
                    "selected_bundles": task_analysis.recommended_tools,
                    "planner_suggested": planner_suggested_tools
                }
            )

            # 8. 创建 DynamicActor 实例
            from .dynamic_actor import DynamicActor
            return DynamicActor(
                actor_id=config.actor_id,
                task_id=config.task_id,
                task_description=task_spec.description,
                llm_client=self.llm,
                tools=selected_tools,
                functions=functions,
                system_prompt=config.system_prompt,
                config=config
            )

        except Exception as e:
            # 降级策略：创建基础配置
            return self._create_fallback_agent(task_spec, str(e))

    async def _analyze_task_requirements(self, task_spec: TaskSpecification, context_info: dict = None) -> TaskAnalysis:
        """使用LLM分析任务需求，支持上下文增强."""

        # 构建增强的上下文信息
        context_section = ""
        if context_info:
            current_progress = context_info.get("current_progress", "")
            failure_history = context_info.get("failure_history", [])
            tool_usage_stats = context_info.get("tool_usage_stats", {})

            context_section = f"""
        
        执行上下文增强：
        当前进度：{current_progress}
        历史失败：{'; '.join(failure_history) if failure_history else "无"}
        工具使用统计：{json.dumps(tool_usage_stats, ensure_ascii=False) if tool_usage_stats else "无"}"""

        analysis_prompt = f"""
        分析以下任务的需求：

        任务描述：{task_spec.description}
        父级目标：{task_spec.parent_goal or "无"}
        已知约束：{', '.join(task_spec.constraints) if task_spec.constraints else "无"}
        执行上下文：{json.dumps(task_spec.context, ensure_ascii=False)}{context_section}

        请分析：
        1. 任务类型和领域
        2. 所需专业能力
        3. 可能的挑战和风险
        4. 推荐的工具类型
        5. 需要的专业知识

        可用工具包详情：
        - web_research: 网络搜索(Brave/Tavily)、网页内容提取、信息收集与总结
        - file_operations: 文件读写、目录列表、文档创建与管理
        - data_processing: JSON解析、数据转换、结构化处理
        - weather_services: 实时天气查询、气象信息获取
        - travel_services: 货币转换、时区查询、公共节假日信息（涉及货币/汇率请优先推荐该包）
        - code_execution: Python代码执行、脚本运行、计算任务处理

        重要指引：
        - 若任务涉及货币换算/汇率计算，请优先推荐 travel_services（如 currency_convert），不要将网络搜索作为首选方案。

        返回JSON格式：
        {{
            "task_type": "research|analysis|creation|communication",
            "domain": "travel|coding|data|general|business",
            "required_capabilities": ["web_search", "file_write"],
            "complexity": "low|medium|high",
            "key_challenges": ["时间限制", "信息不足"],
            "recommended_tools": ["web_research", "file_operations"],
            "knowledge_areas": ["相关专业领域"]
        }}
        """

        try:
            response = await self.llm.complete(analysis_prompt)
            analysis_data = json.loads(response)
            return TaskAnalysis(**analysis_data)
        except Exception:
            # 降级策略：使用简化LLM分析
            return await self._fallback_task_analysis(task_spec)

    def _select_tool_bundles(self, task_analysis: TaskAnalysis) -> list[Tool]:
        """基于分析结果选择工具包."""

        selected_tools = []

        # 根据推荐工具包选择
        for bundle_name in task_analysis.recommended_tools:
            if bundle_name in self.tool_bundles:
                bundle = self.tool_bundles[bundle_name]
                selected_tools.extend(bundle["tools"])

        # 始终补充旅行相关工具，避免策略与能力清单不一致
        if "travel_services" in self.tool_bundles:
            selected_tools.extend(self.tool_bundles["travel_services"]["tools"])

        # 如果没有选择任何工具，提供默认工具
        if not selected_tools:
            selected_tools.extend(self.tool_bundles["web_research"]["tools"])
            # 同时提供旅行相关工具，避免单纯依赖搜索导致的误选
            if "travel_services" in self.tool_bundles:
                selected_tools.extend(self.tool_bundles["travel_services"]["tools"])
            selected_tools.extend(self.tool_bundles["file_operations"]["tools"][:2])  # 只取前2个

        # 去重：按工具名称去重，保持添加顺序
        unique_tools: dict[str, Tool] = {}
        for tool in selected_tools:
            try:
                tool_name = tool.name  # type: ignore[attr-defined]
            except Exception:
                tool_name = str(tool)
            if tool_name not in unique_tools:
                unique_tools[tool_name] = tool

        return list(unique_tools.values())

    def _tools_to_functions(self, tools: list[Tool]) -> list[dict]:
        """将 LangChain 工具转换为 OpenAI Function Calling 格式."""

        functions = []
        for tool in tools:
            schema = self._build_function_schema(tool)
            functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                }
            )

        return functions

    def _build_function_schema(self, tool: Tool) -> dict[str, Any]:
        """Build JSON schema for a LangChain tool."""

        # StructuredTool already exposes args_schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                schema = tool.args_schema.schema()
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            except Exception:
                pass  # fallback to signature-based schema

        signature = inspect.signature(tool.func)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in signature.parameters.items():
            if name.startswith("_"):
                continue

            annotation = param.annotation if param.annotation is not inspect._empty else str
            properties[name] = self._annotation_to_schema(annotation, name)

            if param.default is inspect._empty:
                required.append(name)

        # additionalProperties 默认允许，为保持严格性可设为 False
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return schema

    def _annotation_to_schema(self, annotation: Any, param_name: str) -> dict[str, Any]:
        """Convert Python type annotation to JSON schema fragment."""

        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())

        if annotation in (str, inspect._empty):
            return {"type": "string", "description": f"Parameter {param_name}"}
        if annotation in (int,):
            return {"type": "integer", "description": f"Parameter {param_name}"}
        if annotation in (float,):
            return {"type": "number", "description": f"Parameter {param_name}"}
        if annotation in (bool,):
            return {"type": "boolean", "description": f"Parameter {param_name}"}

        if origin is list or origin is list:
            item_schema = (
                self._annotation_to_schema(args[0], f"{param_name}_item") if args else {"type": "string"}
            )
            return {
                "type": "array",
                "items": item_schema,
                "description": f"Parameter {param_name}",
            }

        if origin is Union:
            schemas = [self._annotation_to_schema(arg, param_name) for arg in args if arg is not type(None)]
            if schemas:
                return {
                    "anyOf": schemas,
                    "description": f"Parameter {param_name}",
                }

        # default fallback
        return {"type": "string", "description": f"Parameter {param_name}"}

    async def _retrieve_knowledge(self, task_spec: TaskSpecification, task_analysis: TaskAnalysis) -> str:
        """检索和生成相关知识."""

        knowledge_prompt = f"""
        为以下任务生成相关专业知识：

        任务：{task_spec.description}
        领域：{task_analysis.domain}
        知识领域：{', '.join(task_analysis.knowledge_areas)}

        请提供：
        1. 关键概念和原则（100字以内）
        2. 最佳实践建议（100字以内）
        3. 常见陷阱提醒（100字以内）

        保持简洁实用，直接可操作。
        """

        try:
            knowledge = await self.llm.complete(knowledge_prompt)
            return knowledge
        except Exception:
            # 降级策略：提供通用知识
            return self._get_default_knowledge(task_analysis.domain)

    async def _generate_persona(self, task_spec: TaskSpecification, task_analysis: TaskAnalysis) -> str:
        """生成专业人格设定."""

        persona_prompt = f"""
        基于任务需求生成专业人格设定：

        任务：{task_spec.description}
        领域：{task_analysis.domain}
        任务类型：{task_analysis.task_type}

        生成一个专业、友好、有帮助的人格设定（50字以内）：
        例如："您好！我是专业的旅行规划助手，擅长制定个性化的旅行方案。"
        """

        try:
            persona = await self.llm.complete(persona_prompt)
            return persona.strip()
        except Exception:
            # 降级策略：基于领域的默认人格
            return self._get_default_persona(task_analysis.domain)

    def _compose_prompt_for_functions(self, persona: str, knowledge: str, environment: str, selected_tools: list = None) -> str:
        """Compose system prompt optimized for Function Calling mode."""

        # 构建工具能力描述
        tools_description = ""
        if selected_tools:
            # 仅当实际包含 currency_convert 时，才加入对应策略行
            has_currency_convert = any(
                getattr(tool, "name", "") == "currency_convert" for tool in selected_tools
            )

            currency_strategy_line = (
                "\n- 涉及货币换算/汇率的任务，请优先调用 travel_services.currency_convert，不要将网络搜索作为首选（除非工具不可用或失败）"
                if has_currency_convert
                else ""
            )

            tools_description = f"""

可用工具能力：
{self._format_selected_tools_description(selected_tools)}

工具使用策略：
- 优先选择最适合的工具类型，避免过度依赖单一工具
- 如果连续两次搜索无新信息，请总结现有结果或请求重新规划
- 遇到足够信息时可提前总结，无需持续搜索直到超时
- 文件操作、数据处理等工具同样重要，根据任务性质灵活选择{currency_strategy_line}"""

        return f"""{persona}

相关知识：
{knowledge}

环境信息：
{environment}{tools_description}

工作方式：
1. 仔细分析任务需求，制定清晰的执行计划
2. 使用提供的工具函数高效完成任务
3. 每完成关键步骤后主动报告进度
4. 遇到问题时尝试其他方法或寻求帮助
5. 完成后提供结构化的结果报告

注意事项：
- 工具调用将通过 Function Calling 自动处理
- 专注于任务逻辑和用户体验
- 保持专业、准确、有帮助的态度
- 及时反馈执行状态和遇到的问题
"""

    def _get_environment(self) -> str:
        """获取环境信息."""
        return f"""当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
系统版本：Mini-Aime v0.1.0
执行环境：Python 异步环境
可用资源：网络访问、文件系统、数据处理"""

    def _build_execution_config(self, task_analysis: TaskAnalysis) -> dict[str, Any]:
        """构建执行配置."""

        # 根据复杂度调整参数
        complexity_configs = {
            "low": {"max_iterations": 5, "timeout": 150},
            "medium": {"max_iterations": 10, "timeout": 180},
            "high": {"max_iterations": 20, "timeout": 300}
        }

        config = complexity_configs.get(task_analysis.complexity, complexity_configs["medium"])

        config.update({
            "enable_progress_reporting": True,
            "auto_retry_on_error": True,
            "max_retries": 1,
            "log_level": "INFO"
        })

        return config

    def _format_selected_tools_description(self, selected_tools: list) -> str:
        """格式化选中工具的描述。"""
        if not selected_tools:
            return "无特定工具"

        descriptions = []
        for tool in selected_tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                descriptions.append(f"- {tool.name}: {tool.description}")
            else:
                descriptions.append(f"- {tool!s}: 专用工具")

        return "\n".join(descriptions)

    def record_tool_recommendation(self, task_id: str, recommended_tools: list[str], planner_suggested: list[str] = None):
        """记录工具推荐，用于工具闭环反馈。"""
        recommendation_record = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "recommended_tools": recommended_tools,
            "planner_suggested": planner_suggested or [],
            "gap_analysis": list(set(recommended_tools) - set(planner_suggested or []))
        }
        self.tool_recommendations_history.append(recommendation_record)

        # 更新工具使用统计
        for tool in recommended_tools:
            self.tool_usage_stats[tool] = self.tool_usage_stats.get(tool, 0) + 1

    def get_tool_feedback_for_planner(self) -> dict:
        """为 Planner 提供工具使用反馈，形成闭环。"""
        if not self.tool_recommendations_history:
            return {}

        recent_recommendations = self.tool_recommendations_history[-5:]  # 最近5次

        # 分析工具使用模式
        tool_frequency = {}
        gap_patterns = []

        for record in recent_recommendations:
            for tool in record["recommended_tools"]:
                tool_frequency[tool] = tool_frequency.get(tool, 0) + 1

            if record["gap_analysis"]:
                gap_patterns.extend(record["gap_analysis"])

        return {
            "most_used_tools": sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True)[:3],
            "underused_capabilities": list(set(gap_patterns)),
            "tool_diversity_score": len(tool_frequency) / max(len(self.tool_bundles), 1),
            "recommendations": self._generate_planner_recommendations(tool_frequency, gap_patterns)
        }

    def _generate_planner_recommendations(self, tool_frequency: dict, gap_patterns: list) -> list[str]:
        """生成给 Planner 的工具使用建议。"""
        recommendations = []

        # 检查工具使用多样性
        if len(tool_frequency) <= 2:
            recommendations.append("建议任务拆分时考虑更多样化的工具类型")

        # 检查过度依赖某个工具
        if tool_frequency:
            max_usage = max(tool_frequency.values())
            total_usage = sum(tool_frequency.values())
            if max_usage / total_usage > 0.7:
                recommendations.append("避免过度依赖单一工具类型，尝试组合使用")

        # 检查未充分利用的工具
        if gap_patterns:
            underused = list(set(gap_patterns))
            recommendations.append(f"考虑更多使用这些工具类型：{', '.join(underused)}")

        return recommendations

    async def _fallback_task_analysis(self, task_spec: TaskSpecification) -> TaskAnalysis:
        """使用简化LLM提示的降级任务分析策略."""

        # 即使在降级模式下，也要使用LLM进行分析
        simplified_prompt = f"""
        快速分析任务需求（降级模式）：
        
        任务：{task_spec.description}
        
        请快速判断：
        1. 这个任务主要是什么类型？(research/creation/analysis/communication)
        2. 需要什么工具？(web_research/file_operations/data_processing/communication)
        3. 属于什么领域？(general/travel/coding/data/business)
        
        返回简单JSON：
        {{
            "task_type": "research",
            "domain": "general", 
            "recommended_tools": ["web_research"],
            "complexity": "medium"
        }}
        """

        try:
            # 使用更简单的LLM调用
            response = await self.llm.complete(simplified_prompt)
            fallback_data = json.loads(response)

            return TaskAnalysis(
                task_type=fallback_data.get("task_type", "general"),
                domain=fallback_data.get("domain", "general"),
                required_capabilities=fallback_data.get("recommended_tools", ["web_research"]),
                complexity=fallback_data.get("complexity", "medium"),
                key_challenges=["降级分析：需要仔细验证需求"],
                recommended_tools=fallback_data.get("recommended_tools", ["web_research"]),
                knowledge_areas=[fallback_data.get("domain", "general")]
            )

        except Exception:
            # 最终降级：返回最保守的配置，但不使用关键词匹配
            return TaskAnalysis(
                task_type="general",
                domain="general",
                required_capabilities=["web_research", "file_operations"],
                complexity="medium",
                key_challenges=["无法分析任务需求，使用通用配置"],
                recommended_tools=["web_research", "file_operations"],
                knowledge_areas=["general"]
            )

    def _get_default_knowledge(self, domain: str) -> str:
        """获取默认知识."""

        knowledge_map = {
            "travel": "旅行规划要点：研究目的地、制定预算、安排行程、预订住宿。注意安全和当地文化。",
            "coding": "编程最佳实践：编写清晰代码、充分测试、遵循规范、及时重构。注意错误处理和性能优化。",
            "data": "数据处理原则：验证数据质量、选择合适工具、保证数据安全、记录处理过程。",
            "business": "商业分析要点：明确目标、收集信息、分析趋势、提出建议。保持客观和专业。",
            "general": "通用工作原则：理解需求、制定计划、分步执行、及时反馈。保持专业和高效。"
        }

        return knowledge_map.get(domain, knowledge_map["general"])

    def _get_default_persona(self, domain: str) -> str:
        """获取默认人格设定."""

        persona_map = {
            "travel": "您好！我是专业的旅行规划助手，擅长制定个性化旅行方案。",
            "coding": "您好！我是经验丰富的编程助手，专注于高质量代码开发。",
            "data": "您好！我是数据分析专家，擅长处理和分析各类数据。",
            "business": "您好！我是商业分析顾问，专注于提供专业的商业洞察。",
            "research": "您好！我是研究助手，擅长信息搜集和分析整理。",
            "general": "您好！我是智能助手，致力于高效完成各类任务。"
        }

        return persona_map.get(domain, persona_map["general"])

    def _create_fallback_agent(self, task_spec: TaskSpecification, error_msg: str) -> "DynamicActor":
        """创建降级智能体."""

        # 使用最基础的配置
        basic_tools = [self.tool_bundles["web_research"]["tools"][0]]  # 只用搜索工具
        basic_functions = self._tools_to_functions(basic_tools)

        fallback_config = ActorConfiguration(
            actor_id=f"fallback_{uuid.uuid4().hex[:8]}",
            task_id=task_spec.task_id,
            persona="您好！我是基础智能助手，会尽力完成任务。",
            tools=[tool.name for tool in basic_tools],
            knowledge="基础工作原则：理解需求、尝试解决、及时反馈。",
            system_prompt="您是一个基础智能助手。请使用可用工具尽力完成用户的任务。",
            execution_config={"max_iterations": 5, "timeout": 60},
            metadata={"fallback_reason": error_msg, "created_at": datetime.now().isoformat()}
        )

        # TODO: 实现 DynamicActor 后替换
        from .dynamic_actor import DynamicActor
        return DynamicActor(
            actor_id=fallback_config.actor_id,
            task_id=fallback_config.task_id,
            task_description=task_spec.description,
            llm_client=self.llm,
            tools=basic_tools,
            functions=basic_functions,
            system_prompt=fallback_config.system_prompt,
            config=fallback_config
        )
