"""基于 LangChain 工具的动态智能体工厂。"""

import inspect
import json
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Union


if TYPE_CHECKING:
    from .dynamic_actor import DynamicActor

import requests
from langchain_core.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

from src.tools.file_tools import _find_project_root

from ..llm.base import BaseLLMClient
from ..tools.web_tools import BraveSearchTool


logger = logging.getLogger(__name__)


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
                "web_research": {"tools": []},
                "file_operations": {"tools": []},
                "weather_services": {"tools": []},
                "data_processing": {"tools": []},
                "travel_services": {"tools": []},
                "research_integration": {"tools": []},
            }

    def _initialize_tool_bundles(self):
        """初始化工具包。"""

        # 网络搜索工具 - 优先使用 Brave Search，回退到 Tavily
        web_search_tools = []

        # 尝试初始化 Brave Search
        brave_api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if brave_api_key:
            try:
                brave_search_tool = BraveSearchTool(api_key=brave_api_key, max_results=5)

                # 创建 LangChain 工具包装器
                def brave_search_func(query: str) -> str:
                    """使用 Brave Search 进行网络搜索，返回更详细的结果用于深度研究。"""
                    try:
                        result = brave_search_tool.execute_sync(query)
                        return brave_search_tool.format_results(result, max_length=3000)
                    except Exception as e:
                        return f"Brave 搜索失败: {e!s}"

                brave_search_langchain = Tool(
                    name="brave_search",
                    func=brave_search_func,
                    description="使用 Brave Search API 进行网络搜索，返回格式化的搜索结果",
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

        # 如果没有可用的搜索工具，只是警告
        if not web_search_tools:
            print("警告: 未设置 BRAVE_SEARCH_API_KEY，网络搜索功能将不可用")

        # 统一的 docs 基准目录（用于文件相关工具的相对路径解析）
        # 使用 session 隔离：每个 session 的文件保存在独立目录中
        project_root = _find_project_root()

        # 注意：这里我们暂时使用 docs/ 作为基准，实际的 session_id 会在 Agent 创建时动态传递
        # 但为了简化，我们先保持现有逻辑，因为修改需要传递 session_id 到这里
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
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                # 文件不存在时，返回特殊标记，让智能体知道需要等待
                return "FILE_NOT_FOUND: 文件不存在，请稍后重试"
            except Exception as e:
                return f"读取文件失败: {e!s}"

        class ReadFilesInput(BaseModel):
            file_paths: list[str] = Field(
                ...,
                description="要读取的文件路径列表",
            )

        def read_files_func(file_paths: list[str]) -> str:
            """批量读取多个文件并返回内容映射。"""
            if not isinstance(file_paths, list) or not file_paths:
                return "读取失败：请提供非空的文件路径列表"

            results: dict[str, str] = {}
            errors: dict[str, str] = {}

            for path in file_paths:
                resolved = _resolve_in_docs(path)
                try:
                    with open(resolved, encoding="utf-8") as f:
                        results[path] = f.read()
                except Exception as exc:
                    errors[path] = str(exc)

            payload = {
                "results": results,
                "errors": errors,
            }

            return json.dumps(payload, ensure_ascii=False)

        def write_file_func(file_path: str, content: str) -> str:
            """
            写入文件内容到 docs/ 目录。

            **路径规则**：
            - 相对路径会自动解析到 docs/ 目录下
            - 例如：写 "report.md" → 实际保存到 "docs/report.md"
            - 例如：写 "task_123/report.md" → 实际保存到 "docs/task_123/report.md"
            - ⚠️ 不要在路径前加 "docs/"，否则会变成 "docs/docs/"
            """
            try:
                file_path = _resolve_in_docs(file_path)
                dir_path = os.path.dirname(file_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"成功写入文件: {file_path}"
            except Exception as e:
                return f"写入文件失败: {e!s}"

        class WriteFilesInput(BaseModel):
            files: dict[str, str] = Field(
                ...,
                description="键为文件路径（相对路径，会自动写入 docs/），值为要写入的内容",
            )

        def write_files_func(files: dict[str, str]) -> str:
            """批量写入多个文件。"""
            if not isinstance(files, dict) or not files:
                return "写入失败：请提供至少一个文件"

            successes: list[str] = []
            errors: dict[str, str] = {}

            for path, content in files.items():
                resolved = _resolve_in_docs(path)
                try:
                    directory = os.path.dirname(resolved)
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    with open(resolved, "w", encoding="utf-8") as f:
                        f.write(content)
                    successes.append(path)
                except Exception as exc:
                    errors[path] = str(exc)

            payload = {
                "written": successes,
                "errors": errors,
            }

            return json.dumps(payload, ensure_ascii=False)

        class WriteFileInput(BaseModel):
            file_path: str = Field(
                ...,
                description="目标文件路径（相对路径，会自动保存到 docs/ 目录下，不要加 docs/ 前缀）",
            )
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
                "key": api_key,
                "q": city,
                "aqi": "no",  # 不需要空气质量数据
            }

            try:
                # 调用天气 API
                response = requests.get(base_url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    # 拿到天气和温度
                    weather = data["current"]["condition"]["text"]
                    temperature = data["current"]["temp_c"]
                    humidity = data["current"]["humidity"]
                    wind_kph = data["current"]["wind_kph"]
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
                    return f"{amount} {from_currency.upper()} -> {converted} {to_currency.upper()} (rate: {rate})"
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
        def get_timezone_func(
            zone: str | None = None, lat: float | None = None, lng: float | None = None
        ) -> str:
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
                resp = requests.get(
                    "https://api.timezonedb.com/v2.1/get-time-zone", params=params, timeout=10
                )
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
                items = [
                    f"{item.get('date')} - {item.get('localName')} ({item.get('name')})"
                    for item in data[:10]
                ]
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

        # 研究整合工具
        # 定义工具包
        self.tool_bundles = {
            "web_research": {
                "tools": web_search_tools,
                "description": "通过 Brave Search 或 Tavily 搜索网络并返回结构化摘要",
                "use_cases": ["实时信息搜索", "背景调研", "事实验证", "资料收集"],
            },
            "file_operations": {
                "tools": [
                    Tool(
                        name="read_file",
                        func=read_file_func,
                        description="读取指定路径的文件内容",
                    ),
                    StructuredTool.from_function(
                        name="read_files",
                        func=read_files_func,
                        args_schema=ReadFilesInput,
                        description="批量读取多个文件，返回内容和错误信息的 JSON 结构",
                    ),
                    StructuredTool.from_function(
                        name="write_file",
                        func=write_file_func,
                        args_schema=WriteFileInput,
                        description="将内容写入指定路径的文件",
                    ),
                    StructuredTool.from_function(
                        name="write_files",
                        func=write_files_func,
                        args_schema=WriteFilesInput,
                        description="批量写入多个文件，一次完成多份内容的保存",
                    ),
                    Tool(
                        name="list_directory",
                        func=list_directory_func,
                        description="列出指定目录的所有文件和子目录",
                    ),
                ],
                "description": "在工作目录内读取/写入文件并管理目录",
                "use_cases": ["查看/保存文档", "生成报告", "整理工作区", "辅助代码/笔记编辑"],
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
                "use_cases": ["活动规划", "行程备忘", "简单气候分析"],
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
                "use_cases": ["解析 API 响应", "数据清洗", "快速格式校验"],
            },
            "travel_services": {
                "tools": [
                    StructuredTool.from_function(
                        name="currency_convert",
                        func=currency_convert_func,
                        args_schema=CurrencyConvertInput,
                        description="货币转换：from_currency, to_currency, amount。优先使用此工具进行汇率换算，避免通过网络搜索获取汇率数据。",
                    ),
                    StructuredTool.from_function(
                        name="get_timezone",
                        func=get_timezone_func,
                        args_schema=TimezoneInput,
                        description="查询时区：提供 zone 或 (lat,lng)",
                    ),
                    StructuredTool.from_function(
                        name="get_public_holidays",
                        func=get_public_holidays_func,
                        args_schema=PublicHolidaysInput,
                        description="查询公共假期：country_code, year",
                    ),
                ],
                "description": "通用辅助：汇率换算、时区确认、节假日查询",
                "use_cases": ["行程制定", "跨国会议安排", "预算换算", "节假日提醒"],
            },
            "research_integration": {
                "tools": [
                    StructuredTool.from_function(
                        func=self._create_research_integration_func(),
                        name="integrate_research",
                        description="智能发现和整合任务目录下的所有研究文件，生成综合报告。用于最终报告生成任务。",
                    )
                ],
                "description": "研究整合：自动发现、分析和整合子任务的研究成果",
                "use_cases": ["最终报告生成", "研究数据整合", "多源信息汇总"],
            },
        }

    def _create_research_integration_func(self):
        """创建研究整合工具函数。"""
        from ..tools.research_integration_tool import ResearchIntegrationTool

        def integrate_research_func(
            task_directory: str, output_file: str, original_goal: str = ""
        ) -> str:
            """智能发现和整合任务目录下的所有研究文件，生成综合报告。"""
            tool = ResearchIntegrationTool()
            return tool.execute_sync(
                task_directory=task_directory,
                output_file=output_file,
                original_goal=original_goal,
                llm_client=self.llm,
            )

        return integrate_research_func

    async def create_agent(self, task_spec: TaskSpecification) -> "DynamicActor":
        """
        创建针对特定任务需求定制的智能体。

        Args:
            task_spec: 包含需求与上下文的任务规格

        Returns:
            配置完整的 DynamicActor 实例
        """
        try:
            # 检测是否为汇总任务
            if (
                task_spec.task_id == "task_final_summary"
                or task_spec.context.get("task_type") == "summary"
            ):
                return await self._create_summary_agent(task_spec)

            # 🚀 优化：合并 LLM 调用 - 一次性获取所有配置信息
            # 原先: 3次独立调用（分析、知识、人格）
            # 现在: 1次综合调用，节省 66% 的调用次数
            agent_design = await self._design_complete_agent(task_spec)

            task_analysis = agent_design["task_analysis"]
            persona = agent_design["persona"]
            knowledge = agent_design["knowledge"]

            # 2. 选择工具包（LangChain Tools）
            selected_tools = self._select_tool_bundles(task_analysis)

            # 3. 转换为 Function Calling 格式
            functions = self._tools_to_functions(selected_tools)

            # 6. 组装系统提示（传递 session_id 用于文件隔离规范）
            session_id = task_spec.context.get("session_id")
            system_prompt = self._compose_prompt_for_functions(
                persona=persona,
                knowledge=knowledge,
                environment=self._get_environment(),
                selected_tools=selected_tools,
                session_id=session_id,
            )

            # 7. 记录工具推荐（工具闭环）
            planner_suggested_tools = task_spec.context.get("required_tools", [])
            self.record_tool_recommendation(
                task_id=task_spec.task_id,
                recommended_tools=task_analysis.recommended_tools,
                planner_suggested=planner_suggested_tools,
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
                    "planner_suggested": planner_suggested_tools,
                },
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
                config=config,
            )

        except Exception as e:
            # 降级策略：创建基础配置
            return self._create_fallback_agent(task_spec, str(e))

    async def _design_complete_agent(self, task_spec: TaskSpecification) -> dict:
        """
        一次性设计完整的智能体配置（合并的 LLM 调用）。

        原先需要 3 次 LLM 调用：
        1. _analyze_task_requirements - 分析任务
        2. _retrieve_knowledge - 生成知识
        3. _generate_persona - 生成人格

        现在合并为 1 次调用，让 LLM 在完整上下文中综合思考，
        更智能且节省成本。
        """

        comprehensive_prompt = f"""
你是一个智能体设计专家，负责为特定任务设计最优的 AI 智能体配置。

## 任务信息
- 任务描述：{task_spec.description}
- 父级目标：{task_spec.parent_goal or "无"}
- 已知约束：{", ".join(task_spec.constraints) if task_spec.constraints else "无"}
- 执行上下文：{json.dumps(task_spec.context, ensure_ascii=False)}

## 可用工具能力
- web_research: 网络搜索(Brave/Tavily)、网页内容提取、信息收集
- file_operations: 文件读写、目录列表、文档管理
- data_processing: JSON解析、数据转换、结构化处理
- weather_services: 实时天气查询、气象信息
- travel_services: 货币转换、时区查询、节假日信息（涉及货币优先推荐）
- code_execution: Python代码执行、脚本运行、计算任务
- research_integration: 智能整合研究文件、生成综合报告（最终报告任务必须推荐）

## 你的任务
请综合分析并返回完整的智能体设计方案，包括：

1. **任务分析**：
   - 任务类型和领域
   - 所需能力和工具
   - 复杂度评估
   - 关键挑战

2. **人格设定**：
   - 设计专业、友好、有帮助的角色（50字以内）
   - 要体现该任务的专业性

3. **知识注入**：
   - 关键概念和原则（100字以内）
   - 最佳实践建议（100字以内）
   - 常见陷阱提醒（100字以内）

## 返回格式（JSON）
{{
    "task_analysis": {{
        "task_type": "research|analysis|creation|communication",
        "domain": "general|coding|data|business|education",
        "required_capabilities": ["web_search", "file_write"],
        "complexity": "low|medium|high",
        "key_challenges": ["时间限制", "信息不足"],
        "recommended_tools": ["web_research", "file_operations"],
        "knowledge_areas": ["相关专业领域"]
    }},
    "persona": "您好！我是专业的...助手，擅长...",
    "knowledge": "关键概念：...\\n最佳实践：...\\n注意事项：..."
}}

**重要**：
- 涉及货币/汇率时，recommended_tools 必须包含 travel_services
- 最终报告生成任务（包含"最终报告"、"汇总"、"整合"等关键词），recommended_tools 必须包含 research_integration
"""

        try:
            # 使用专门的JSON模式调用LLM
            response = await self.llm.complete_chat_json(
                [
                    {
                        "role": "system",
                        "content": "你是一个专业的任务分析助手，必须返回有效的JSON格式响应。",
                    },
                    {"role": "user", "content": comprehensive_prompt},
                ]
            )

            # 验证响应结构
            if not isinstance(response, dict):
                raise ValueError("LLM response is not a dictionary")

            if "task_analysis" not in response:
                raise ValueError("Missing task_analysis in response")

            # 转换 task_analysis 为 TaskAnalysis 对象
            analysis_data = response["task_analysis"]
            task_analysis = TaskAnalysis(
                task_type=analysis_data.get("task_type", "general"),
                domain=analysis_data.get("domain", "general"),
                required_capabilities=analysis_data.get("required_capabilities", []),
                complexity=analysis_data.get("complexity", "medium"),
                key_challenges=analysis_data.get("key_challenges", []),
                recommended_tools=analysis_data.get("recommended_tools", []),
                knowledge_areas=analysis_data.get("knowledge_areas", []),
            )

            return {
                "task_analysis": task_analysis,
                "persona": response.get("persona", "您好！我是智能助手。"),
                "knowledge": response.get("knowledge", ""),
            }

        except Exception as e:
            # 降级策略：使用基础配置
            logger.warning(f"Comprehensive agent design failed, using fallback: {e}")

            # 创建基础的任务分析
            fallback_analysis = TaskAnalysis(
                task_type="general",
                domain="general",
                required_capabilities=["web_research", "file_operations", "research_integration"],
                complexity="medium",
                key_challenges=["使用基础配置"],
                recommended_tools=["web_research", "file_operations", "research_integration"],
                knowledge_areas=["general"],
            )

            return {
                "task_analysis": fallback_analysis,
                "persona": "您好！我是智能助手，会尽力完成任务。",
                "knowledge": "基础工作原则：理解需求、尝试解决、及时反馈。",
            }

    def _select_tool_bundles(self, task_analysis: TaskAnalysis) -> list[Tool]:
        """基于分析结果选择工具包."""

        selected_tools = []

        # 根据推荐工具包选择
        for bundle_name in task_analysis.recommended_tools:
            if bundle_name in self.tool_bundles:
                bundle = self.tool_bundles[bundle_name]
                selected_tools.extend(bundle["tools"])

        # 始终补充通用工具，避免策略与能力清单不一致
        if "travel_services" in self.tool_bundles:
            selected_tools.extend(self.tool_bundles["travel_services"]["tools"])

        if "file_operations" in self.tool_bundles:
            core_file_tools = [
                t for t in self.tool_bundles["file_operations"]["tools"]
                if getattr(t, "name", "") in ["read_file", "write_file", "list_directory"]
            ]
            selected_tools.extend(core_file_tools)

        # 如果没有选择任何工具，提供默认工具
        if not selected_tools:
            selected_tools.extend(self.tool_bundles["web_research"]["tools"])
            # 同时提供通用工具，避免单纯依赖搜索导致的误选
            if "travel_services" in self.tool_bundles:
                selected_tools.extend(self.tool_bundles["travel_services"]["tools"])
            if "file_operations" in self.tool_bundles:
                selected_tools.extend(self.tool_bundles["file_operations"]["tools"][:3])

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
                self._annotation_to_schema(args[0], f"{param_name}_item")
                if args
                else {"type": "string"}
            )
            return {
                "type": "array",
                "items": item_schema,
                "description": f"Parameter {param_name}",
            }

        if origin is Union:
            schemas = [
                self._annotation_to_schema(arg, param_name) for arg in args if arg is not type(None)
            ]
            if schemas:
                return {
                    "anyOf": schemas,
                    "description": f"Parameter {param_name}",
                }

        # default fallback
        return {"type": "string", "description": f"Parameter {param_name}"}

    def _compose_prompt_for_functions(
        self,
        persona: str,
        knowledge: str,
        environment: str,
        selected_tools: list | None = None,
        session_id: str | None = None,
    ) -> str:
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

        # 构建文件保存规范（如果有 session_id 且包含文件操作工具）
        file_isolation_rule = ""
        if (
            session_id
            and selected_tools
            and any(
                "write_file" in getattr(tool, "name", "")
                or "read_file" in getattr(tool, "name", "")
                or "list_directory" in getattr(tool, "name", "")
                for tool in selected_tools
            )
        ):
            file_isolation_rule = f"""

**⚠️ 文件操作强制规范**（请严格遵守）：
- 当前 Session ID: `{session_id}`
- **所有文件路径必须以 `{session_id}/` 开头**
- **🚨 重要：必须使用当前Session ID，不要使用其他ID！**
- 示例：
  - ✅ 正确：write_file("{session_id}/research_report.md", content)
  - ✅ 正确：read_file("{session_id}/temp/data.json")
  - ✅ 正确：list_directory("{session_id}/temp")
  - ❌ 错误：write_file("research_report.md", ...) - 缺少 session_id 前缀
  - ❌ 错误：read_file("temp/data.json") - 缺少 session_id 前缀
  - ❌ 错误：write_file("task_20251008_173854/file.md", ...) - 使用了错误的session_id
- **这是强制要求，确保不同任务的文件完全隔离**
- **🔥 检查清单：每次调用write_file前，确认路径以 `{session_id}/` 开头！**
"""

        return f"""{persona}

相关知识：
{knowledge}

环境信息：
{environment}{tools_description}{file_isolation_rule}

工作方式：
1. 仔细分析任务需求，制定清晰的执行计划
2. 使用提供的工具函数高效完成任务
3. 遇到问题时尝试其他方法或寻求帮助
4. 完成后在 observation 中明确说明 "任务已完成" 或 "TASK_COMPLETE"

**🔥 研究任务必须生成文件！**
- 如果任务涉及研究、调查、分析、规划等内容，必须使用 `write_file` 将结果保存为文件
- 文件名应该清晰描述内容，使用 `.md` 或 `.json` 格式
- 只搜索信息但不保存文件的任务会被系统判定为失败
- **🚨 文件路径必须使用当前Session ID：`{session_id}/`**
- 示例：
  - ✅ 正确：搜索签证信息 → 使用 write_file 保存为 `{session_id}/visa_requirements.md` → 说明"任务已完成"
  - ❌ 错误：只搜索信息就说"任务已完成"，没有保存文件
  - ❌ 错误：使用错误的session_id路径

任务完成标准：
- 当你认为任务目标已达成时，在最后一步的 thought 或工具输出中包含 "任务已完成" 或 "TASK_COMPLETE"
- 对于研究类任务，必须先调用 write_file 保存研究结果，然后再说明任务完成
- 系统会自动识别这些标记并结束执行

注意事项：
- 工具调用将通过 Function Calling 自动处理
- 专注于任务逻辑和用户体验
- 保持专业、准确、有帮助的态度
- 明确表达任务完成状态，不要模糊不清
"""

    def _get_environment(self) -> str:
        """获取环境信息."""
        return f"""当前时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
系统版本：Mini-Aime v0.1.0
执行环境：Python 异步环境
可用资源：网络访问、文件系统、数据处理"""

    def _build_execution_config(self, task_analysis: TaskAnalysis) -> dict[str, Any]:
        """构建执行配置."""

        # 根据复杂度调整参数 - 优化后的配置，减少步数以降低成本
        complexity_configs = {
            "low": {"max_iterations": 3, "timeout": 120},
            "medium": {"max_iterations": 5, "timeout": 150},
            "high": {"max_iterations": 8, "timeout": 200},
        }

        config = complexity_configs.get(task_analysis.complexity, complexity_configs["medium"])

        config.update(
            {
                "enable_progress_reporting": True,
                "auto_retry_on_error": True,
                "max_retries": 1,
                "log_level": "INFO",
            }
        )

        return config

    def _format_selected_tools_description(self, selected_tools: list) -> str:
        """格式化选中工具的描述。"""
        if not selected_tools:
            return "无特定工具"

        descriptions = []
        for tool in selected_tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                descriptions.append(f"- {tool.name}: {tool.description}")
            else:
                descriptions.append(f"- {tool!s}: 专用工具")

        return "\n".join(descriptions)

    def record_tool_recommendation(
        self, task_id: str, recommended_tools: list[str], planner_suggested: list[str] | None = None
    ):
        """记录工具推荐，用于工具闭环反馈。"""
        recommendation_record = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "recommended_tools": recommended_tools,
            "planner_suggested": planner_suggested or [],
            "gap_analysis": list(set(recommended_tools) - set(planner_suggested or [])),
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
            "recommendations": self._generate_planner_recommendations(tool_frequency, gap_patterns),
        }

    def _generate_planner_recommendations(
        self, tool_frequency: dict, gap_patterns: list
    ) -> list[str]:
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

    async def _create_summary_agent(self, task_spec: TaskSpecification) -> "DynamicActor":
        """创建专门的汇总Agent，用于整合所有子任务报告。"""
        from .dynamic_actor import DynamicActor

        # 获取汇总任务的元数据
        metadata = task_spec.context
        original_goal = metadata.get("original_goal", "未知目标")
        total_subtasks = metadata.get("total_subtasks", 0)
        session_id = metadata.get("session_id", "unknown")

        # 选择必要的工具：文件操作 + 研究整合
        tools = []
        if "file_operations" in self.tool_bundles:
            tools.extend(self.tool_bundles["file_operations"]["tools"])
        if "research_integration" in self.tool_bundles:
            tools.extend(self.tool_bundles["research_integration"]["tools"])

        # 转换为 Function Calling 格式
        functions = self._tools_to_functions(tools)

        system_prompt = f"""# 你的身份与使命

你是一位**资深的内容整合专家和分析师**，拥有以下核心能力：

## 专业技能
- 📖 **深度阅读理解**：能够快速理解复杂文档的核心要点
- 🔗 **信息整合能力**：善于发现不同内容间的关联和模式
- 📊 **结构化思维**：能够将零散信息组织成清晰的逻辑结构
- 💡 **洞察力**：不仅总结事实，更能提炼深层见解
- ✍️ **专业写作**：使用清晰、专业、客观的语言表达

## 当前任务背景

### 用户原始目标
```
{original_goal}
```

### 执行情况
- ✅ 已完成 **{total_subtasks}** 个子任务
- 📁 Session ID: `{session_id}`
- 📄 每个子任务都已生成独立的详细报告

### 你的使命
将所有子任务报告整合成一份**完整、连贯、有洞察力**的最终报告，让用户能够：
1. 快速了解整体完成情况
2. 理解各任务间的关联
3. 获得综合性的见解和价值
4. 追溯详细信息和工件

---

**⚠️ 文件操作强制规范**（请严格遵守）：
- 当前 Session ID: `{session_id}`
- **所有文件路径必须以 `{session_id}/` 开头**
- 示例：
  - ✅ 正确：read_file("{session_id}/final_report_T1.md")
  - ✅ 正确：list_directory("{session_id}")
  - ✅ 正确：write_file("{session_id}/final_report.md", content)
  - ❌ 错误：read_file("final_report_T1.md") - 缺少 session_id 前缀
  - ❌ 错误：list_directory("docs") - 应该使用 "{session_id}"
- **这是强制要求，确保读取的是当前任务的文件，不会读到其他任务的文件**

---

## 工作流程

### 第1步：系统性阅读
1. 使用 `list_directory("{session_id}")` 工具列出当前 session 下的所有报告文件
2. 使用 `read_file("{session_id}/文件名")` 工具依次阅读每个子任务报告（**路径必须包含 {session_id}/**）
3. 边读边记录：
   - 每个任务的核心成果和实质性内容
   - 关键数据、发现和洞察
   - 用户需要的具体信息（不要记录系统执行过程）
4. 注意任务间的逻辑关系和内容连贯性

### 第1.5步：智能整合研究数据
**🚨 关键步骤：使用智能工具整合所有真实的研究成果！**

在读取子任务报告后，必须执行以下操作：

1. **优先使用智能整合工具**：
   - 使用 `integrate_research` 工具智能发现和整合任务目录下的所有研究文件
   - 这个工具会自动扫描 `{session_id}/` 目录，找到所有研究文件（排除 `final_report_*.md`）
   - 使用LLM分析每个文件的内容，提取关键信息和具体数据

   **具体调用示例**：
   ```
   integrate_research(
       task_directory="{session_id}",
       output_file="{session_id}/research_integration_report.md",
       original_goal="{original_goal}"
   )
   ```

2. **手动补充（如果需要）**：
   - 如果智能整合工具遗漏了某些文件，手动使用 `read_file` 读取
   - 特别关注子任务报告中"引用与工件"部分提到的文件
   - 确保获取所有原始的研究数据和详细内容

3. **验证数据完整性**：
   - 确保整合了所有子任务的研究成果
   - 检查是否包含具体数据（价格、时间、地点等）
   - 不要遗漏任何重要的研究发现

4. **生成基于真实数据的报告**：
   - 使用整合的研究数据生成综合报告
   - 确保报告包含具体的研究数据，而不是通用模板
   - 突出每个任务的实际产出和发现

**⚠️ 重要提醒**：
- 优先使用 `integrate_research` 工具进行智能整合
- 子任务报告可能只是结构化壳，真实内容在研究文件中
- 必须基于实际研究数据生成汇总报告

### 第2步：信息提取与整合
针对每个子任务报告和研究文件，提取：
- **实质性内容**：具体的数据、信息、建议
- **关键发现**：重要的发现和洞察
- **可操作信息**：用户可以直接使用的内容

同时思考：
- 如何将这些内容有机地整合在一起？
- 如何让报告结构清晰、易读？
- 用户最关心什么信息？

### 第3步：综合分析
- 📊 **整体整合**：将所有内容整合成连贯的报告
- 🔗 **逻辑组织**：按照用户的阅读习惯组织内容
- 💡 **价值提炼**：提供有价值的总结和建议
- 🎯 **用户导向**：确保所有内容对用户有实际价值

### 第4步：撰写最终报告
**🚨 这是你的核心任务！你必须完成这一步！**

使用 `write_file` 工具，**必须**将报告保存到以下路径（请严格遵守）：

**文件路径（必须使用此路径）**：
```
{session_id}/final_report.md
```

**⚠️ 关键要求**：
- 🔥 **这是你的主要目标**：生成面向最终用户的、完整的、详尽的、基于所有研究数据的综合报告
- 🔥 **不要只是读取文件**：你必须调用 `write_file` 生成报告
- 🔥 **路径**：`{session_id}/final_report.md`（不要以 `docs/` 开头）
- 🔥 **文件名**：必须是 `final_report.md`
- 🔥 **内容丰富性要求（非常重要！）**：
  - **必须详尽**：报告长度应该足够详细，充分展示所有研究成果
  - **充分利用所有数据**：每个子任务的研究结果、每次 function_calling 获取的信息都必须体现在报告中
  - **具体化**：所有价格、时间、地点、联系方式、网址、具体建议都要包含
  - **不能遗漏**：如果研究文件中提到了某个景点、某个价格、某个建议，报告中必须体现
- 🔥 **内容格式**：
  - 如果原始目标包含时间规划（如"X周""X个月""X天"计划等），必须生成按时间分阶段的详细计划表
  - 每个时间段要有具体的活动安排、预算明细、实用建议
  - 报告必须是用户友好的、可直接使用的文档，而不是技术性的执行报告
  - 确保包含具体的、可操作的建议和数据

**❌ 严重错误（绝对禁止）**：
- 只读取1-2个文件就结束 ❌❌❌
- 生成通用的模板而不包含具体研究数据 ❌❌❌
- 报告内容太简略，没有充分利用研究数据 ❌❌❌
- 遗漏了某些研究文件中的重要信息 ❌❌❌
- 生成技术性的任务执行报告（包含"任务ID"、"执行步数"等元数据）❌❌❌

**✅ 正确行为（必须遵守）**：
1. **系统性读取**：使用 `list_directory` 列出所有文件，然后用 `read_file` 逐个读取所有 .md 和 .json 文件（排除 final_report_*.md）
2. **全面提取**：提取每个文件中的所有具体数据：
   - 价格信息（机票、住宿、餐饮、活动等）
   - 时间信息（最佳时间、每日安排等）
   - 地点信息（景点、酒店、交通枢纽等）
   - 实用建议（签证流程、注意事项等）
   - 所有其他具体数据
3. **丰富整合**：将所有数据整合成详尽的报告，确保：
   - 每个研究主题都有详细阐述
   - 所有具体数据都被包含
   - 报告长度足够详细（不要太简略）
4. **保存报告**：调用 `write_file` 保存到 `{session_id}/final_report.md`

**📏 内容丰富度检查清单**：
- ✅ 是否包含了所有子任务的研究成果？
- ✅ 是否包含了所有具体的价格信息？
- ✅ 是否包含了所有具体的时间安排？
- ✅ 是否包含了所有景点和地点的详细介绍？
- ✅ 是否包含了所有实用建议和注意事项？
- ✅ 报告长度是否足够详细（而不是几段话的摘要）？

---

## 最终报告格式规范

**🎯 核心原则：用户友好、实用性强、内容详实**

报告应该是普通用户可以直接阅读和使用的文档，而不是技术性的执行报告。

### 📋 推荐格式结构

根据任务类型选择合适的格式：

#### 格式A：时间规划类（学习计划、项目计划等）

**重要提示：以下格式只是结构示意，实际报告内容必须非常详尽！**

```markdown
# {original_goal}

## 📋 概览
- **总时长**：[具体天数/周数/月数]
- **主要区域**：[列出所有涉及的地区/主题]
- **总预算**：[具体金额，包括所有明细]
- **最佳时间**：[基于研究的具体建议]
- **行程亮点**：[列出所有重要景点/活动]

## 🗓️ 第一阶段：[阶段名称]（第X-Y周/天）

### 🎯 本阶段目标
[清晰的目标说明]

### 📅 详细日程
**第X天：[地点] - [主题]**
- **上午**：[具体活动，包括地点名称、地址、开放时间]
  - 活动1：[名称、简介、门票价格、预计时间]
  - 活动2：[名称、简介、门票价格、预计时间]
- **下午**：[具体活动]
  - 活动3：[详细信息]
- **晚上**：[具体活动]
  - 餐厅推荐：[名称、特色菜、人均消费]
- **住宿**：[酒店名称、价格区间、预订建议]

**第X+1天：[地点] - [主题]**
[继续详细列出每天的安排...]

### 💰 本阶段预算明细
| 项目 | 具体内容 | 单价 | 数量 | 小计 |
|------|---------|------|------|------|
| 住宿 | [酒店类型] | ¥XXX/晚 | X晚 | ¥XXX |
| 餐饮 | [每日平均] | ¥XXX/天 | X天 | ¥XXX |
| 交通 | [具体方式] | ¥XXX | - | ¥XXX |
| 门票 | [景点名称] | ¥XXX | X人 | ¥XXX |
| **阶段小计** | | | | **¥XXX** |

### 🚗 交通安排
- **到达交通**：[具体方式、路线、价格、时长]
- **区域内交通**：[租车/公交/其他，具体费用和建议]
- **离开交通**：[具体方式、路线、价格、时长]

### 📌 重要提示
- ⚠️ [具体注意事项1]
- ⚠️ [具体注意事项2]
- 💡 [实用建议1]
- 💡 [实用建议2]

### 🏨 住宿推荐
1. **[酒店/民宿名称]**
   - 位置：[具体地址]
   - 价格：¥XXX-XXX/晚
   - 特点：[设施、优势]
   - 预订建议：[提前时间、平台]

## 🗓️ 第二阶段：[阶段名称]（第X-Y周/天）
[按照同样详细的格式继续...]

## 💵 总预算估算

### 详细费用清单
| 大类 | 明细 | 金额 |
|------|------|------|
| **国际交通** | | |
| 往返机票 | [出发地-目的地] | ¥XXX |
| **境内交通** | | |
| 租车 | [天数、车型] | ¥XXX |
| 油费 | [预估] | ¥XXX |
| 停车费 | [预估] | ¥XXX |
| **住宿** | | |
| 第一阶段 | [X晚] | ¥XXX |
| 第二阶段 | [X晚] | ¥XXX |
| **餐饮** | | |
| 日常餐饮 | [X天 × ¥XXX] | ¥XXX |
| 特色餐厅 | [X次] | ¥XXX |
| **活动门票** | | |
| [具体活动1] | | ¥XXX |
| [具体活动2] | | ¥XXX |
| **其他** | | |
| 签证 | | ¥XXX |
| 保险 | | ¥XXX |
| 应急预备金 | | ¥XXX |
| **总计** | | **¥XX,XXX** |

## 💡 实用建议

### 签证办理
[详细的签证流程、材料清单、办理时间]

### 最佳时间
[基于研究的详细分析，包括天气、节假日、价格等因素]

### 必备物品清单
- [分类列出所有建议携带的物品]

### 当地实用信息
- **紧急电话**：[具体号码]
- **常用APP**：[名称和用途]
- **文化习俗**：[需要注意的事项]

## 🔗 参考资源
- [重要网站链接和说明]
- [有用的文档和指南]
- [联系方式和咨询渠道]
```

**⚠️ 内容丰富度标准**：
- 如果是两个月计划，报告应该包含60天的详细安排
- 每天的安排要具体到活动、地点、时间、价格
- 所有价格信息必须来自研究数据，不能空缺
- 景点介绍要包含名称、特色、门票、开放时间等
- 住宿推荐要包含具体名称、价格区间、特点

#### 格式B：研究报告类（技术调研、市场分析等）

```markdown
# {original_goal} - 研究报告

## 执行摘要
[核心发现和结论的简要总结]

## 研究背景
[为什么进行这项研究，目标是什么]

## 研究发现

### 主题1：[具体主题]
[详细的研究内容和数据]

### 主题2：[具体主题]
[详细的研究内容和数据]

## 综合分析
[跨主题的分析和洞察]

## 结论与建议
[基于研究的具体建议]

## 参考资料
[来源、链接等]
```

**⚠️ 格式要求**：
- **禁止**包含：任务ID、Session ID、执行步数等技术元数据
- **必须**包含：具体的研究数据、价格、时间、地点、链接等实用信息
- **突出**可操作性：确保用户可以直接使用报告中的信息

---

### 📝 内容质量要求

**核心原则：用户价值第一**

1. **内容完整性**：
   - 必须整合所有子任务的研究成果和数据
   - 确保包含具体信息（价格、时间、地点、联系方式等）
   - 不遗漏重要的发现和建议

2. **实用性**：
   - 用户可以直接根据报告采取行动
   - 信息具体、可操作
   - 避免空洞的描述和通用模板

3. **结构清晰**：
   - 使用清晰的标题和子标题
   - 合理使用列表、表格等格式
   - 逻辑连贯，易于阅读

4. **数据准确**：
   - 所有数据来源于实际的研究文件
   - 不编造或猜测信息
   - 引用具体的数据和发现

---

## 注意事项

⚠️ **必须做**：
- 仔细阅读所有子任务报告
- 基于实际内容进行分析
- 遵循三部分格式规范
- 提供深度洞察，不是简单总结
- 保持原始目标的核心意图

⚠️ **禁止**：
- 编造报告中没有的内容
- 简单复制粘贴子任务报告
- 忽略任务间的关联
- 使用模糊或笼统的表述

---

## 开始执行

**🚨 重要提醒**：
- 你的任务是生成**半年学习计划**，不是记录执行过程
- 必须调用 `write_file` 工具生成 `{session_id}/final_report.md`
- 不要陷入无限读取循环，读取完必要文件后立即生成报告

现在，请开始系统性地阅读所有子任务报告，提取关键信息，进行综合分析，最终生成一份高质量的、符合规范的最终汇总报告。

**记住**：你的目标不是完成一个形式化的文档，而是为用户提供真正有价值的、有洞察力的综合性分析报告！

**🔥 关键**：读取文件后，必须调用 `write_file` 生成学习计划！

加油！💪
"""

        # 创建汇总Agent配置
        config = ActorConfiguration(
            actor_id=f"summary_agent_{uuid.uuid4().hex[:8]}",
            task_id=task_spec.task_id,
            persona="资深内容整合专家与分析师",
            tools=[tool.name for tool in tools],
            knowledge=f"原始目标：{original_goal}\n子任务数：{total_subtasks}\nSession: {session_id}",
            system_prompt=system_prompt,
            execution_config={
                "max_iterations": 10,
                "timeout": 300,
                "require_detailed_report": True,
            },
            metadata={
                "created_at": datetime.now().isoformat(),
                "task_type": "summary",
                "original_goal": original_goal,
                "total_subtasks": total_subtasks,
                "session_id": session_id,
            },
        )

        # 创建DynamicActor实例
        actor = DynamicActor(
            actor_id=config.actor_id,
            task_id=config.task_id,
            task_description=task_spec.description,
            llm_client=self.llm,
            tools=tools,
            functions=functions,
            system_prompt=config.system_prompt,
            config=config,
        )

        return actor

    def _create_fallback_agent(
        self, task_spec: TaskSpecification, error_msg: str
    ) -> "DynamicActor":
        """创建降级智能体."""

        # 使用最基础的配置 - 优先使用 web_research，如果不可用则使用 file_operations
        basic_tools = []

        # 优先尝试使用网络搜索工具
        if self.tool_bundles.get("web_research") and self.tool_bundles["web_research"].get("tools"):
            basic_tools = [self.tool_bundles["web_research"]["tools"][0]]
        # 如果没有网络搜索，尝试使用文件操作工具
        elif self.tool_bundles.get("file_operations") and self.tool_bundles["file_operations"].get(
            "tools"
        ):
            basic_tools = [self.tool_bundles["file_operations"]["tools"][0]]
        # 如果所有工具都不可用，使用任何可用的第一个工具
        else:
            for _bundle_name, bundle_data in self.tool_bundles.items():
                if bundle_data.get("tools"):
                    basic_tools = [bundle_data["tools"][0]]
                    break

        # 如果完全没有工具，创建一个空列表（允许无工具运行）
        if not basic_tools:
            logger.warning("No tools available for fallback agent, creating agent without tools")

        basic_functions = self._tools_to_functions(basic_tools)

        fallback_config = ActorConfiguration(
            actor_id=f"fallback_{uuid.uuid4().hex[:8]}",
            task_id=task_spec.task_id,
            persona="您好！我是基础智能助手，会尽力完成任务。",
            tools=[tool.name for tool in basic_tools],
            knowledge="基础工作原则：理解需求、尝试解决、及时反馈。",
            system_prompt="您是一个基础智能助手。请使用可用工具尽力完成用户的任务。",
            execution_config={"max_iterations": 5, "timeout": 60},
            metadata={"fallback_reason": error_msg, "created_at": datetime.now().isoformat()},
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
            config=fallback_config,
        )
