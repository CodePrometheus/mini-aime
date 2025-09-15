"""基于 LangChain 工具的动态智能体工厂。"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..llm.base import BaseLLMClient
from .models import Task, TaskStatus


class TaskSpecification(BaseModel):
    """用于创建智能体的任务规格。"""
    
    task_id: str = Field(..., description="任务唯一标识")
    description: str = Field(..., description="任务描述")
    parent_goal: str | None = Field(None, description="父级目标")
    context: Dict[str, Any] = Field(default_factory=dict, description="执行上下文")
    constraints: List[str] = Field(default_factory=list, description="约束条件")
    priority: str = Field(default="medium", description="优先级")
    estimated_duration: int | None = Field(None, description="预估时长(分钟)")


class TaskAnalysis(BaseModel):
    """LLM 对任务需求分析的结果。"""
    
    task_type: str = Field(..., description="任务类型")
    domain: str = Field(..., description="领域")
    required_capabilities: List[str] = Field(..., description="所需能力")
    complexity: str = Field(..., description="复杂度")
    key_challenges: List[str] = Field(..., description="关键挑战")
    recommended_tools: List[str] = Field(..., description="推荐工具包")
    knowledge_areas: List[str] = Field(..., description="知识领域")


class ActorConfiguration(BaseModel):
    """智能体配置。"""
    
    actor_id: str = Field(..., description="智能体ID")
    task_id: str = Field(..., description="关联任务ID")
    persona: str = Field(..., description="人格设定")
    tools: List[str] = Field(..., description="工具名称列表")
    knowledge: str = Field(..., description="注入的知识")
    system_prompt: str = Field(..., description="完整系统提示")
    execution_config: Dict[str, Any] = Field(default_factory=dict, description="执行配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ActorFactory:
    """按需创建具备动态配置的专用智能体的工厂。"""
    
    def __init__(self, llm_client: BaseLLMClient, initialize_tools: bool = True):
        self.llm = llm_client
        if initialize_tools:
            self._initialize_tool_bundles()
        else:
            # 测试模式：使用空的工具包
            self.tool_bundles = {
                "web_research": [],
                "file_operations": [],
                "data_processing": [],
                "communication": []
            }
    
    def _initialize_tool_bundles(self):
        """初始化工具包。"""
        
        # 网络搜索工具
        tavily_search = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
        )
        
        # 文件操作工具
        def read_file_func(file_path: str) -> str:
            """读取文件内容。"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"读取文件失败: {str(e)}"
        
        def write_file_func(file_path: str, content: str) -> str:
            """写入文件内容。"""
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"成功写入文件: {file_path}"
            except Exception as e:
                return f"写入文件失败: {str(e)}"
        
        def list_directory_func(directory_path: str) -> str:
            """列出目录内容。"""
            import os
            try:
                items = os.listdir(directory_path)
                return f"目录 {directory_path} 内容:\n" + "\n".join(items)
            except Exception as e:
                return f"列出目录失败: {str(e)}"
        
        # 数据处理工具
        def parse_json_func(json_string: str) -> str:
            """解析 JSON 字符串。"""
            try:
                parsed = json.loads(json_string)
                return f"JSON解析成功: {json.dumps(parsed, ensure_ascii=False, indent=2)}"
            except Exception as e:
                return f"JSON解析失败: {str(e)}"
        
        # 定义工具包
        self.tool_bundles = {
            "web_research": {
                "tools": [
                    tavily_search,
                    Tool(
                        name="extract_content",
                        func=lambda url: f"从 {url} 提取的内容（模拟）",
                        description="从网页URL提取结构化内容",
                    )
                ],
                "description": "网络信息搜索和提取",
                "use_cases": ["研究", "信息收集", "验证事实"]
            },
            
            "file_operations": {
                "tools": [
                    Tool(
                        name="read_file",
                        func=read_file_func,
                        description="读取指定路径的文件内容",
                    ),
                    Tool(
                        name="write_file", 
                        func=write_file_func,
                        description="将内容写入指定路径的文件",
                    ),
                    Tool(
                        name="list_directory",
                        func=list_directory_func,
                        description="列出指定目录的所有文件和子目录",
                    )
                ],
                "description": "文件系统操作",
                "use_cases": ["文档生成", "数据存储", "文件管理"]
            },
            
            "data_processing": {
                "tools": [
                    Tool(
                        name="parse_json",
                        func=parse_json_func,
                        description="解析JSON格式的字符串数据",
                    ),
                    Tool(
                        name="format_data",
                        func=lambda data: f"格式化数据: {data}",
                        description="格式化和美化数据输出",
                    )
                ],
                "description": "数据解析和处理", 
                "use_cases": ["数据分析", "格式转换", "验证"]
            },
            
            "communication": {
                "tools": [
                    Tool(
                        name="send_notification",
                        func=lambda message: f"发送通知: {message}",
                        description="发送通知消息",
                    ),
                    Tool(
                        name="generate_report",
                        func=lambda content: f"生成报告: {content}",
                        description="生成结构化报告",
                    )
                ],
                "description": "通信和报告",
                "use_cases": ["通知", "报告", "协调"]
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
            )
            
            # 7. 构建配置
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
                    "selected_bundles": task_analysis.recommended_tools
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
    
    async def _analyze_task_requirements(self, task_spec: TaskSpecification) -> TaskAnalysis:
        """使用LLM分析任务需求."""
        
        analysis_prompt = f"""
        分析以下任务的需求：

        任务描述：{task_spec.description}
        父级目标：{task_spec.parent_goal or "无"}
        已知约束：{', '.join(task_spec.constraints) if task_spec.constraints else "无"}
        执行上下文：{json.dumps(task_spec.context, ensure_ascii=False)}

        请分析：
        1. 任务类型和领域
        2. 所需专业能力
        3. 可能的挑战和风险
        4. 推荐的工具类型
        5. 需要的专业知识

        可用工具包：
        - web_research: 网络搜索和信息提取
        - file_operations: 文件读写和管理
        - data_processing: 数据解析和处理
        - communication: 通信和报告

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
        except Exception as e:
            # 降级策略：使用简化LLM分析
            return await self._fallback_task_analysis(task_spec)
    
    def _select_tool_bundles(self, task_analysis: TaskAnalysis) -> List[Tool]:
        """基于分析结果选择工具包."""
        
        selected_tools = []
        
        # 根据推荐工具包选择
        for bundle_name in task_analysis.recommended_tools:
            if bundle_name in self.tool_bundles:
                bundle = self.tool_bundles[bundle_name]
                selected_tools.extend(bundle["tools"])
        
        # 如果没有选择任何工具，提供默认工具
        if not selected_tools:
            selected_tools.extend(self.tool_bundles["web_research"]["tools"])
            selected_tools.extend(self.tool_bundles["file_operations"]["tools"][:2])  # 只取前2个
        
        return selected_tools
    
    def _tools_to_functions(self, tools: List[Tool]) -> List[Dict]:
        """将 LangChain 工具转换为 OpenAI Function Calling 格式."""
        
        functions = []
        for tool in tools:
            function = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # 检查是否是 StructuredTool
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.schema()
                    function["function"]["parameters"]["properties"] = schema.get("properties", {})
                    function["function"]["parameters"]["required"] = schema.get("required", [])
                except Exception:
                    # 降级为简单参数
                    function["function"]["parameters"]["properties"] = {
                        "input": {
                            "type": "string",
                            "description": f"输入参数给 {tool.name}"
                        }
                    }
                    function["function"]["parameters"]["required"] = ["input"]
            else:
                # 简单工具，推断参数
                if "file" in tool.name.lower():
                    if "read" in tool.name.lower():
                        function["function"]["parameters"]["properties"] = {
                            "file_path": {
                                "type": "string",
                                "description": "要读取的文件路径"
                            }
                        }
                        function["function"]["parameters"]["required"] = ["file_path"]
                    elif "write" in tool.name.lower():
                        function["function"]["parameters"]["properties"] = {
                            "file_path": {
                                "type": "string", 
                                "description": "要写入的文件路径"
                            },
                            "content": {
                                "type": "string",
                                "description": "要写入的内容"
                            }
                        }
                        function["function"]["parameters"]["required"] = ["file_path", "content"]
                    else:
                        function["function"]["parameters"]["properties"] = {
                            "directory_path": {
                                "type": "string",
                                "description": "目录路径"
                            }
                        }
                        function["function"]["parameters"]["required"] = ["directory_path"]
                else:
                    # 默认单一输入参数
                    function["function"]["parameters"]["properties"] = {
                        "input": {
                            "type": "string",
                            "description": f"输入给 {tool.name} 的参数"
                        }
                    }
                    function["function"]["parameters"]["required"] = ["input"]
            
            functions.append(function)
        
        return functions
    
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
    
    def _compose_prompt_for_functions(self, persona: str, knowledge: str, environment: str) -> str:
        """Compose system prompt optimized for Function Calling mode."""
        
        return f"""{persona}

相关知识：
{knowledge}

环境信息：
{environment}

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
    
    def _build_execution_config(self, task_analysis: TaskAnalysis) -> Dict[str, Any]:
        """构建执行配置."""
        
        # 根据复杂度调整参数
        complexity_configs = {
            "low": {"max_iterations": 5, "timeout": 60},
            "medium": {"max_iterations": 10, "timeout": 180},
            "high": {"max_iterations": 20, "timeout": 300}
        }
        
        config = complexity_configs.get(task_analysis.complexity, complexity_configs["medium"])
        
        config.update({
            "enable_progress_reporting": True,
            "auto_retry_on_error": True,
            "max_retries": 3,
            "log_level": "INFO"
        })
        
        return config
    
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
