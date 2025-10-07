"""支持 ReAct 范式与 Function Calling 的动态智能体实现。"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.tools import Tool

from ..config.settings import settings
from ..llm.base import BaseLLMClient
from .models import ExecutionStep, TaskStatus, UserEvent, UserEventType


logger = logging.getLogger(__name__)
ACTOR_LOG_PREFIX = "MiniAime|Actor|"
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


CACHEABLE_TOOLS = {"read_file", "read_files", "list_directory"}

CACHEABLE_TOOLS = {"read_file", "read_files", "list_directory"}


class DynamicActor:
    """用于执行特定任务的自主智能体（采用 ReAct 范式）。"""

    def __init__(
        self,
        actor_id: str,
        task_id: str,
        task_description: str,
        llm_client: BaseLLMClient,
        tools: list[Tool],
        functions: list[dict],
        system_prompt: str,
        config: Any,  # ActorConfiguration
    ):
        self.actor_id = actor_id
        self.task_id = task_id
        self.task_description = task_description
        self.llm = llm_client
        self.tools = tools
        self.functions = functions
        self.system_prompt = system_prompt
        self.config = config

        # 执行状态
        self.memory: list[ExecutionStep] = []
        self.status = TaskStatus.PENDING
        self.progress_manager = None
        self.start_time = None
        self.end_time = None

        # 工具映射
        self.tool_map = {tool.name: tool for tool in tools}
        self.tool_call_cache: dict[tuple[str, str], dict[str, Any]] = {}

    async def execute(self, progress_manager) -> dict[str, Any]:
        """
        执行任务的主循环。

        Args:
            progress_manager: 进度管理器

        Returns:
            执行结果字典
        """
        self.progress_manager = progress_manager
        self.start_time = datetime.now()
        self.status = TaskStatus.IN_PROGRESS
        logger.info(f"{ACTOR_LOG_PREFIX} actor_start actor={self.actor_id} task={self.task_id}")

        try:
            # 报告开始执行
            await self._report_progress("开始执行任务", "info")

            # 执行配置
            # generate_final_report 任务需要足够的步数来完成：列目录、读文件、整合、生成报告
            if self.task_id == "generate_final_report":
                max_iterations = 15  # 足够的步数来完成完整流程
            elif self.task_id == "task_final_summary":
                max_iterations = self.config.execution_config.get(
                    "max_iterations", 10
                )  # 汇总任务步数
            else:
                max_iterations = self.config.execution_config.get(
                    "max_iterations", 5
                )  # 普通任务步数

            for step in range(max_iterations):
                logger.info(
                    f"{ACTOR_LOG_PREFIX} react_step_begin actor={self.actor_id} step={step + 1}"
                )
                # ReAct cycle: thought -> action -> observation
                try:
                    thought, action_result = await self._react_step()

                    # 记录执行步骤
                    # 详细日志：检查字段类型
                    observation_raw = action_result.get("observation", "")
                    if not isinstance(observation_raw, str):
                        logger.warning(
                            f"{ACTOR_LOG_PREFIX} non_string_observation actor={self.actor_id} "
                            f"step={step + 1} type={type(observation_raw).__name__} "
                            f"repr={repr(observation_raw)[:200]}"
                        )

                    execution_step = ExecutionStep(
                        thought=thought,
                        action=action_result.get("action", ""),
                        observation=str(observation_raw) if observation_raw else "",
                        step_id=f"{self.actor_id}_step_{step + 1}",
                    )
                    self.memory.append(execution_step)

                    # 检查是否标记为完成（通过特殊标记或工具调用）
                    # LLM 会在 ReAct 中主动表达完成意图
                    if self._check_completion_signal(action_result):
                        self.status = TaskStatus.COMPLETED
                        logger.info(
                            f"{ACTOR_LOG_PREFIX} task_completed actor={self.actor_id} steps={len(self.memory)}"
                        )
                        break

                except Exception as e:
                    # 错误处理
                    error_handled = await self._handle_error(e)
                    if not error_handled:
                        self.status = TaskStatus.FAILED
                        logger.error(
                            f"{ACTOR_LOG_PREFIX} task_failed actor={self.actor_id} error={e!s}"
                        )
                        break

            # 检查任务完成状态
            if self.status != TaskStatus.FAILED and self.status != TaskStatus.COMPLETED:
                # 如果是研究任务但没有生成文件，标记为失败
                if self._is_research_task() and not self._verify_research_file_generated():
                    self.status = TaskStatus.FAILED
                    logger.error(
                        f"{ACTOR_LOG_PREFIX} task_failed_no_file actor={self.actor_id} task={self.task_id}"
                    )
                    await self._report_progress(
                        "任务失败：研究任务必须生成文件，但未检测到文件输出", 
                        "error"
                    )
                else:
                    # 非研究任务或已生成文件，标记为完成
                    self.status = TaskStatus.COMPLETED

            # 生成最终报告
            final_result = await self._generate_final_report()

            # 报告完成
            await self._report_progress("任务执行完成", "completed")
            logger.info(
                f"{ACTOR_LOG_PREFIX} actor_end actor={self.actor_id} status={self.status.value}"
            )

            return final_result

        except Exception as e:
            self.status = TaskStatus.FAILED
            await self._report_progress(f"任务执行失败: {e!s}", "error")
            logger.error(f"{ACTOR_LOG_PREFIX} actor_exception actor={self.actor_id} error={e!s}")
            return {"status": "failed", "error": str(e)}
        finally:
            self.end_time = datetime.now()

    async def _react_step(self) -> tuple[str, dict[str, Any]]:
        """
        执行一个 ReAct 步骤：思考 -> 行动 -> 观察。

        Returns:
            (thought, action_result) 元组
        """
        # 构建对话消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"任务：{self.task_description}"},
        ]

        # 添加执行历史（最近3步）
        for step in self.memory[-3:]:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"之前的思考：{step.thought}\n之前的行动：{step.action}",
                    },
                    {"role": "assistant", "content": f"观察结果：{step.observation}"},
                ]
            )

        # 当前推理请求
        messages.append(
            {
                "role": "user",
                "content": "基于以上上下文，请进行下一步的思考和行动。如果任务已完成，请说明完成情况。",
            }
        )

        try:
            # 调用 LLM（支持 Function Calling）
            if hasattr(self.llm, "complete_with_functions"):
                # 如果 LLM 客户端支持 Function Calling
                logger.info(
                    f"{ACTOR_LOG_PREFIX} llm_request actor={self.actor_id} mode=function_call"
                )
                response = await self.llm.complete_with_functions(messages, self.functions)
            else:
                # 降级到普通对话
                logger.info(f"{ACTOR_LOG_PREFIX} llm_request actor={self.actor_id} mode=text")
                response = await self.llm.complete_with_context(messages)

            # 解析响应
            if isinstance(response, dict):
                if response.get("function_call"):
                    logger.info(
                        f"{ACTOR_LOG_PREFIX} llm_response actor={self.actor_id} type=function_call"
                    )
                    # Log function_call payload preview for debugging purposes
                    try:
                        fc = response.get("function_call", {})
                        fc_name = fc.get("name", "")
                        fc_args = fc.get("arguments", {})
                        if isinstance(fc_args, str):
                            args_preview = fc_args[:200]
                        else:
                            args_preview = json.dumps(fc_args, ensure_ascii=False)[:200]
                        content_preview = (response.get("content") or "")[:200]
                        logger.info(
                            f"{ACTOR_LOG_PREFIX} function_call payload actor={self.actor_id} "
                            f"name={fc_name} args_preview={args_preview} content_preview={content_preview}"
                        )
                    except Exception:
                        pass
                    return await self._handle_function_call(response)

                if response.get("tool_calls"):
                    logger.info(
                        f"{ACTOR_LOG_PREFIX} llm_response actor={self.actor_id} type=tool_calls"
                    )
                    return await self._handle_tool_calls(response["tool_calls"])

                logger.info(f"{ACTOR_LOG_PREFIX} llm_response actor={self.actor_id} type=json_text")
                return await self._handle_text_response(response)

            # 兼容纯文本响应
            logger.info(f"{ACTOR_LOG_PREFIX} llm_response actor={self.actor_id} type=text")
            return await self._handle_text_response(response)

        except Exception as e:
            # LLM 调用失败，返回错误
            # 详细日志：记录异常类型和堆栈跟踪
            import traceback

            error_details = {
                "exception_type": type(e).__name__,
                "exception_value": str(e),
                "exception_repr": repr(e),
                "is_slice_object": isinstance(e, slice),
            }
            logger.error(
                f"{ACTOR_LOG_PREFIX} llm_error actor={self.actor_id} "
                f"type={error_details['exception_type']} "
                f"value={error_details['exception_value'][:200]} "
                f"is_slice={error_details['is_slice_object']}"
            )
            logger.debug(
                f"{ACTOR_LOG_PREFIX} llm_error_traceback actor={self.actor_id}\n{traceback.format_exc()}"
            )

            thought = f"LLM调用失败: {e!s}"
            action_result = {
                "action": "error",
                "observation": f"无法获取LLM响应: {e!s}",
                "success": False,
            }
            return thought, action_result

    async def _emit_user_event(
        self,
        event_type: UserEventType,
        title: str,
        content: str,
        level: str = "info",
        collapsible: bool = False,
        details: dict | None = None,
    ):
        """发送用户事件到 ProgressManager"""
        if not self.progress_manager:
            return

        user_event = UserEvent(
            event_type=event_type,
            title=title,
            content=content,
            timestamp=datetime.now(),
            agent_id=self.actor_id,
            task_id=self.task_id,
            level=level,
            collapsible=collapsible,
            details=details,
        )

        await self.progress_manager.emit_user_event(user_event)

    async def _handle_function_call(self, response: dict) -> tuple[str, dict[str, Any]]:
        """处理函数调用响应。"""

        function_call = response["function_call"]
        function_name = function_call["name"]
        raw_args = function_call.get("arguments", {})

        if isinstance(raw_args, str):
            try:
                function_args = json.loads(raw_args)
            except json.JSONDecodeError:
                function_args = {"tool_input": raw_args}
        elif isinstance(raw_args, dict):
            function_args = raw_args
        else:
            function_args = {"tool_input": str(raw_args)}

        # 提取思考内容
        thought_content = response.get("content") or f"决定调用工具 {function_name}"
        thought = f"决定调用工具 {function_name}"

        # 1️⃣ 发送 THOUGHT 事件
        await self._emit_user_event(
            event_type=UserEventType.THOUGHT,
            title=f"思考步骤 {len(self.memory) + 1}",
            content=thought_content,
            collapsible=len(thought_content) > 100,
        )

        # 2️⃣ 发送 ACTION 事件
        await self._emit_user_event(
            event_type=UserEventType.ACTION,
            title=f"调用工具: {function_name}",
            content=f"执行工具 {function_name}",
            level="info",
        )

        logger.info(f"{ACTOR_LOG_PREFIX} tool_call actor={self.actor_id} name={function_name}")
        # Log function arguments preview
        try:
            args_preview = (
                raw_args
                if isinstance(raw_args, str)
                else json.dumps(function_args, ensure_ascii=False)
            )[:300]
        except Exception:
            args_preview = str(raw_args)[:300]
        logger.info(
            f"{ACTOR_LOG_PREFIX} tool_args actor={self.actor_id} name={function_name} args_preview={args_preview}"
        )

        # 构建缓存键
        cache_key = self._build_cache_key(function_name, function_args)

        # 执行工具
        if function_name in self.tool_map:
            try:
                if cache_key in self.tool_call_cache:
                    cached = self.tool_call_cache[cache_key]
                    logger.info(
                        f"{ACTOR_LOG_PREFIX} tool_cache_hit actor={self.actor_id} name={function_name}"
                    )
                    action_result = {
                        "action": f"命中缓存 {function_name}({json.dumps(function_args, ensure_ascii=False)})",
                        "observation": cached.get("observation", ""),
                        "success": True,
                        "cached": True,
                    }
                    await self._emit_user_event(
                        event_type=UserEventType.OBSERVATION,
                        title="观察结果",
                        content=cached.get("observation", "")[:300],
                        level="success",
                        collapsible=len(cached.get("observation", "")) > 150,
                    )
                    return thought, action_result

                tool = self.tool_map[function_name]

                # 统一通过 execute_with_retry（若工具不支持则降级）
                max_retries = getattr(settings, "default_tool_retry", 0)
                backoff_ms = getattr(settings, "retry_backoff_ms", 0)

                # LangChain Tool 适配：尝试调用底层 BaseTool，如果不可用则走原始接口
                underlying = getattr(tool, "tool", None)
                if hasattr(underlying, "execute_with_retry"):
                    # 传参约定：将 function_args 作为 kwargs 透传
                    safe_kwargs = (
                        function_args
                        if isinstance(function_args, dict)
                        else {"tool_input": function_args}
                    )
                    safe_result = await underlying.execute_with_retry(
                        max_retries=max_retries, backoff_ms=backoff_ms, **safe_kwargs
                    )
                    if not safe_result.get("success") and settings.human_in_loop_enabled:
                        # 工具失败超限 → 阻塞并等待输入
                        self.status = TaskStatus.BLOCKED
                        await self._report_progress(
                            f"工具失败并已达重试上限，进入等待用户输入: {function_name}", "blocked"
                        )
                        return (
                            f"工具 {function_name} 失败，等待用户输入以继续",
                            {
                                "action": f"调用工具 {function_name}",
                                "observation": safe_result.get("error", "unknown error"),
                                "success": False,
                                "blocked": True,
                            },
                        )

                    # 成功或未启用人机环：返回统一结果
                    result = safe_result.get("result")
                else:
                    # 回退到原始 LangChain 接口
                    if isinstance(function_args, dict):
                        if "tool_input" in function_args:
                            result = await tool.arun(function_args["tool_input"])
                        elif hasattr(tool, "args_schema") and tool.args_schema:
                            # StructuredTool 需要完整的字典参数
                            result = await tool.ainvoke(function_args)
                        elif len(function_args) == 1:
                            result = await tool.arun(next(iter(function_args.values())))
                        else:
                            result = await tool.ainvoke(function_args)
                    else:
                        result = await tool.arun(function_args)

                # Log tool result preview
                try:
                    result_preview = str(result)[:500]
                except Exception:
                    result_preview = "<unprintable result>"
                logger.info(
                    f"{ACTOR_LOG_PREFIX} tool_result actor={self.actor_id} name={function_name} result_preview={result_preview}"
                )

                # 检查是否是文件不存在的情况
                if "FILE_NOT_FOUND:" in str(result):
                    # 文件不存在，标记任务为pending状态
                    if self.progress_manager:
                        await self.progress_manager.update_progress(
                            self.task_id, 
                            "pending", 
                            "文件不存在，等待文件创建完成后重试"
                        )
                    return {
                        "status": "pending",
                        "reason": "file_not_found",
                        "message": "文件不存在，任务将重新调度"
                    }

                action_result = {
                    "action": f"调用工具 {function_name}({json.dumps(function_args, ensure_ascii=False)})",
                    "observation": str(result),
                    "success": True,
                }

                if cache_key:
                    self.tool_call_cache[cache_key] = {
                        "observation": str(result),
                    }

                if function_name in {"write_file", "write_files"}:
                    self.tool_call_cache.clear()

            except Exception as e:
                action_result = {
                    "action": f"调用工具 {function_name}",
                    "observation": f"工具执行失败: {e!s}",
                    "success": False,
                }
                logger.error(
                    f"{ACTOR_LOG_PREFIX} tool_fail actor={self.actor_id} name={function_name} error={e!s}"
                )
        else:
            action_result = {
                "action": f"尝试调用工具 {function_name}",
                "observation": f"工具 {function_name} 不存在",
                "success": False,
            }

        # 发送 OBSERVATION 事件
        observation_content = action_result.get("observation", "")
        observation_preview = (
            observation_content[:300] if len(observation_content) > 300 else observation_content
        )
        await self._emit_user_event(
            event_type=UserEventType.OBSERVATION,
            title="观察结果",
            content=observation_preview,
            level="success" if action_result.get("success") else "error",
            collapsible=len(observation_content) > 150,
        )

        return thought, action_result

    async def _handle_text_response(self, response: Any) -> tuple[str, dict[str, Any]]:
        """处理文本或 JSON 响应（非函数调用）。"""

        if isinstance(response, dict):
            text = response.get("content") or json.dumps(response, ensure_ascii=False)
        else:
            text = str(response or "")

        lines = text.split("\n")
        thought = ""
        action = ""

        for line in lines:
            line = line.strip()
            if line.startswith("思考：") or line.startswith("Thought:"):
                thought = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("行动：") or line.startswith("Action:"):
                action = line.split("：", 1)[-1].split(":", 1)[-1].strip()

        if not thought:
            # 安全地获取前100个字符（处理字典和字符串）
            if isinstance(response, dict):
                thought = json.dumps(response, ensure_ascii=False)[:100]
            elif isinstance(response, str):
                thought = response[:100]
            else:
                thought = str(response)[:100]

        if not action:
            action = "继续分析任务"

        # 如果没有明确的工具调用，作为思考记录
        action_result = {
            "action": action,
            "observation": "继续思考中，尚未调用具体工具",
            "success": True,
        }

        logger.info(f"{ACTOR_LOG_PREFIX} text_step actor={self.actor_id} action={action}")
        return thought, action_result

    def _check_completion_signal(self, action_result: dict) -> bool:
        """
        检查任务完成信号（无 LLM 调用）。

        完成信号可以来自：
        1. action 中包含明确的完成标记
        2. observation 中有完成关键词
        3. 达到足够的执行步数且有实质性结果

        这是轻量级的启发式判断，让 LLM 在 ReAct 中通过
        thought/action 表达完成意图，而不是额外询问。
        """
        observation = action_result.get("observation", "")
        action = action_result.get("action", "")

        # 特殊处理：generate_final_report 任务必须确保生成了 final_report.md
        if self.task_id == "generate_final_report":
            # 方法1: 检查是否调用了 write_file 并且结果成功
            if "write_file" in action and "final_report.md" in action:
                # 检查 action_result 中的 success 标志
                if action_result.get("success"):
                    return True
                # 备用：检查关键词（兼容不同的返回格式）
                if "成功" in observation or "写入" in observation or "保存" in observation:
                    return True

            # 方法2: 检查历史步骤，看是否已经成功生成了 final_report.md
            for step in self.memory:
                step_action = step.action or ""
                step_observation = step.observation or ""
                if (
                    "write_file" in step_action
                    and "final_report.md" in step_action
                    and (
                        "成功" in step_observation
                        or "写入" in step_observation
                        or "保存" in step_observation
                    )
                ):
                    return True

            # 方法3: 最终兜底 - 直接检查文件是否存在
            if self.progress_manager:
                session_id = getattr(self.progress_manager, "session_id", None)
                if session_id:
                    import os

                    from src.tools.file_tools import _find_project_root

                    project_root = _find_project_root()
                    expected_file = os.path.join(
                        project_root, "docs", session_id, "final_report.md"
                    )
                    if os.path.exists(expected_file) and os.path.getsize(expected_file) > 1000:
                        # 文件存在且有实质内容（>1KB），认为已完成
                        logger.info(
                            f"{ACTOR_LOG_PREFIX} completion_verified actor={self.actor_id} "
                            f"file_exists={expected_file}"
                        )
                        return True

            # 如果所有检查都未通过，不算完成
            return False

        # 1. 检查明确的完成标记
        completion_markers = [
            "TASK_COMPLETE",
            "任务已完成",
            "任务完成",
            "Task completed successfully",
            "已生成最终报告",
            "Final report generated",
        ]

        for marker in completion_markers:
            if marker in observation or marker in action:
                # 对于研究任务，需要验证是否真正生成了文件
                if self._is_research_task() and not self._verify_research_file_generated():
                    logger.warning(f"{ACTOR_LOG_PREFIX} completion_marker_found_but_no_file actor={self.actor_id} task={self.task_id}")
                    return False
                return True

        # 2. 检查是否已执行足够步骤且有实质性输出
        # 这是保底机制，防止永久循环
        min_steps = 3
        if len(self.memory) >= min_steps and len(observation) > 100 and "read_file" not in action:
            # 检查是否有成功/完成类关键词
            success_keywords = [
                "成功",
                "完成",
                "已生成",
                "已保存",
                "success",
                "completed",
                "generated",
            ]
            if any(kw in observation for kw in success_keywords):
                # 对于研究任务，需要验证是否真正生成了文件
                if self._is_research_task() and not self._verify_research_file_generated():
                    logger.warning(f"{ACTOR_LOG_PREFIX} success_keyword_found_but_no_file actor={self.actor_id} task={self.task_id}")
                    return False
                return True

        return False

    def _is_research_task(self) -> bool:
        """判断当前任务是否是研究任务（需要生成文件的任务）。"""
        # 默认所有任务都需要生成文件，除非明确说明不需要
        task_desc_lower = self.task_description.lower()
        
        # 检查是否有明确说明不需要文件的关键词
        no_file_keywords = [
            "不需要文件",
            "不生成文件",
            "不保存文件",
            "仅搜索",
            "仅查询",
            "仅检查",
            "仅验证"
        ]
        
        # 如果明确说明不需要文件，则不是研究任务
        if any(keyword in task_desc_lower for keyword in no_file_keywords):
            return False
            
        # 其他所有任务都默认需要生成文件
        return True

    def _verify_research_file_generated(self) -> bool:
        """验证研究任务是否真正生成了预期的文件。"""
        try:
            import os
            import time
            from src.tools.file_tools import _find_project_root
            
            if not self.progress_manager:
                return False
                
            session_id = getattr(self.progress_manager, "session_id", None)
            if not session_id:
                return False
            
            project_root = _find_project_root()
            docs_dir = os.path.join(project_root, "docs", session_id)
            
            if not os.path.exists(docs_dir):
                return False
            
            # 检查最近10分钟内是否有新文件生成
            current_time = time.time()
            recent_files = []
            
            for filename in os.listdir(docs_dir):
                file_path = os.path.join(docs_dir, filename)
                if os.path.isfile(file_path) and filename.endswith('.md'):
                    # 检查文件是否在最近10分钟内创建或修改
                    file_mtime = os.path.getmtime(file_path)
                    if current_time - file_mtime < 600:  # 10分钟
                        recent_files.append(filename)
            
            # 如果有新文件生成，说明任务确实完成了文件输出
            if recent_files:
                logger.info(f"{ACTOR_LOG_PREFIX} file_verification_success actor={self.actor_id} recent_files={recent_files}")
                return True
            
            # 如果没有新文件，检查任务是否真的需要生成文件
            # 如果任务描述中没有明确提到要生成文件，可能不需要文件输出
            task_desc_lower = self.task_description.lower()
            if not any(keyword in task_desc_lower for keyword in ["保存", "生成", "写入", "创建", "文件", "报告", "总结", "计划"]):
                logger.info(f"{ACTOR_LOG_PREFIX} task_no_file_output_required actor={self.actor_id}")
                return True  # 不需要文件输出的任务也算完成
            
            logger.warning(f"{ACTOR_LOG_PREFIX} file_verification_failed actor={self.actor_id} recent_files={recent_files}")
            return False
            
        except Exception as e:
            logger.error(f"{ACTOR_LOG_PREFIX} file_verification_error actor={self.actor_id} error={e}")
            return False

    async def _handle_error(self, error: Exception) -> bool:
        """
        处理执行错误。

        Returns:
            是否成功处理错误（True 表示可以继续，False 表示需要停止）
        """
        error_msg = str(error)

        # 记录错误步骤
        error_step = ExecutionStep(
            thought=f"遇到错误: {error_msg}",
            action="error_handling",
            observation=f"错误详情: {error_msg}",
            step_id=f"{self.actor_id}_error_{len(self.memory)}",
        )
        self.memory.append(error_step)

        # 报告错误
        await self._report_progress(f"遇到错误: {error_msg[:100]}", "error")
        logger.error(f"{ACTOR_LOG_PREFIX} error actor={self.actor_id} msg={error_msg}")

        # 使用LLM分析错误并制定恢复策略
        recovery_prompt = f"""
        分析这个错误并制定恢复策略：

        错误信息：{error_msg}
        当前任务：{self.task_description}
        执行步数：{len(self.memory)}

        请分析：
        1. 这是什么类型的错误？
        2. 错误是否可以恢复？
        3. 应该采取什么恢复策略？

        返回JSON：
        {{
            "error_type": "network|permission|timeout|logic|unknown",
            "recoverable": true/false,
            "recovery_strategy": "retry|skip|abort|alternative_approach",
            "reasoning": "分析和建议的详细说明"
        }}
        """

        try:
            response = await self.llm.complete(recovery_prompt)
            recovery_plan = json.loads(response)

            if recovery_plan.get("recoverable", False):
                strategy = recovery_plan.get("recovery_strategy", "retry")
                reasoning = recovery_plan.get("reasoning", "")
                await self._report_progress(f"错误恢复策略: {strategy} - {reasoning}", "retry")
                logger.info(
                    f"{ACTOR_LOG_PREFIX} recovery actor={self.actor_id} strategy={strategy}"
                )
                return True
            else:
                reasoning = recovery_plan.get("reasoning", "无法恢复")
                await self._report_progress(f"无法恢复错误: {reasoning}", "error")
                logger.info(f"{ACTOR_LOG_PREFIX} unrecoverable actor={self.actor_id}")
                return False

        except Exception:
            # 降级策略：基于执行次数的简单判断
            if len(self.memory) < 10:
                await self._report_progress("尝试继续执行（降级恢复）", "retry")
                logger.info(
                    f"{ACTOR_LOG_PREFIX} recovery_fallback actor={self.actor_id} mode=continue"
                )
                return True
            else:
                await self._report_progress("执行步骤过多，停止尝试", "error")
                logger.info(f"{ACTOR_LOG_PREFIX} recovery_fallback actor={self.actor_id} mode=stop")
                return False

    async def _generate_final_report(self) -> dict[str, Any]:
        """
        生成结构化最终报告。

        根据论文要求，最终报告包含三个基本部分：
        1. 状态更新 (Status Update)
        2. 结论摘要 (Conclusion Summary)
        3. 参考指针 (Reference Pointers)
        """

        execution_time = 0
        if self.start_time and self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()

        # 1. 状态更新 - 明确的任务状态
        status_update = {
            "task_id": self.task_id,
            "final_status": self.status.value,
            "completed": self.status == TaskStatus.COMPLETED,
            "execution_steps": len(self.memory),
            "execution_time_seconds": execution_time,
        }

        # 2. 结论摘要 - 任务执行的叙述性总结
        conclusion_summary = await self._generate_conclusion_summary()

        # 3. 参考指针 - 关键工件的结构化指针集合
        reference_pointers = self._extract_reference_pointers()

        # 构建完整的结构化报告
        final_report = {
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            # 论文要求的三个核心部分
            "status_update": status_update,
            "conclusion_summary": conclusion_summary,
            "reference_pointers": reference_pointers,
            # 额外的有用信息
            "metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_steps": len(self.memory),
                "tools_used": list(
                    set(
                        step.action.split("(")[0].replace("调用工具 ", "")
                        for step in self.memory
                        if "调用工具" in step.action
                    )
                ),
            },
        }

        # 注意：不再自动生成技术性的 final_report_{task_id}.md 文件
        # Actor 应该在执行过程中通过 write_file 工具主动生成用户需要的文件
        # 这样生成的文件才是真正有价值的输出，而不是技术元数据

        # 只记录执行报告的路径到 final_report 中（用于调试和追踪）
        try:
            from src.tools.file_tools import _find_project_root

            project_root = _find_project_root()
            session_id = (
                getattr(self.progress_manager, "session_id", None)
                if self.progress_manager
                else None
            )

            if session_id:
                docs_dir = os.path.join(project_root, "docs", session_id)
            else:
                docs_dir = os.path.join(project_root, "docs")

            # 记录执行报告路径（但不生成文件）
            final_report["execution_log_path"] = os.path.join(
                docs_dir, f"execution_log_{self.task_id}.md"
            )
        except Exception:
            pass

        # 提交最终报告给进度管理器
        if self.progress_manager:
            await self.progress_manager.submit_final_report(
                task_id=self.task_id, agent_id=self.actor_id, report=final_report
            )
        logger.info(
            f"{ACTOR_LOG_PREFIX} final_report actor={self.actor_id} steps={len(self.memory)}"
        )

        return final_report

    def _build_markdown_report(self, report: dict[str, Any]) -> str:
        """将最终报告渲染为 Markdown 文本。"""
        status = report.get("status_update", {})
        conclusion = report.get("conclusion_summary", {})
        pointers = report.get("reference_pointers", {})

        lines: list[str] = []
        lines.append(f"# 最终报告 - {report.get('task_id', '')}")
        lines.append("")
        lines.append("## 状态更新")
        lines.append(f"- 任务ID: {status.get('task_id')}")
        lines.append(f"- 最终状态: {status.get('final_status')}")
        lines.append(f"- 是否完成: {status.get('completed')}")
        lines.append(f"- 执行步数: {status.get('execution_steps')}")
        lines.append(f"- 执行耗时(秒): {status.get('execution_time_seconds')}")
        lines.append("")
        lines.append("## 结论摘要")
        lines.append(f"- 最终结论: {conclusion.get('final_outcome', '')}")
        obstacles = conclusion.get("obstacles_encountered") or []
        if obstacles:
            lines.append("- 遇到的障碍:")
            for item in obstacles:
                lines.append(f"  - {item}")
        insights = conclusion.get("key_insights") or []
        if insights:
            lines.append("- 关键洞察:")
            for item in insights:
                lines.append(f"  - {item}")
        narrative = conclusion.get("execution_summary", "")
        if narrative:
            lines.append("")
            lines.append("### 执行过程")
            lines.append(narrative)
        lines.append("")
        files = pointers.get("files") or []
        urls = pointers.get("urls") or []
        data_outputs = pointers.get("data_outputs") or []
        if files or urls or data_outputs:
            lines.append("## 引用与工件")
            if files:
                lines.append("### 文件")
                for f in files:
                    desc = f.get("description", "")
                    path = f.get("path", "")
                    if path:
                        lines.append(f"- {desc} ({path})")
                    else:
                        lines.append(f"- {desc}")
            if urls:
                lines.append("### 链接")
                for u in urls:
                    lines.append(f"- {u.get('description', '')}")
            if data_outputs:
                lines.append("### 数据输出（节选）")
                for d in data_outputs[:5]:
                    lines.append(f"- {d.get('data_preview', '')}")
        lines.append("")
        meta = report.get("metadata", {})
        if meta:
            lines.append("## 元数据")
            lines.append(f"- 开始时间: {meta.get('start_time')}")
            lines.append(f"- 结束时间: {meta.get('end_time')}")
            tools_used = ", ".join(meta.get("tools_used") or [])
            if tools_used:
                lines.append(f"- 使用工具: {tools_used}")

        return "\n".join(lines)

    async def _generate_conclusion_summary(self) -> dict[str, str]:
        """生成任务执行的叙述性摘要。"""

        # 提取关键信息
        key_observations = []
        obstacles = []
        insights = []

        for step_idx, step in enumerate(self.memory):
            # 详细日志：检查 observation 类型
            obs_type = type(step.observation).__name__
            if not isinstance(step.observation, str):
                logger.warning(
                    f"{ACTOR_LOG_PREFIX} unexpected_observation_type actor={self.actor_id} "
                    f"step={step_idx} type={obs_type} value={repr(step.observation)[:200]}"
                )

            # 确保 observation 是字符串
            observation_str = str(step.observation) if step.observation else ""

            # 识别关键观察
            if any(kw in observation_str for kw in ["成功", "完成", "找到", "创建"]):
                key_observations.append(observation_str[:200])

            # 识别障碍
            if any(kw in observation_str for kw in ["失败", "错误", "无法", "问题"]):
                obstacles.append(observation_str[:200])

            # 识别洞察（重要发现）
            thought_str = str(step.thought) if step.thought else ""
            if any(kw in thought_str for kw in ["发现", "注意到", "意识到", "了解到"]):
                insights.append(thought_str[:200])

        # 构建摘要
        # 详细日志：检查 final_outcome 数据类型
        final_outcome_raw = key_observations[-1] if key_observations else "任务执行完成"
        logger.debug(
            f"{ACTOR_LOG_PREFIX} conclusion_summary actor={self.actor_id} "
            f"final_outcome_type={type(final_outcome_raw).__name__} "
            f"key_observations_count={len(key_observations)} "
            f"obstacles_count={len(obstacles)}"
        )

        summary = {
            "final_outcome": final_outcome_raw,
            "obstacles_encountered": obstacles[:3] if obstacles else [],
            "key_insights": insights[:3] if insights else [],
            "execution_summary": self._create_execution_narrative(),
        }

        return summary

    def _create_execution_narrative(self) -> str:
        """创建执行过程的叙述性描述。"""

        if len(self.memory) == 0:
            return "任务未执行任何步骤"

        # 构建叙述
        narrative_parts = []

        # 开始
        narrative_parts.append(f"任务开始执行，目标是：{self.task_description}")

        # 中间过程（取关键步骤）
        if len(self.memory) > 2:
            mid_step = self.memory[len(self.memory) // 2]
            narrative_parts.append(f"执行过程中，{mid_step.thought}")

        # 结果
        final_step = self.memory[-1]
        narrative_parts.append(f"最终，{final_step.observation}")

        return " ".join(narrative_parts)

    def _extract_reference_pointers(self) -> dict[str, Any]:
        """提取关键工件的结构化指针集合。"""

        pointers = {"files": [], "urls": [], "data_outputs": [], "tool_results": {}}

        # 从执行历史中提取工件
        for step in self.memory:
            observation = step.observation

            # 提取文件路径
            if any(
                kw in observation for kw in ["文件", "保存", "创建", ".txt", ".json", ".md"]
            ) and ("/" in observation or "\\" in observation):
                # 简单的文件路径提取（实际应该更智能）
                pointers["files"].append(
                    {"step_id": step.step_id, "description": observation[:100]}
                )

            # 提取URL
            if "http" in observation or "www." in observation:
                pointers["urls"].append({"step_id": step.step_id, "description": observation[:100]})

            # 提取数据输出
            if "{" in observation and "}" in observation:
                try:
                    # 尝试解析JSON数据
                    json_start = observation.index("{")
                    json_end = observation.rindex("}") + 1
                    json_str = observation[json_start:json_end]
                    data = json.loads(json_str)
                    pointers["data_outputs"].append(
                        {"step_id": step.step_id, "data_preview": str(data)[:200]}
                    )
                except (ValueError, json.JSONDecodeError):
                    pass

            # 记录工具调用结果
            if "调用工具" in step.action:
                tool_name = step.action.split("调用工具 ")[1].split("(")[0]
                if tool_name not in pointers["tool_results"]:
                    pointers["tool_results"][tool_name] = []
                pointers["tool_results"][tool_name].append(
                    {"step_id": step.step_id, "result_preview": step.observation[:100]}
                )

        return pointers

    async def _report_progress(self, message: str, level: str = "info"):
        """报告进度给进度管理器。"""

        if self.progress_manager:
            await self.progress_manager.update_progress(
                task_id=self.task_id,
                agent_id=self.actor_id,
                status=self.status.value,
                message=message,
                details={
                    "level": level,
                    "step_count": len(self.memory),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def get_current_state(self) -> dict[str, Any]:
        """获取当前状态。"""

        return {
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "steps_completed": len(self.memory),
            "tools_available": len(self.tools),
            "execution_time": (
                (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            ),
        }

    async def _is_critical_error(self, error: Exception) -> bool:
        """
        使用LLM判断是否为严重错误，需要停止执行。

        严格遵循LLM-First原则，避免硬编码关键词匹配。
        """
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "task_description": self.task_description,
            "current_step": len(self.memory),
            "recent_actions": [step.action for step in self.memory[-3:]] if self.memory else [],
        }

        prompt = f"""
Analyze whether this error is critical and requires stopping task execution:

Task Context:
- Task: {error_context["task_description"]}
- Current step: {error_context["current_step"]}
- Recent actions: {error_context["recent_actions"]}

Error Details:
- Type: {error_context["error_type"]}
- Message: {error_context["error_message"]}

Please determine:
1. Can this error be resolved through retry or alternative approaches?
2. Is this a critical error that requires manual intervention or stopping execution?
3. What is the potential impact if we continue despite this error?

Return JSON format:
{{
    "is_critical": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed analysis of why this error is/isn't critical",
    "suggested_action": "retry/skip/stop/alternative"
}}
"""

        try:
            response = await self.llm.complete(prompt)
            result = json.loads(response)

            # 记录LLM的分析结果
            logger.info(
                f"{ACTOR_LOG_PREFIX} critical_error_analysis actor={self.actor_id} "
                f"critical={result.get('is_critical', True)} "
                f"confidence={result.get('confidence', 0.0)}"
            )

            return result.get("is_critical", True)  # 默认保守策略

        except Exception as llm_error:
            # LLM调用失败时的降级策略：基于错误类型而非字符串匹配
            logger.warning(
                f"{ACTOR_LOG_PREFIX} llm_critical_analysis_failed actor={self.actor_id} "
                f"error={llm_error!s} fallback=type_based"
            )

            # 使用Python异常类型层次结构进行判断
            return self._fallback_critical_error_check(error)

    def _fallback_critical_error_check(self, error: Exception) -> bool:
        """
        降级策略：基于异常类型层次结构判断，避免字符串匹配。
        """
        # 系统级严重错误
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return True

        # 权限和安全相关错误
        if isinstance(error, (PermissionError, OSError)):
            return True

        # 网络和连接错误通常可以重试
        if isinstance(error, (ConnectionError, TimeoutError)):
            return False

        # 参数和值错误通常可以通过调整策略解决
        # 对于未知错误类型，采用保守策略
        return not isinstance(error, (ValueError, TypeError, AttributeError))

    def _format_memory(self) -> str:
        """
        格式化执行历史为结构化字符串。

        用于错误恢复和上下文传递，保持ReAct格式的一致性。
        """
        if not self.memory:
            return "No execution history available."

        # 只保留最近的关键步骤，避免上下文过载
        recent_steps = self.memory[-5:]  # 最近5步

        formatted_lines = []
        formatted_lines.append(f"Execution History for Task: {self.task_description}")
        formatted_lines.append(f"Total Steps: {len(self.memory)}")
        formatted_lines.append("Recent Steps:")
        formatted_lines.append("-" * 50)

        for i, step in enumerate(recent_steps, 1):
            step_num = len(self.memory) - len(recent_steps) + i
            formatted_lines.append(f"Step {step_num}:")
            formatted_lines.append(f"  Thought: {step.thought}")
            formatted_lines.append(f"  Action: {step.action}")

            # 限制观察结果的长度，保持可读性
            observation = step.observation
            if len(observation) > 200:
                observation = observation[:200] + "... [truncated]"
            formatted_lines.append(f"  Observation: {observation}")
            formatted_lines.append("")  # 空行分隔

        return "\n".join(formatted_lines)

    def _build_cache_key(self, function_name: str, function_args: dict) -> str:
        """构建工具调用的缓存键。"""
        import hashlib
        import json

        # 创建包含函数名和参数的唯一键
        cache_data = {"function": function_name, "args": function_args}

        # 使用 JSON 序列化并生成哈希
        cache_string = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_string.encode("utf-8")).hexdigest()

        return f"{function_name}:{cache_hash}"
