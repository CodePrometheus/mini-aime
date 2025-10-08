"""具备实时自适应能力的动态任务规划器。"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from ..llm.base import BaseLLMClient
from .models import ExecutionStep, Task, TaskStatus, UserEvent, UserEventType
from src.config.settings import settings


logger = logging.getLogger(__name__)
PLANNER_LOG_PREFIX = "MiniAime|Planner|"
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class PlannerConfig:
    """动态规划器的行为配置。"""

    def __init__(
        self,
        enable_user_clarification: bool = False,
        enable_user_interaction: bool = False,
        max_clarification_rounds: int = 2,
        max_parallel_tasks: int = 3,
        max_task_depth: int = None,
    ):
        self.enable_user_clarification = enable_user_clarification
        self.enable_user_interaction = enable_user_interaction
        self.max_clarification_rounds = max_clarification_rounds
        self.max_parallel_tasks = max_parallel_tasks or settings.max_parallel_tasks
        self.max_task_depth = max_task_depth or settings.max_task_depth


class DynamicPlanner:
    """动态任务规划器，支持实时自适应与智能分解。"""

    def __init__(
        self, llm_client: BaseLLMClient, config: PlannerConfig | None = None, progress_manager=None
    ):
        self.llm = llm_client
        self.config = config or PlannerConfig()
        self.goal: str | None = None
        self.task_list: list[Task] = []
        self.planning_history: list[dict[str, Any]] = []
        self.progress_manager = progress_manager

    async def plan_and_dispatch(
        self,
        goal: str,
        current_tasks: list[Task],
        execution_history: list[ExecutionStep],
        user_feedback: str | None = None,
    ) -> tuple[list[Task], Task | None]:
        """
        核心方法：动态规划与任务派发。

        基于当前目标、任务状态与执行历史进行分析，生成更新后的任务计划，
        并选择下一步要执行的任务。

        Args:
            goal: 用户目标描述
            current_tasks: 当前任务列表
            execution_history: 最近的执行历史
            user_feedback: 可选的用户反馈（用于交互式规划）

        Returns:
            (更新后的任务列表, 下一步要执行的任务)
        """
        self.goal = goal
        logger.info(
            f"{PLANNER_LOG_PREFIX} plan_begin goal_len={len(goal)} tasks={len(current_tasks)} history={len(execution_history)}"
        )
        self.task_list = current_tasks

        # 发送 PLANNING 事件
        await self._emit_user_event(
            event_type=UserEventType.PLANNING,
            title="Planner 正在规划任务",
            content=f"目标: {goal}\n当前有 {len(current_tasks)} 个任务，{len(execution_history)} 步执行历史",
            collapsible=True,
        )

        # Optional user clarification (if enabled)
        if self.config.enable_user_clarification and not current_tasks:
            clarification = self._optional_quick_clarification(goal)
            if clarification:
                goal = f"{goal}\n\nAdditional context: {clarification}"

        # Interactive guidance for ambiguous situations (if enabled)
        if self.config.enable_user_interaction:
            guidance_result = await self._progressive_user_guidance(
                goal, current_tasks, execution_history, user_feedback
            )
            if guidance_result:
                goal = f"{goal}\n\n引导结果: {guidance_result}"

        # Build conversation messages for multi-round context
        messages = self._build_planning_messages(goal, current_tasks, execution_history)

        try:
            # Get LLM planning response
            planning_result = await self.llm.complete_chat_json(messages)
            logger.info(
                f"{PLANNER_LOG_PREFIX} plan_llm_ok updates={len(planning_result.get('task_updates', []))}"
            )

            # Apply task updates
            updated_tasks = self._apply_task_updates(
                current_tasks, planning_result.get("task_updates", [])
            )

            # 若存在用户触发的重规划请求（通过 user_feedback / 外部信号注入），执行子树重建
            # 约定：user_feedback 可包含 JSON，如 {"replan": {"target_task_id": "...", "hint": "...", "scope": "subtree|global"}}
            replan_spec = None
            try:
                if user_feedback and user_feedback.strip().startswith("{"):
                    parsed = json.loads(user_feedback)
                    replan_spec = parsed.get("replan")
            except Exception:
                replan_spec = None

            if replan_spec:
                updated_tasks = await self._apply_replan(updated_tasks, replan_spec)

            # Select next task to execute
            next_task = self._select_next_task(updated_tasks, planning_result.get("next_action"))

            # Record planning decision in history
            self.planning_history.append(
                {
                    "timestamp": datetime.now(),
                    "goal": goal,
                    "analysis": planning_result.get("analysis", ""),
                    "task_updates_count": len(planning_result.get("task_updates", [])),
                    "next_task_id": next_task.id if next_task else None,
                }
            )

            # 发送 TASK_UPDATE 事件
            await self._emit_task_update_event(updated_tasks)

            logger.info(
                f"{PLANNER_LOG_PREFIX} plan_end next_task={(next_task.id if next_task else None)} total_tasks={len(updated_tasks)}"
            )
            return updated_tasks, next_task

        except Exception as e:
            # Fallback to simple decomposition on LLM failure
            logger.error(f"{PLANNER_LOG_PREFIX} plan_llm_fail fallback, error={e}")
            return self._fallback_planning(goal, current_tasks), None

    async def _apply_replan(self, tasks: list[Task], replan_spec: dict) -> list[Task]:
        """根据用户提示进行子树/全局重规划，将旧子树标记为 SUPERSEDED。"""
        scope = replan_spec.get("scope", "subtree")
        target_id = replan_spec.get("target_task_id")
        hint = replan_spec.get("hint", "")

        # 标记 superseded 并递增 subtree_revision
        def mark_superseded(task_list: list[Task]) -> None:
            for t in task_list:
                t.status = TaskStatus.SUPERSEDED
                t.updated_at = datetime.now()
                t.subtree_revision = (t.subtree_revision or 0) + 1
                if t.subtasks:
                    mark_superseded(t.subtasks)

        # 选择重建范围
        if scope == "global" or not target_id:
            # 全局重建：全部标记 superseded，并基于新目标+hint 生成新的根任务
            mark_superseded(tasks)
            new_root = Task(
                id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"[Replan] {self.goal} | hint: {hint[:80]}",
                status=TaskStatus.PENDING,
                subtasks=[],
                result=None,
            )
            return [new_root]

        # 子树重建：找到目标子树，标记 superseded，并在同层插入一个新的替代子树根
        target_task = self._find_task_by_id(tasks, target_id)
        if not target_task:
            return tasks  # 找不到目标则不变

        # 定位父层，移除旧节点并插入新节点
        def replace_in_tree(task_list: list[Task]) -> bool:
            for i, t in enumerate(task_list):
                if t.id == target_id:
                    # 标记旧子树 superseded（保留在历史中不再派发，这里直接替换为新节点更清晰）
                    mark_superseded([t])
                    new_node = Task(
                        id=f"task_{uuid.uuid4().hex[:8]}",
                        description=f"[Replan Subtree] {t.description} | hint: {hint[:80]}",
                        status=TaskStatus.PENDING,
                        subtasks=[],
                        result=None,
                        subtree_revision=(t.subtree_revision or 0) + 1,
                    )
                    task_list[i] = new_node
                    return True
                if t.subtasks and replace_in_tree(t.subtasks):
                    return True
            return False

        replace_in_tree(tasks)
        return tasks

    async def plan_and_dispatch_batch(
        self,
        goal: str,
        current_tasks: list[Task],
        execution_history: list[ExecutionStep],
        max_parallel: int | None = None,
        user_feedback: str | None = None,
    ) -> tuple[list[Task], list[Task]]:
        """
        批量规划方法：识别可并行执行的多个任务。

        该方法在基础规划的基础上，进一步识别并返回可并行的任务集合，
        以提升系统吞吐量。

        Args:
            goal: 用户目标描述
            current_tasks: 当前任务列表
            execution_history: 最近的执行历史
            max_parallel: 并行任务数量上限（为 None 时使用配置默认值）

        Returns:
            (更新后的任务列表, 可并行执行的任务列表)
        """
        max_parallel = max_parallel or self.config.max_parallel_tasks

        # First, perform standard planning to get updated task list
        updated_tasks, primary_task = await self.plan_and_dispatch(
            goal, current_tasks, execution_history, user_feedback
        )

        if not primary_task:
            return updated_tasks, []

        # Find additional tasks that can be executed in parallel
        parallel_tasks = await self._identify_parallel_tasks(
            updated_tasks, primary_task, max_parallel
        )
        logger.info(
            f"{PLANNER_LOG_PREFIX} plan_batch primary={(primary_task.id if primary_task else None)} parallel={len(parallel_tasks)}"
        )

        return updated_tasks, parallel_tasks

    async def _identify_parallel_tasks(
        self, tasks: list[Task], primary_task: Task, max_parallel: int
    ) -> list[Task]:
        """
        识别可与主要任务并行执行的任务。

        使用 LLM 分析任务依赖与资源冲突，从而判断哪些任务可以安全并行。
        """
        if max_parallel <= 1:
            return [primary_task]

        # Get all pending tasks from the provided task list
        all_tasks = []

        def collect_tasks(task_list: list[Task]):
            for task in task_list:
                all_tasks.append(task)
                if task.subtasks:
                    collect_tasks(task.subtasks)

        collect_tasks(tasks)
        pending_tasks = [t for t in all_tasks if t.status == TaskStatus.PENDING]

        if len(pending_tasks) <= 1:
            return [primary_task] if primary_task in pending_tasks else []

        # Use LLM to analyze parallel execution possibilities
        try:
            parallel_analysis = await self._analyze_parallel_execution(
                pending_tasks, primary_task, max_parallel
            )

            parallel_task_ids = parallel_analysis.get("parallel_task_ids", [])

            # Find and return the identified tasks
            parallel_tasks = []
            for task_id in parallel_task_ids[:max_parallel]:
                task = self._find_task_by_id(tasks, task_id)
                if task and task.status == TaskStatus.PENDING:
                    parallel_tasks.append(task)

            # Ensure primary task is included if not already
            if primary_task not in parallel_tasks:
                parallel_tasks.insert(0, primary_task)

            logger.info(
                f"{PLANNER_LOG_PREFIX} parallel_selected count={len(parallel_tasks[:max_parallel])}"
            )
            return parallel_tasks[:max_parallel]

        except Exception:
            # Fallback: return only primary task
            logger.info(f"{PLANNER_LOG_PREFIX} parallel_fallback primary_only")
            return [primary_task]

    async def _analyze_parallel_execution(
        self, pending_tasks: list[Task], primary_task: Task, max_parallel: int
    ) -> dict:
        """使用 LLM 分析哪些任务可以并行执行。"""

        task_descriptions = []
        for task in pending_tasks:
            task_descriptions.append(f"- {task.id}: {task.description}")

        prompt = f"""
        分析哪些任务可以并行执行而不产生冲突：

        主要任务（必须包含）：{primary_task.id}: {primary_task.description}

        可用的待执行任务：
        {chr(10).join(task_descriptions)}

        考虑因素：
        1. 资源冲突（相同的外部服务、文件等）
        2. 信息依赖（一个任务需要另一个任务的输出）
        3. 逻辑序列要求
        4. 冲突操作的风险

        选择最多 {max_parallel} 个可以安全并行运行的任务。
        始终在选择中包含主要任务。

        返回JSON格式：
        {{
            "analysis": "并行执行策略的简要说明",
            "parallel_task_ids": ["task_id1", "task_id2", ...],
            "reasoning": "为什么这些任务可以并行运行",
            "excluded_tasks": {{"task_id": "排除原因", ...}}
        }}
        """

        try:
            response = await self.llm.complete(prompt)
            return json.loads(response)
        except Exception:
            # Fallback: only primary task
            return {
                "parallel_task_ids": [primary_task.id],
                "analysis": "降级处理：由于分析失败，执行单个任务",
            }

    def _build_planning_messages(
        self, goal: str, current_tasks: list[Task], execution_history: list[ExecutionStep]
    ) -> list[dict[str, str]]:
        """构建用于多轮规划上下文的对话消息。"""
        # 获取可用工具包信息（从执行历史中推断或使用默认）
        available_tool_bundles = self._extract_available_tool_bundles(execution_history)
        messages = [
            {"role": "system", "content": self._get_planner_system_prompt(available_tool_bundles)}
        ]

        # Add planning history context
        if self.planning_history:
            history_context = self._format_planning_history()
            messages.append({"role": "user", "content": f"之前的规划决策：\n{history_context}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": "我理解之前的规划上下文，会在此基础上继续规划。",
                }
            )

        # Current planning request
        current_request = self._format_planning_request(goal, current_tasks, execution_history)
        messages.append({"role": "user", "content": current_request})

        return messages

    def _extract_available_tool_bundles(self, execution_history: list[ExecutionStep]) -> list[str]:
        """从执行历史中提取可用的工具包信息。"""
        # 默认工具包
        default_bundles = [
            "web_research",
            "file_operations",
            "data_processing",
            "weather_services",
            "travel_services",
        ]

        # 从执行历史中分析实际使用的工具
        used_tools = set()
        for step in execution_history[-10:]:  # 只看最近10步
            if "brave_search" in step.action or "search" in step.action.lower():
                used_tools.add("web_research")
            elif "file" in step.action.lower():
                used_tools.add("file_operations")
            elif "weather" in step.action.lower():
                used_tools.add("weather_services")
            elif any(word in step.action.lower() for word in ["currency", "timezone", "holiday"]):
                used_tools.add("travel_services")

        return list(used_tools) if used_tools else default_bundles

    def _format_tool_bundles_description(self, tool_bundles: list[str]) -> str:
        """格式化工具包描述。"""
        descriptions = {
            "web_research": "- web_research: 网络搜索、内容提取、信息收集",
            "file_operations": "- file_operations: 文件读写、目录操作、文档管理",
            "data_processing": "- data_processing: JSON解析、数据转换、结构化处理",
            "weather_services": "- weather_services: 实时天气查询、气象信息",
            "travel_services": "- travel_services: 货币转换、时区查询、节假日信息",
            "code_execution": "- code_execution: Python代码执行、脚本运行、计算任务",
        }

        return "\n".join(
            [descriptions.get(bundle, f"- {bundle}: 专用工具包") for bundle in tool_bundles]
        )

    def _get_planner_system_prompt(self, available_tool_bundles: list[str] | None = None) -> str:
        """生成动态规划器的系统提示。"""

        # 获取当前 session_id
        session_id = (
            getattr(self.progress_manager, "session_id", "current")
            if self.progress_manager
            else "current"
        )

        # 构建工具能力描述
        tool_capabilities = ""
        if available_tool_bundles:
            tool_capabilities = f"""

当前可用工具能力：
{self._format_tool_bundles_description(available_tool_bundles)}

任务拆分时请考虑：
- 优先使用多样化的工具组合，避免只依赖网络搜索
- 根据任务性质选择合适的工具类型
- 文件操作、数据处理、代码执行等工具同样重要"""

        return f"""你是一个动态任务规划器，能够基于实时反馈自适应地分解目标。

核心原则：
1. 动态维护任务树：可以合并、删除、重排任务，不要把初始分解当作固定蓝图
2. 从宽泛开始，基于执行发现进行细化
3. 学习到新信息时生成新任务
4. 移除/修改不再合理的任务
5. **⚠️ 任务数量严格控制**：总任务数不得超过15个，单次添加不得超过5个任务
6. **⚠️ 优先合并策略**：当任务数量接近15个时，必须合并相似任务而不是添加新任务
7. 基于依赖关系和约束确定优先级
8. 保持任务具体可执行
9. 尽可能支持并行执行{tool_capabilities}

**⚠️ 任务状态管理规则（重要）**：
- **禁止将有未完成子任务的父任务标记为 completed**
- 只有当一个任务的所有子任务都完成后，才能将其标记为 completed
- 如果不确定子任务完成情况，不要修改父任务状态
- 根任务通常不需要手动标记完成，系统会自动处理

**文件保存规范（强制要求）**：
- 当前 Session ID: `{session_id}`
- 所有使用 write_file 保存的文件，路径必须以 `{session_id}/` 开头
- 这确保不同任务执行的文件完全隔离，不会互相干扰
- 示例：
  - ✅ 正确：write_file("{session_id}/kafka_basics.md", content)
  - ✅ 正确：write_file("{session_id}/temp/data.json", content)
  - ❌ 错误：write_file("kafka_basics.md", content) - 缺少 session_id 前缀
  - ❌ 错误：write_file("temp/data.json", content) - 缺少 session_id 前缀
- 读取文件时也要使用相同的 session_id 前缀
- 这样可以防止后续任务读取到之前任务的文件

交付物与工件要求：
- 规划中必须显式包含"生成最终报告文件"的任务节点（如：生成 Markdown 报告）
- **最终报告任务描述必须明确包含以下步骤**：
  1. **数据整合阶段**：优先使用 integrate_research 工具智能整合所有研究文件，生成包含完整研究数据的初步报告
  2. **智能分析阶段**：如果 integrate_research 工具不可用，则使用 list_directory 工具列出 '{session_id}/' 目录下的所有文件，识别研究文件（排除 final_report_*.md 文件）
  3. **内容读取**：使用 read_file 工具逐个读取所有研究文件（.md 和 .json 文件）的内容
  4. **LLM深度分析**：使用LLM分析并整合所有子任务的研究成果和数据，提取关键信息，提供洞察和建议
  5. **结构化输出**：如果原始目标包含时间规划（如"X周""X个月"），必须生成按时间分阶段的详细计划
  6. **报告保存**：使用 write_file 保存综合报告到 '{session_id}/final_report.md'（文件名必须是 final_report.md）
  7. **质量保证**：确保最终报告包含具体的研究数据（价格、时间、地点等），而不是通用模板
- 最终报告需汇总主要发现、关键结论与可操作建议
- 若执行中已产出数据/链接/文件，需在报告中引用并在结果中返回文件路径

层级限制：
- 任务树的最大深度为 {self.config.max_task_depth}（根任务计为深度1）。
- 不要创建超过该深度的子任务；如需更多步骤，请创建同层任务或将任务提升为根层。

分解策略：
- **适度分解原则**：优先创建3-5个核心任务，避免过度细化
- **合并优先**：相似任务（如多个城市研究）应合并为一个综合任务
- **核心优先**：优先创建预算、签证、交通、住宿等核心任务
- **避免重复**：不要为每个城市单独创建研究任务，应创建"研究意大利主要城市"的综合任务
- 约束驱动：让发现的约束驱动新任务生成
- 失败自适应：失败的任务产生替代方法

**任务分解示例**：
❌ 错误做法（过度细化）：
- 研究罗马景点
- 研究米兰景点  
- 研究佛罗伦萨景点
- 研究威尼斯景点
- 研究那不勒斯景点

✅ 正确做法（适度合并）：
- 研究意大利主要城市和景点（包含罗马、米兰、佛罗伦萨、威尼斯等）
- 估算三个月旅行预算（包含住宿、餐饮、交通、活动费用）
- 了解签证和入境要求

输出格式：
返回JSON格式：
{{
    "analysis": "对当前情况的分析和变化说明",
    "task_updates": [
        {{"action": "add", "parent_id": "task_id", "task": {{...}}}},
        {{"action": "modify", "task_id": "task_id", "changes": {{...}}}},
        {{"action": "remove", "task_id": "task_id", "reason": "..."}}
    ],
    "next_action": {{
        "task_id": "task_id",
        "reasoning": "为什么应该下一步执行这个任务"
    }},
    "parallel_opportunities": ["task_id1", "task_id2"],
    "should_print_task_tree": true,
    "task_tree_summary": "任务树的简要状态描述（可选）"
}}

任务对象结构：
{{
    "id": "unique_task_id",
    "description": "清晰可执行的任务描述",
    "status": "pending|in_progress|completed|failed",
    "subtasks": [...], // 可选的子任务
    "result": null, // 执行后填充
    "created_at": "ISO timestamp",
    "updated_at": "ISO timestamp"
}}"""

    def _format_planning_request(
        self, goal: str, current_tasks: list[Task], execution_history: list[ExecutionStep]
    ) -> str:
        """格式化当前的规划请求文本。"""
        task_summary = self._format_task_tree(current_tasks)
        history_summary = self._format_execution_history(execution_history[-5:])  # Last 5 steps

        return f"""动态规划请求

目标：{goal}

当前任务树：
{task_summary}

最近执行历史：
{history_summary}

需要分析的问题：
- 我们从执行中学到了什么新信息？
- 哪些假设是错误的或需要更新？
- 出现了什么新的机会或约束？
- 基于依赖关系，哪些任务应该优先处理？

交付物约束：
- 请确保在任务树中加入"生成最终报告文件（Markdown）并保存到 docs/ 目录"的任务节点
- **关键要求**：最终报告任务的描述必须明确包含以下执行步骤：
  步骤1: 使用 list_directory 工具列出 'temp/' 目录下的所有文件
  步骤2: 使用 read_file 工具逐个读取所有找到的 .md 和 .json 文件
  步骤3: 整合所有文件内容，生成综合报告
  步骤4: 如果原始目标包含时间要求（如"X周""X个月"学习计划），必须生成按时间分阶段的详细计划表
  步骤5: 使用 write_file 保存最终报告到 docs/ 目录
- 该报告应汇总所有子任务的研究成果、结论与关键证据

请提供你的分析和任务更新。"""

    def _format_task_tree(self, tasks: list[Task], indent: int = 0) -> str:
        """以可读文本格式化任务树。"""
        if not tasks:
            return "暂无任务 - 需要初始分解"

        lines = []
        for task in tasks:
            prefix = "  " * indent
            status_icon = {
                TaskStatus.PENDING: "[ ]",
                TaskStatus.IN_PROGRESS: "[~]",
                TaskStatus.COMPLETED: "[x]",
                TaskStatus.FAILED: "[!]",
            }.get(task.status, "[ ]")

            lines.append(f"{prefix}{status_icon} {task.description} (ID: {task.id})")

            if task.subtasks:
                lines.append(self._format_task_tree(task.subtasks, indent + 1))

        return "\n".join(lines)

    def _format_execution_history(self, history: list[ExecutionStep]) -> str:
        """格式化执行历史以便上下文理解。"""
        if not history:
            return "暂无执行历史"

        lines = []
        for i, step in enumerate(history, 1):
            lines.append(f"步骤 {i}:")
            lines.append(f"  思考: {step.thought}")
            lines.append(f"  行动: {step.action}")
            lines.append(f"  观察: {step.observation}")
            lines.append("")

        return "\n".join(lines)

    def _format_planning_history(self) -> str:
        """格式化规划历史以便上下文理解。"""
        if not self.planning_history:
            return "暂无之前的规划决策"

        lines = []
        for i, entry in enumerate(self.planning_history[-3:], 1):  # Last 3 decisions
            lines.append(f"决策 {i} ({entry['timestamp'].strftime('%H:%M:%S')}):")
            lines.append(f"  目标: {entry['goal']}")
            lines.append(f"  分析: {entry['analysis']}")
            lines.append(f"  更新: {entry['task_updates_count']} 个任务变更")
            lines.append("")

        return "\n".join(lines)

    def _apply_task_updates(self, current_tasks: list[Task], updates: list[dict]) -> list[Task]:
        """将 LLM 生成的任务更新应用到任务树。"""
        # 任务数量限制检查
        MAX_TOTAL_TASKS = 15
        MAX_SINGLE_UPDATE_TASKS = 5
        
        # 计算当前总任务数（包括子任务）
        def count_all_tasks(tasks: list[Task]) -> int:
            total = 0
            for task in tasks:
                total += 1
                if task.subtasks:
                    total += count_all_tasks(task.subtasks)
            return total
        
        current_total = count_all_tasks(current_tasks)
        
        # 计算本次更新要添加的任务数
        add_updates = [u for u in updates if u.get("action") == "add"]
        
        # 硬性限制：单次添加不得超过5个任务
        if len(add_updates) > MAX_SINGLE_UPDATE_TASKS:
            logger.warning(f"{PLANNER_LOG_PREFIX} task_limit_exceeded single_update={len(add_updates)} max={MAX_SINGLE_UPDATE_TASKS}")
            add_updates = add_updates[:MAX_SINGLE_UPDATE_TASKS]
            updates = [u for u in updates if u.get("action") != "add"] + add_updates
        
        # 硬性限制：总任务数不得超过15个
        if current_total + len(add_updates) > MAX_TOTAL_TASKS:
            allowed_additions = max(0, MAX_TOTAL_TASKS - current_total)
            if allowed_additions == 0:
                logger.warning(f"{PLANNER_LOG_PREFIX} task_limit_exceeded total={current_total} max={MAX_TOTAL_TASKS}, rejecting all additions")
                updates = [u for u in updates if u.get("action") != "add"]
            else:
                logger.warning(f"{PLANNER_LOG_PREFIX} task_limit_partial total={current_total} additions={len(add_updates)} allowed={allowed_additions}")
                # 优先保留最重要的任务（通过任务描述长度和关键词判断）
                prioritized_updates = self._prioritize_task_additions(add_updates, allowed_additions)
                updates = [u for u in updates if u.get("action") != "add"] + prioritized_updates
        
        # Create a working copy
        task_dict = self._build_task_dict(current_tasks)
        updated_tasks = current_tasks.copy()

        for update in updates:
            action = update.get("action")

            if action == "add":
                new_task = self._create_task_from_dict(update.get("task", {}))
                parent_id = update.get("parent_id")

                if parent_id and parent_id in task_dict:
                    # 深度校验：若超出最大深度，则挂到根层
                    parent_depth = self._find_task_depth(updated_tasks, parent_id)
                    max_depth = max(self.config.max_task_depth, 1)

                    if parent_depth is None:
                        updated_tasks.append(new_task)
                    else:
                        # 根为1层，因此新子任务的目标深度 = parent_depth + 1
                        if parent_depth + 1 > max_depth:
                            updated_tasks.append(new_task)
                        else:
                            parent_task = task_dict[parent_id]
                            parent_task.subtasks.append(new_task)
                else:
                    # Add as root task
                    updated_tasks.append(new_task)

                task_dict[new_task.id] = new_task

            elif action == "modify":
                task_id = update.get("task_id")
                if task_id in task_dict:
                    task = task_dict[task_id]
                    changes = update.get("changes", {})

                    # Apply changes
                    if "status" in changes:
                        new_status = TaskStatus(changes["status"])

                        # 校验：如果要将任务标记为完成，检查是否有未完成的子任务
                        if new_status == TaskStatus.COMPLETED and task.subtasks:
                            incomplete_subtasks = [
                                st
                                for st in task.subtasks
                                if st.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                            ]
                            if incomplete_subtasks:
                                logger.warning(
                                    f"{PLANNER_LOG_PREFIX} invalid_status_change task={task_id} "
                                    f"attempted_status=completed but has {len(incomplete_subtasks)} incomplete subtasks"
                                )
                                # 不应用这个状态变更
                            else:
                                task.status = new_status
                        else:
                            task.status = new_status

                    if "description" in changes:
                        task.description = changes["description"]

                    task.updated_at = datetime.now()

            elif action == "remove":
                task_id = update.get("task_id")
                if task_id in task_dict:
                    self._remove_task_from_tree(updated_tasks, task_id)

        return updated_tasks

    def _find_task_depth(self, tasks: list[Task], target_id: str, depth: int = 1) -> int | None:
        """在任务树中查找目标任务的深度（根为1）。"""
        for task in tasks:
            if task.id == target_id:
                return depth
            if task.subtasks:
                found = self._find_task_depth(task.subtasks, target_id, depth + 1)
                if found is not None:
                    return found
        return None

    def _build_task_dict(self, tasks: list[Task]) -> dict[str, Task]:
        """构建任务字典以便快速查找。"""
        task_dict = {}

        def add_tasks(task_list: list[Task]):
            for task in task_list:
                task_dict[task.id] = task
                if task.subtasks:
                    add_tasks(task.subtasks)

        add_tasks(tasks)
        return task_dict

    def _create_task_from_dict(self, task_data: dict) -> Task:
        """根据字典数据创建 Task 对象。"""
        return Task(
            id=task_data.get("id", f"task_{uuid.uuid4().hex[:8]}"),
            description=task_data.get("description", ""),
            status=TaskStatus(task_data.get("status", "pending")),
            subtasks=[],
            result=task_data.get("result"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def _remove_task_from_tree(self, tasks: list[Task], task_id: str) -> bool:
        """从任务树中移除指定任务。"""
        for i, task in enumerate(tasks):
            if task.id == task_id:
                tasks.pop(i)
                return True
            if task.subtasks and self._remove_task_from_tree(task.subtasks, task_id):
                return True
        return False

    def _select_next_task(self, tasks: list[Task], next_action: dict | None) -> Task | None:
        """基于 LLM 的推荐选择下一步要执行的任务。"""
        if not next_action:
            return self._find_first_pending_task(tasks)

        # Try to find the recommended task
        recommended_task_id = next_action.get("task_id")
        if recommended_task_id:
            task = self._find_task_by_id(tasks, recommended_task_id)
            if task and task.status == TaskStatus.PENDING:
                return task

        # Fallback to first pending task
        return self._find_first_pending_task(tasks)

    def _find_task_by_id(self, tasks: list[Task], task_id: str) -> Task | None:
        """在任务树中按照 ID 查找任务。"""
        for task in tasks:
            if task.id == task_id:
                return task
            if task.subtasks:
                found = self._find_task_by_id(task.subtasks, task_id)
                if found:
                    return found
        return None

    def _find_first_pending_task(self, tasks: list[Task]) -> Task | None:
        """在任务树中按深度优先查找第一个待执行任务。"""
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                return task
            if task.subtasks:
                found = self._find_first_pending_task(task.subtasks)
                if found:
                    return found
        return None

    def _optional_quick_clarification(self, goal: str) -> str | None:
        """可选的快速澄清（在启用时与用户交互）。"""
        if not self.config.enable_user_clarification:
            return None

        # For now, return None - actual user interaction would be implemented
        # in the main application layer
        return None

    async def _progressive_user_guidance(
        self,
        goal: str,
        current_tasks: list[Task],
        execution_history: list[ExecutionStep],
        user_feedback: str | None = None,
    ) -> str | None:
        """
        通过温和提问进行渐进式用户引导。

        分析当前局面以识别模糊点，并通过策略性问题引导用户补充更具体的信息。
        """
        if not self.config.enable_user_interaction:
            return None

        # Analyze current situation for ambiguities
        ambiguity_analysis = await self._analyze_ambiguities(goal, current_tasks, execution_history)

        if not ambiguity_analysis.get("needs_guidance", False):
            return user_feedback  # No guidance needed, just return user feedback

        # Generate guiding questions based on ambiguities
        guiding_questions = await self._generate_guiding_questions(
            goal, ambiguity_analysis, user_feedback
        )

        if not guiding_questions:
            return user_feedback

        # In a real implementation, this would interact with the user
        # For now, we simulate the guidance process
        guidance_result = await self._simulate_user_guidance(guiding_questions)

        return guidance_result

    async def _analyze_ambiguities(
        self, goal: str, current_tasks: list[Task], execution_history: list[ExecutionStep]
    ) -> dict[str, Any]:
        """分析当前局面以识别需要澄清的模糊点。"""

        task_summary = self._format_task_tree(current_tasks)
        history_summary = self._format_execution_history(execution_history[-3:])

        analysis_prompt = f"""分析当前情况，识别需要用户澄清的模糊方面：

目标：{goal}
当前任务状态：
{task_summary}

最近执行历史：
{history_summary}

请分析以下方面：
1. 目标是否具体明确？
2. 是否有多种可能的解释？
3. 是否缺少关键约束或偏好？
4. 执行中是否遇到需要用户决策的情况？

返回JSON格式：
{{
    "needs_guidance": true/false,
    "ambiguous_aspects": [
        {{
            "aspect": "时间约束",
            "description": "用户没有明确时间要求",
            "priority": "high/medium/low"
        }}
    ],
    "suggested_questions": [
        "您对时间有什么要求吗？",
        "您有特别的偏好或约束吗？"
    ]
}}"""

        try:
            response = await self.llm.complete(analysis_prompt)
            return json.loads(response)
        except Exception:
            return {"needs_guidance": False}

    async def _generate_guiding_questions(
        self, goal: str, ambiguity_analysis: dict[str, Any], user_feedback: str | None = None
    ) -> list[str]:
        """基于模糊点分析生成策略性引导问题。"""

        if not ambiguity_analysis.get("needs_guidance", False):
            return []

        ambiguous_aspects = ambiguity_analysis.get("ambiguous_aspects", [])
        suggested_questions = ambiguity_analysis.get("suggested_questions", [])

        # Focus on high priority aspects first
        high_priority_aspects = [
            aspect for aspect in ambiguous_aspects if aspect.get("priority") == "high"
        ]

        if not high_priority_aspects:
            return suggested_questions[:2]  # Limit to 2 questions max

        # Generate contextual questions
        question_prompt = f"""基于以下分析，生成1-2个循循善诱的引导问题：

原始目标：{goal}
用户反馈：{user_feedback or "无"}

需要澄清的高优先级方面：
{json.dumps(high_priority_aspects, ensure_ascii=False, indent=2)}

要求：
1. 问题要自然、不突兀
2. 引导用户思考但不强迫回答
3. 一次最多问2个相关问题
4. 用温和、专业的语气

返回JSON格式：
{{
    "questions": [
        "为了给您制定更合适的方案，我想了解一下您的时间安排是怎样的？",
        "您有什么特别想要体验或者需要避免的吗？"
    ]
}}"""

        try:
            response = await self.llm.complete(question_prompt)
            result = json.loads(response)
            return result.get("questions", [])
        except Exception:
            return suggested_questions[:2]

    async def _simulate_user_guidance(self, questions: list[str]) -> str | None:
        """
        模拟用户引导交互。

        实际实现中应：
        1. 通过界面向用户展示问题
        2. 等待用户回答
        3. 将回答综合为可执行的指导信息

        目前返回 None，表示未获得额外指导。
        """
        # TODO: Implement actual user interaction in the application layer
        # This could be through:
        # - WebSocket for real-time chat
        # - HTTP endpoints for question/answer flow
        # - Event-driven architecture with user response callbacks

        return None

    def _fallback_planning(self, goal: str, current_tasks: list[Task]) -> list[Task]:
        """LLM 失败时的简易降级规划。"""
        if current_tasks:
            return current_tasks

        # Create basic initial task decomposition
        return [
            Task(
                id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Analyze and break down: {goal}",
                status=TaskStatus.PENDING,
                subtasks=[],
                result=None,
            )
        ]

    def get_current_state(self) -> dict[str, Any]:
        """获取当前规划器状态（用于监控）。"""
        return {
            "goal": self.goal,
            "task_count": len(self.task_list),
            "completed_tasks": len(
                [t for t in self._get_all_tasks() if t.status == TaskStatus.COMPLETED]
            ),
            "pending_tasks": len(
                [t for t in self._get_all_tasks() if t.status == TaskStatus.PENDING]
            ),
            "planning_decisions": len(self.planning_history),
        }

    def _get_all_tasks(self) -> list[Task]:
        """从任务树拉平成列表以获取全部任务。"""
        all_tasks = []

        def collect_tasks(task_list: list[Task]):
            for task in task_list:
                all_tasks.append(task)
                if task.subtasks:
                    collect_tasks(task.subtasks)

        collect_tasks(self.task_list)
        return all_tasks

    async def _emit_user_event(
        self,
        event_type: UserEventType,
        title: str,
        content: str,
        level: str = "info",
        collapsible: bool = False,
        details: dict[str, Any] | None = None,
    ):
        """发送用户事件到 ProgressManager"""
        if not self.progress_manager:
            return

        user_event = UserEvent(
            event_type=event_type,
            title=title,
            content=content,
            timestamp=datetime.now(),
            level=level,
            collapsible=collapsible,
            details=details,
        )

        await self.progress_manager.emit_user_event(user_event)

    async def _emit_task_update_event(self, tasks: list[Task]):
        """发送任务树更新事件"""
        if not self.progress_manager:
            return

        # 统计任务状态
        all_tasks = []

        def collect(task_list):
            for t in task_list:
                all_tasks.append(t)
                if t.subtasks:
                    collect(t.subtasks)

        collect(tasks)

        total = len(all_tasks)
        completed = len([t for t in all_tasks if t.status == TaskStatus.COMPLETED])
        in_progress = len([t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS])
        pending = len([t for t in all_tasks if t.status == TaskStatus.PENDING])
        failed = len([t for t in all_tasks if t.status == TaskStatus.FAILED])

        # 将任务转换为 JSON 格式供前端 TaskTree 组件使用
        task_data = [self._task_to_dict(task) for task in tasks]
        
        # 格式化任务列表为易读的 Markdown 格式供事件列表显示
        markdown_content = self._format_tasks_for_display(
            tasks, all_tasks, completed, in_progress, pending, failed
        )

        await self._emit_user_event(
            event_type=UserEventType.TASK_UPDATE,
            title=f"任务列表已更新 [{completed}/{total}]",
            content=json.dumps(task_data, ensure_ascii=False),
            level="info",
            collapsible=True,
            details={"markdown_content": markdown_content}
        )

    def _task_to_dict(self, task: Task) -> dict:
        """将 Task 对象转换为字典"""
        return {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "subtasks": [self._task_to_dict(st) for st in task.subtasks] if task.subtasks else [],
        }

    def _format_tasks_for_display(
        self,
        tasks: list[Task],
        all_tasks: list[Task],
        completed: int,
        in_progress: int,
        pending: int,
        failed: int,
    ) -> str:
        """将任务列表格式化为易读的 Markdown 格式"""
        lines = []

        # 添加总体统计
        progress_bar = self._create_progress_bar(completed, len(all_tasks))
        lines.append(f"**总体进度：** {progress_bar} {completed}/{len(all_tasks)}")
        lines.append("")

        # 添加状态统计
        status_parts = []
        if completed > 0:
            status_parts.append(f"✅ 已完成: {completed}")
        if in_progress > 0:
            status_parts.append(f"⏳ 进行中: {in_progress}")
        if pending > 0:
            status_parts.append(f"📝 待处理: {pending}")
        if failed > 0:
            status_parts.append(f"❌ 失败: {failed}")

        if status_parts:
            lines.append(" | ".join(status_parts))
            lines.append("")

        # 分组显示任务
        lines.append("---")
        lines.append("")

        # 显示进行中的任务
        in_progress_tasks = [t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]
        if in_progress_tasks:
            lines.append("### ⏳ 进行中")
            for task in in_progress_tasks:
                lines.append(f"- **{task.description}**")
                if task.subtasks:
                    for subtask in task.subtasks:
                        status_icon = self._get_status_icon(subtask.status)
                        lines.append(f"  - {status_icon} {subtask.description}")
            lines.append("")

        # 显示待处理的任务
        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        if pending_tasks:
            lines.append("### 📝 待处理")
            for i, task in enumerate(pending_tasks, 1):
                lines.append(f"{i}. {task.description}")
                if task.subtasks:
                    for subtask in task.subtasks:
                        status_icon = self._get_status_icon(subtask.status)
                        lines.append(f"   - {status_icon} {subtask.description}")
            lines.append("")

        # 显示已完成的任务（折叠显示）
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        if completed_tasks:
            lines.append("### ✅ 已完成")
            for task in completed_tasks:
                lines.append(f"- ~~{task.description}~~")
            lines.append("")

        # 显示失败的任务
        failed_tasks = [t for t in all_tasks if t.status == TaskStatus.FAILED]
        if failed_tasks:
            lines.append("### ❌ 失败")
            for task in failed_tasks:
                lines.append(f"- **{task.description}**")
                if task.blocked_reason:
                    lines.append(f"  - 原因: {task.blocked_reason}")
            lines.append("")

        return "\n".join(lines)

    def _get_status_icon(self, status: TaskStatus) -> str:
        """根据任务状态返回对应的图标"""
        icon_map = {
            TaskStatus.COMPLETED: "✅",
            TaskStatus.IN_PROGRESS: "⏳",
            TaskStatus.PENDING: "📝",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.SUPERSEDED: "⏭️",
        }
        return icon_map.get(status, "📌")

    def _create_progress_bar(self, completed: int, total: int, length: int = 20) -> str:
        """创建进度条"""
        if total == 0:
            return "░" * length

        filled = int(length * completed / total)
        bar = "█" * filled + "░" * (length - filled)
        percentage = int(100 * completed / total)
        return f"{bar} {percentage}%"

    def _prioritize_task_additions(self, add_updates: list[dict], max_allowed: int) -> list[dict]:
        """优先保留最重要的任务，避免重复研究"""
        if len(add_updates) <= max_allowed:
            return add_updates
        
        # 任务优先级评分 - 基于任务特征而非关键词
        def score_task(update: dict) -> float:
            task = update.get("task", {})
            description = task.get("description", "")
            
            # 基础分数
            score = 0.0
            
            # 1. 任务描述长度 - 更具体详细的任务优先级更高
            score += min(len(description) / 50, 2.0)
            
            # 2. 任务复杂度 - 包含多个步骤的任务优先级更高
            complexity_indicators = ["和", "以及", "包括", "涵盖", "整合", "汇总"]
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in description)
            score += min(complexity_score * 0.5, 1.5)
            
            # 3. 任务重要性 - 基于任务ID和描述特征
            task_id = task.get("id", "").lower()
            
            # 最终报告和汇总任务优先级最高
            if any(keyword in task_id for keyword in ["final", "summary", "report", "汇总", "最终"]):
                score += 3.0
            
            # 核心规划任务优先级高
            elif any(keyword in task_id for keyword in ["budget", "visa", "transport", "accommodation", "预算", "签证", "交通", "住宿"]):
                score += 2.0
            
            # 4. 避免过度细分 - 单一城市/地点的任务优先级较低
            # 通过任务描述长度和复杂度判断是否为过度细分
            if len(description) < 30 and complexity_score == 0:
                score -= 1.0
            
            return score
        
        # 按分数排序
        scored_updates = [(score_task(update), update) for update in add_updates]
        scored_updates.sort(key=lambda x: x[0], reverse=True)
        
        # 返回前N个最重要的任务
        prioritized = [update for _, update in scored_updates[:max_allowed]]
        
        logger.info(f"{PLANNER_LOG_PREFIX} task_prioritized from={len(add_updates)} to={len(prioritized)}")
        return prioritized
