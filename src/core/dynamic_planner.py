"""具备实时自适应能力的动态任务规划器。"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from ..llm.base import BaseLLMClient
from .models import ExecutionStep, Task, TaskStatus


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
        max_task_depth: int = 4,
    ):
        self.enable_user_clarification = enable_user_clarification
        self.enable_user_interaction = enable_user_interaction
        self.max_clarification_rounds = max_clarification_rounds
        self.max_parallel_tasks = max_parallel_tasks
        self.max_task_depth = max_task_depth


class DynamicPlanner:
    """动态任务规划器，支持实时自适应与智能分解。"""

    def __init__(self, llm_client: BaseLLMClient, config: PlannerConfig | None = None):
        self.llm = llm_client
        self.config = config or PlannerConfig()
        self.goal: str | None = None
        self.task_list: list[Task] = []
        self.planning_history: list[dict[str, Any]] = []

    async def plan_and_dispatch(
        self, goal: str, current_tasks: list[Task], execution_history: list[ExecutionStep],
        user_feedback: str | None = None
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
        logger.info(f"{PLANNER_LOG_PREFIX} plan_begin goal_len={len(goal)} tasks={len(current_tasks)} history={len(execution_history)}")
        self.task_list = current_tasks

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
            response = await self.llm.complete_with_context(messages)
            planning_result = json.loads(response)
            logger.info(f"{PLANNER_LOG_PREFIX} plan_llm_ok updates={len(planning_result.get('task_updates', []))}")

            # Apply task updates
            updated_tasks = self._apply_task_updates(
                current_tasks, planning_result.get("task_updates", [])
            )

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

            logger.info(f"{PLANNER_LOG_PREFIX} plan_end next_task={(next_task.id if next_task else None)} total_tasks={len(updated_tasks)}")
            return updated_tasks, next_task

        except Exception:
            # Fallback to simple decomposition on LLM failure
            logger.error(f"{PLANNER_LOG_PREFIX} plan_llm_fail fallback")
            return self._fallback_planning(goal, current_tasks), None

    async def plan_and_dispatch_batch(
        self, 
        goal: str, 
        current_tasks: list[Task], 
        execution_history: list[ExecutionStep],
        max_parallel: int | None = None,
        user_feedback: str | None = None
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
        logger.info(f"{PLANNER_LOG_PREFIX} plan_batch primary={(primary_task.id if primary_task else None)} parallel={len(parallel_tasks)}")
        
        return updated_tasks, parallel_tasks

    async def _identify_parallel_tasks(
        self, 
        tasks: list[Task], 
        primary_task: Task, 
        max_parallel: int
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
                
            logger.info(f"{PLANNER_LOG_PREFIX} parallel_selected count={len(parallel_tasks[:max_parallel])}")
            return parallel_tasks[:max_parallel]
            
        except Exception:
            # Fallback: return only primary task
            logger.info(f"{PLANNER_LOG_PREFIX} parallel_fallback primary_only")
            return [primary_task]

    async def _analyze_parallel_execution(
        self, 
        pending_tasks: list[Task], 
        primary_task: Task, 
        max_parallel: int
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
                "analysis": "降级处理：由于分析失败，执行单个任务"
            }

    def _build_planning_messages(
        self, goal: str, current_tasks: list[Task], execution_history: list[ExecutionStep]
    ) -> list[dict[str, str]]:
        """构建用于多轮规划上下文的对话消息。"""
        messages = [{"role": "system", "content": self._get_planner_system_prompt()}]

        # Add planning history context
        if self.planning_history:
            history_context = self._format_planning_history()
            messages.append(
                {"role": "user", "content": f"之前的规划决策：\n{history_context}"}
            )
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

    def _get_planner_system_prompt(self) -> str:
        """生成动态规划器的系统提示。"""
        return """你是一个动态任务规划器，能够基于实时反馈自适应地分解目标。

核心原则：
1. 从宽泛开始，基于执行发现进行细化
2. 学习到新信息时生成新任务
3. 移除/修改不再合理的任务
4. 基于依赖关系和约束确定优先级
5. 保持任务具体可执行
6. 尽可能支持并行执行

分解策略：
- 广度优先探索：从高层开始，逐步细化
- 约束驱动：让发现的约束驱动新任务生成
- 失败自适应：失败的任务产生替代方法

输出格式：
返回JSON格式：
{
    "analysis": "对当前情况的分析和变化说明",
    "task_updates": [
        {"action": "add", "parent_id": "task_id", "task": {...}},
        {"action": "modify", "task_id": "task_id", "changes": {...}},
        {"action": "remove", "task_id": "task_id", "reason": "..."}
    ],
    "next_action": {
        "task_id": "task_id",
        "reasoning": "为什么应该下一步执行这个任务"
    },
    "parallel_opportunities": ["task_id1", "task_id2"]
}

任务对象结构：
{
    "id": "unique_task_id",
    "description": "清晰可执行的任务描述",
    "status": "pending|in_progress|completed|failed",
    "subtasks": [...], // 可选的子任务
    "result": null, // 执行后填充
    "created_at": "ISO timestamp",
    "updated_at": "ISO timestamp"
}"""

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
        # Create a working copy
        task_dict = self._build_task_dict(current_tasks)
        updated_tasks = current_tasks.copy()

        for update in updates:
            action = update.get("action")

            if action == "add":
                new_task = self._create_task_from_dict(update.get("task", {}))
                parent_id = update.get("parent_id")

                if parent_id and parent_id in task_dict:
                    # Add as subtask
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
                        task.status = TaskStatus(changes["status"])
                    if "description" in changes:
                        task.description = changes["description"]

                    task.updated_at = datetime.now()

            elif action == "remove":
                task_id = update.get("task_id")
                if task_id in task_dict:
                    self._remove_task_from_tree(updated_tasks, task_id)

        return updated_tasks

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
        user_feedback: str | None = None
    ) -> str | None:
        """
        通过温和提问进行渐进式用户引导。

        分析当前局面以识别模糊点，并通过策略性问题引导用户补充更具体的信息。
        """
        if not self.config.enable_user_interaction:
            return None
            
        # Analyze current situation for ambiguities
        ambiguity_analysis = await self._analyze_ambiguities(
            goal, current_tasks, execution_history
        )
        
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
        self,
        goal: str,
        current_tasks: list[Task], 
        execution_history: list[ExecutionStep]
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
        self,
        goal: str,
        ambiguity_analysis: dict[str, Any],
        user_feedback: str | None = None
    ) -> list[str]:
        """基于模糊点分析生成策略性引导问题。"""
        
        if not ambiguity_analysis.get("needs_guidance", False):
            return []
            
        ambiguous_aspects = ambiguity_analysis.get("ambiguous_aspects", [])
        suggested_questions = ambiguity_analysis.get("suggested_questions", [])
        
        # Focus on high priority aspects first
        high_priority_aspects = [
            aspect for aspect in ambiguous_aspects 
            if aspect.get("priority") == "high"
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
