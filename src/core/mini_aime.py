"""MiniAime 主控制器 - 协调动态多智能体系统的核心组件。

该模块实现了论文《Aime: Towards Fully-Autonomous Multi-Agent Framework》
中描述的6步工作流程，协调 DynamicPlanner、ActorFactory、DynamicActor 
和 ProgressManager 四个核心组件的协同工作。
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from ..llm.base import BaseLLMClient
from ..config.settings import settings
from .actor_factory import ActorFactory, TaskSpecification
from .dynamic_actor import DynamicActor
from .dynamic_planner import DynamicPlanner, PlannerConfig
from .models import (
    ExecutionStep,
    SystemState,
    Task,
    TaskStatus,
)
from .progress_manager import ProgressManager


logger = logging.getLogger(__name__)
CTRL_LOG_PREFIX = "MiniAime|Controller|"
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class MiniAimeConfig:
    """MiniAime 系统配置。"""

    def __init__(
        self,
        max_parallel_agents: int = 3,
        agent_timeout: int = 300,  # 5 minutes per agent
        enable_persistence: bool = False,
        enable_auto_recovery: bool = True,
        planner_config: PlannerConfig | None = None,
        max_task_retries: int = 1,
        retry_backoff_base: int = 2,
        retry_backoff_max: int = 30,
    ):
        self.max_parallel_agents = max_parallel_agents
        self.agent_timeout = agent_timeout
        self.enable_persistence = enable_persistence
        self.enable_auto_recovery = enable_auto_recovery
        self.planner_config = planner_config or PlannerConfig()
        self.max_task_retries = max_task_retries
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max


class MiniAime:
    """
    MiniAime 主控制器。
    
    负责协调整个多智能体系统的执行，实现论文中的6步工作流程：
    1. Task Decomposition - 任务分解
    2. (Sub)Task Dispatch - 任务派发
    3. Actor Instantiation - 智能体实例化
    4. ReAct Execution - ReAct执行
    5. Progress Update - 进度更新
    6. Evaluation and Iteration - 评估与迭代
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: MiniAimeConfig | None = None,
        initialize_tools: bool = True,
    ):
        """
        初始化 MiniAime 系统。
        
        Args:
            llm_client: LLM 客户端
            config: 系统配置
            initialize_tools: 是否初始化真实工具（测试时可设为False）
        """
        self.config = config or MiniAimeConfig()

        # 初始化四个核心组件
        self.planner = DynamicPlanner(llm_client, self.config.planner_config)
        self.factory = ActorFactory(llm_client, initialize_tools)
        self.progress_manager = ProgressManager()

        # 系统状态
        self.session_id: str | None = None
        self.active_agents: dict[str, asyncio.Task] = {}
        self.completed_agents: set[str] = set()
        self.failed_agents: dict[str, Exception] = {}
        self.execution_history: list[ExecutionStep] = []
        self.max_history_size = 1000  # 限制历史记录大小
        self.task_retry_counts: dict[str, int] = defaultdict(int)

        # 并发控制
        self._agents_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        self.start_time: datetime | None = None

    async def execute_task(
        self,
        user_goal: str,
        session_id: str | None = None,
    ) -> SystemState:
        """
        执行用户任务的主循环。
        
        实现论文中的6步迭代工作流程，持续执行直到所有任务完成。
        
        Args:
            user_goal: 用户的目标描述
            session_id: 会话ID（可选）
            
        Returns:
            最终的系统状态
        """
        # 初始化会话
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.now()
        logger.info(f"{CTRL_LOG_PREFIX} session_start session={self.session_id}")
        # Wire session id to progress manager for consistent state snapshots
        self.progress_manager.set_session_id(self.session_id)

        # Step 1: Task Decomposition (初始任务分解)
        initial_tasks, first_task = await self._step1_task_decomposition(user_goal)

        if not initial_tasks:
            logger.error(f"{CTRL_LOG_PREFIX} decompose_empty")
            return self._create_empty_state("无法分解任务")

        # 初始化进度管理器的任务列表
        await self._initialize_progress_manager(initial_tasks)

        # 主执行循环 (Steps 2-6)
        iteration_count = 0
        max_iterations = 40  # 防止无限循环

        while iteration_count < max_iterations:
            iteration_count += 1
            logger.info(f"{CTRL_LOG_PREFIX} loop_iter iter={iteration_count} active={len(self.active_agents)}")

            # 获取当前系统状态
            current_state = self.progress_manager.get_current_state()

            # 检查是否所有任务都已完成
            if self._all_tasks_completed(current_state):
                logger.info(f"{CTRL_LOG_PREFIX} all_done iter={iteration_count}")
                break

            # 清理已完成的智能体
            await self._cleanup_completed_agents()

            # 若存在被阻塞任务并启用人机环，获取用户输入进行恢复/重规划
            if settings.human_in_loop_enabled:
                blocked = self._find_blocked_tasks(self.progress_manager.task_tree)
                if blocked:
                    # 仅处理第一个阻塞任务，避免一次性提示过多
                    target = blocked[0]
                    try:
                        resolution = await self._prompt_user_resolution(target)
                        if resolution:
                            await self.resolve_blocked(
                                task_id=target.id,
                                resolution_type=resolution.get("resolution_type", "replan"),
                                user_hint=resolution.get("user_hint", ""),
                                scope=resolution.get("scope", "subtree")
                            )
                            # 继续下一轮循环以使用最新任务树
                            await asyncio.sleep(0)
                            continue
                    except Exception as _e:
                        logger.error(f"{CTRL_LOG_PREFIX} resolution_input_error task={target.id} err={_e!s}")

            # Step 2: (Sub)Task Dispatch (任务派发)
            tasks_to_execute = await self._step2_task_dispatch(
                user_goal, current_state
            )

            if not tasks_to_execute and len(self.active_agents) == 0:
                # 如果仍有待办任务，触发一次重规划而非直接退出
                if current_state.pending_count > 0:
                    await self._trigger_replanning()
                    logger.info(f"{CTRL_LOG_PREFIX} idle_pending_replan")
                    # 等待任务列表更新事件或超时
                    await self._wait_for_task_list_update(timeout=3.0)
                    continue

                # 否则确认无待执行任务再退出
                logger.info(f"{CTRL_LOG_PREFIX} idle_no_tasks")
                break

            # Steps 3-4: Actor Instantiation & ReAct Execution
            await self._step3_4_actor_instantiation_and_execution(tasks_to_execute)

            # Step 5: Progress Update (通过 ProgressManager 自动处理)
            # 智能体会在执行过程中自动报告进度

            # Step 6: Evaluation and Iteration
            await self._step6_evaluation_and_iteration(current_state, iteration_count)

            # 短暂等待，让系统处理事件
            await asyncio.sleep(1)

        # 等待所有剩余的智能体完成
        await self._wait_for_all_agents()
        logger.info(f"{CTRL_LOG_PREFIX} session_end session={self.session_id}")

        # 返回最终状态
        return self.progress_manager.get_current_state()

    async def _step1_task_decomposition(
        self, user_goal: str
    ) -> tuple[list[Task], Task | None]:
        """
        Step 1: 任务分解。
        
        将用户目标分解为结构化的任务计划。
        """
        # 获取 ActorFactory 的工具反馈信息，为 Planner 提供工具使用指导
        tool_feedback = self.factory.get_tool_feedback_for_planner()

        # 构建增强的执行历史，包含工具反馈
        enhanced_history = []
        if tool_feedback.get("recommendations"):
            # 将工具反馈转换为虚拟的执行步骤，供 Planner 参考
            feedback_step = ExecutionStep(
                thought="工具使用反馈分析",
                action=f"tool_feedback: {json.dumps(tool_feedback, ensure_ascii=False)}",
                observation="ActorFactory 工具使用统计和建议"
            )
            enhanced_history.append(feedback_step)

        # 使用 DynamicPlanner 进行初始分解
        initial_tasks, first_task = await self.planner.plan_and_dispatch(
            goal=user_goal,
            current_tasks=[],
            execution_history=enhanced_history
        )
        logger.info(f"{CTRL_LOG_PREFIX} decompose_done tasks={len(initial_tasks)} first={(first_task.id if first_task else None)}")

        return initial_tasks, first_task

    async def _step2_task_dispatch(
        self, user_goal: str, current_state: SystemState
    ) -> list[Task]:
        """
        Step 2: 任务派发。
        
        基于当前状态选择下一批要执行的任务。
        """
        # 获取当前任务列表（从 ProgressManager）
        current_tasks = self._extract_tasks_from_state(current_state)

        # 获取工具反馈并增强执行历史
        tool_feedback = self.factory.get_tool_feedback_for_planner()
        enhanced_history = self.execution_history[-10:].copy()  # 最近10条历史

        if tool_feedback.get("recommendations"):
            # 添加工具反馈信息到执行历史
            feedback_step = ExecutionStep(
                thought="工具使用模式分析",
                action=f"tool_feedback: {json.dumps(tool_feedback, ensure_ascii=False)}",
                observation=f"工具多样性评分: {tool_feedback.get('tool_diversity_score', 0):.2f}, 建议: {'; '.join(tool_feedback.get('recommendations', []))}"
            )
            enhanced_history.append(feedback_step)

        # 使用批量规划获取可并行执行的任务
        updated_tasks, parallel_tasks = await self.planner.plan_and_dispatch_batch(
            goal=user_goal,
            current_tasks=current_tasks,
            execution_history=enhanced_history,
            max_parallel=self.config.max_parallel_agents - len(self.active_agents)
        )
        logger.info(f"{CTRL_LOG_PREFIX} dispatch tasks_to_execute={len(parallel_tasks)}")

        # 更新任务列表
        await self._update_task_list(updated_tasks)

        # 过滤出尚未执行的任务
        tasks_to_execute = [
            task for task in parallel_tasks
            if task.id not in self.active_agents
            and task.id not in self.completed_agents
            and task.status == TaskStatus.PENDING
        ]

        # 调试日志：分析为什么 parallel_tasks 被过滤掉
        if parallel_tasks and not tasks_to_execute:
            for task in parallel_tasks:
                logger.warning(
                    f"{CTRL_LOG_PREFIX} dispatch_filtered_out task={task.id} "
                    f"status={task.status.value} "
                    f"in_active={task.id in self.active_agents} "
                    f"in_completed={task.id in self.completed_agents}"
                )

        # 兜底：若规划未返回可派发任务，但仍有待办，则选择第一个 PENDING 任务
        # 注意：使用 updated_tasks 而非 current_tasks，以确保使用最新的任务树
        if not tasks_to_execute:
            for t in self._flatten_task_tree(updated_tasks):
                if (
                    t.status == TaskStatus.PENDING
                    and t.id not in self.active_agents
                    and t.id not in self.completed_agents
                ):
                    tasks_to_execute = [t]
                    logger.info(f"{CTRL_LOG_PREFIX} dispatch_fallback selected={t.id}")
                    break

        return tasks_to_execute

    async def _step3_4_actor_instantiation_and_execution(
        self, tasks: list[Task]
    ) -> None:
        """
        Steps 3-4: 智能体实例化和 ReAct 执行。
        
        为每个任务创建专门的智能体并启动异步执行。
        """
        for task in tasks:
            if len(self.active_agents) >= self.config.max_parallel_agents:
                break

            try:
                # Step 3: Actor Instantiation
                task_spec = self._task_to_specification(task)
                actor = await self.factory.create_agent(task_spec)

                # Step 4: ReAct Execution (异步启动)
                agent_task = asyncio.create_task(
                    self._execute_agent_with_timeout(actor, task)
                )

                async with self._agents_lock:
                    self.active_agents[task.id] = agent_task

                # 记录智能体启动
                await self.progress_manager.update_progress(
                    task_id=task.id,
                    agent_id=actor.actor_id,
                    status="in_progress",
                    message=f"智能体 {actor.actor_id} 开始执行任务"
                )
                logger.info(f"{CTRL_LOG_PREFIX} actor_started actor={actor.actor_id} task={task.id}")

            except Exception as e:
                # 智能体创建失败
                await self._handle_agent_creation_failure(task, e)
                logger.error(f"{CTRL_LOG_PREFIX} actor_create_fail task={task.id} error={e!s}")

    async def _execute_agent_with_timeout(
        self, actor: DynamicActor, task: Task
    ) -> dict[str, Any]:
        """
        使用超时机制执行智能体。
        """
        agent_task = None
        try:
            # 创建执行任务
            agent_task = asyncio.create_task(
                actor.execute(self.progress_manager)
            )

            # 设置超时
            result = await asyncio.wait_for(
                agent_task,
                timeout=self.config.agent_timeout
            )

            # 记录执行历史
            if actor.memory:
                async with self._history_lock:
                    self.execution_history.extend(actor.memory)
                    # 限制历史记录大小
                    if len(self.execution_history) > self.max_history_size:
                        self.execution_history = self.execution_history[-self.max_history_size:]

            # 标记任务完成
            task.status = TaskStatus.COMPLETED
            task.result = result.get("final_report", {})

            logger.info(f"{CTRL_LOG_PREFIX} actor_done actor={actor.actor_id} task={task.id}")
            return result

        except TimeoutError:
            # 尝试取消超时的任务
            if agent_task and not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass  # 正常取消

            # 超时处理
            task.status = TaskStatus.FAILED
            task.result = {"error": "Timeout", "message": f"任务执行超过 {self.config.agent_timeout} 秒"}

            # 更新进度
            await self.progress_manager.update_progress(
                task_id=task.id,
                agent_id=actor.actor_id if hasattr(actor, 'actor_id') else None,
                status="failed",
                message=f"任务超时（{self.config.agent_timeout}秒）"
            )
            logger.error(f"{CTRL_LOG_PREFIX} actor_timeout actor={actor.actor_id} task={task.id}")
            raise Exception(f"任务 {task.id} 执行超时")

        except Exception as e:
            # 其他错误
            task.status = TaskStatus.FAILED
            task.result = {"error": str(e)}
            logger.error(f"{CTRL_LOG_PREFIX} actor_error actor={actor.actor_id} task={task.id} error={e!s}")
            raise e

    async def _step6_evaluation_and_iteration(
        self, current_state: SystemState, iteration_index: int
    ) -> None:
        """
        Step 6: 评估与迭代。
        
        评估当前进度并决定是否需要调整策略。
        """
        # 检查失败的智能体
        if self.failed_agents and self.config.enable_auto_recovery:
            await self._handle_failed_agents()

        # 评估整体进度
        progress_ratio = self._calculate_progress_ratio(current_state)

        # 如果进度停滞，可能需要重新规划
        if progress_ratio < 0.1 and iteration_index > 10:
            await self._trigger_replanning()
            logger.info(f"{CTRL_LOG_PREFIX} replanning_triggered")

    def _task_to_specification(self, task: Task) -> TaskSpecification:
        """
        将 Task 对象转换为 TaskSpecification。
        """
        return TaskSpecification(
            task_id=task.id,
            description=task.description,
            context={
                "parent_task": task.parent_id if hasattr(task, 'parent_id') else None,
                "subtasks": [st.id for st in task.subtasks] if task.subtasks else [],
                "status": task.status.value,
                "session_id": self.session_id,
                "execution_history_length": len(self.execution_history)
            },
            constraints=[
                f"timeout: {self.config.agent_timeout}分钟",
                "max_iterations: 10"
            ],
            expected_output={
                "type": "structured_report",
                "format": "json"
            }
        )

    async def _cleanup_completed_agents(self) -> None:
        """
        清理已完成的智能体任务。
        """
        completed = []

        for task_id, agent_task in list(self.active_agents.items()):
            if agent_task.done():
                completed.append(task_id)

                try:
                    # 获取结果
                    result = await agent_task
                    self.completed_agents.add(task_id)
                    if task_id in self.task_retry_counts:
                        del self.task_retry_counts[task_id]

                    # 更新任务状态
                    await self.progress_manager.update_progress(
                        task_id=task_id,
                        status="completed",
                        message="任务成功完成"
                    )

                except Exception as e:
                    # 记录失败
                    self.failed_agents[task_id] = e

                    await self.progress_manager.update_progress(
                        task_id=task_id,
                        status="failed",
                        message=f"任务失败: {e!s}"
                    )

        # 从活跃列表中移除
        for task_id in completed:
            async with self._agents_lock:
                del self.active_agents[task_id]

    async def _wait_for_all_agents(self) -> None:
        """
        等待所有活跃的智能体完成。
        """
        if self.active_agents:
            # 等待所有任务完成，忽略异常
            await asyncio.gather(
                *self.active_agents.values(),
                return_exceptions=True
            )

            # 最后一次清理
            await self._cleanup_completed_agents()

    async def _wait_for_task_list_update(self, timeout: float = 3.0) -> None:
        """
        等待任务列表更新事件（system updated/replanned）或超时。
        """
        try:
            pm = self.progress_manager
            # 等待直到收到 system 更新事件
            while True:
                event = await asyncio.wait_for(pm.event_queue.get(), timeout=timeout)
                if event.get("type") in {"progress_update"}:
                    data = event.get("data", {})
                    if data.get("task_id") == "system" and data.get("status") in {"updated", "replanned"}:
                        break
                # 非关心事件，忽略继续等待，剩余时间由 wait_for 控制
        except asyncio.TimeoutError:
            # 超时视为无需阻塞，继续流程
            pass

    async def _initialize_progress_manager(self, tasks: list[Task]) -> None:
        """
        初始化进度管理器的任务列表。
        """
        # 设置任务树作为单一真相源
        self.progress_manager.set_task_tree(tasks)

        # 为每个任务发送初始化事件
        for task in self.progress_manager.get_all_tasks():
            await self.progress_manager.update_progress(
                task_id=task.id,
                status=task.status.value,
                message=f"任务已创建: {task.description}"
            )

    async def _update_task_list(self, tasks: list[Task]) -> None:
        """
        更新进度管理器中的任务列表，并同步已完成任务的状态。
        """
        # 同步 completed_agents 和 failed_agents 的状态到任务树
        # 防止已完成的任务被 Planner 重新标记为 pending 导致死循环
        for task in self._flatten_task_tree(tasks):
            if task.id in self.completed_agents:
                task.status = TaskStatus.COMPLETED
                logger.debug(f"{CTRL_LOG_PREFIX} sync_status task={task.id} status=completed")
            elif task.id in self.failed_agents:
                task.status = TaskStatus.FAILED
                logger.debug(f"{CTRL_LOG_PREFIX} sync_status task={task.id} status=failed")
        
        # 更新任务树
        self.progress_manager.set_task_tree(tasks)

        # 发送更新事件
        await self.progress_manager.update_progress(
            task_id="system",
            status="updated",
            message=f"任务列表已更新，共 {len(self.progress_manager.get_all_tasks())} 个任务"
        )

    async def _handle_agent_creation_failure(
        self, task: Task, error: Exception
    ) -> None:
        """
        处理智能体创建失败的情况。
        """
        task.status = TaskStatus.FAILED

        await self.progress_manager.update_progress(
            task_id=task.id,
            status="failed",
            message=f"智能体创建失败: {error!s}"
        )

        # 如果启用自动恢复，可以尝试重新创建
        if self.config.enable_auto_recovery:
            # TODO: 实现恢复逻辑
            pass

    async def _handle_failed_agents(self) -> None:
        """
        处理失败的智能体，尝试恢复或重新规划。
        """
        for task_id, error in list(self.failed_agents.items()):
            try:
                # 获取失败的任务
                task = self.progress_manager.get_task(task_id)
                if not task:
                    continue

                # 使用 LLM 分析失败原因并制定恢复策略
                recovery_plan = await self._analyze_failure_and_plan_recovery(
                    task, error
                )

                if recovery_plan.get("recoverable", False):
                    strategy = recovery_plan.get("strategy", "retry")

                    if strategy == "retry":
                        # 重试相同任务
                        await self._retry_failed_task(task)
                    elif strategy == "modify":
                        # 修改任务后重试
                        await self._modify_and_retry_task(task, recovery_plan)
                    elif strategy == "decompose":
                        # 重新分解任务
                        await self._decompose_failed_task(task, recovery_plan)
                    elif strategy == "skip":
                        # 跳过任务并继续
                        await self._skip_failed_task(task, recovery_plan)

                # 从失败列表中移除
                del self.failed_agents[task_id]

            except Exception as recovery_error:
                # 恢复策略本身失败，记录并继续
                await self.progress_manager.update_progress(
                    task_id=task_id,
                    status="failed",
                    message=f"恢复策略失败: {recovery_error!s}"
                )

    async def _analyze_failure_and_plan_recovery(
        self, task: Task, error: Exception
    ) -> dict[str, Any]:
        """
        使用 LLM 分析失败原因并制定恢复策略。
        """

        # 构建分析提示
        error_context = {
            "task_id": task.id,
            "task_description": task.description,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "task_status": task.status.value,
            "execution_history": len(self.execution_history)
        }

        analysis_prompt = f"""
        分析以下任务失败情况并制定恢复策略：
        
        任务信息：
        - ID: {task.id}
        - 描述: {task.description}
        - 当前状态: {task.status.value}
        
        错误信息：
        - 错误类型: {type(error).__name__}
        - 错误消息: {error!s}
        
        请分析：
        1. 这是什么类型的失败？(技术错误/逻辑错误/资源问题/超时等)
        2. 失败是否可以恢复？
        3. 最佳的恢复策略是什么？
        
        可选策略：
        - retry: 直接重试相同任务
        - modify: 修改任务描述或参数后重试
        - decompose: 将任务分解为更小的子任务
        - skip: 跳过这个任务，标记为可选
        
        返回JSON格式：
        {{
            "failure_type": "technical|logical|resource|timeout|network",
            "recoverable": true/false,
            "strategy": "retry|modify|decompose|skip",
            "reasoning": "详细的分析和建议",
            "modifications": {{
                "new_description": "修改后的任务描述",
                "parameters": {{"key": "value"}}
            }},
            "decomposition": [
                "子任务1描述",
                "子任务2描述"
            ]
        }}
        """

        try:
            # 使用规划器的 LLM 进行分析
            response = await self.planner.llm.complete(analysis_prompt)
            return json.loads(response)
        except Exception:
            # 降级策略：基于错误类型的简单判断
            return self._simple_recovery_analysis(error)

    def _simple_recovery_analysis(self, error: Exception) -> dict[str, Any]:
        """简单的错误恢复分析（降级策略）。"""

        # 即使是降级策略，也尝试使用结构化的方法而非关键词匹配
        # 基于错误类型本身而非字符串匹配

        if isinstance(error, asyncio.TimeoutError) or isinstance(error, TimeoutError):
            return {
                "failure_type": "timeout",
                "recoverable": True,
                "strategy": "retry",
                "reasoning": "任务执行超时，系统资源可能暂时繁忙"
            }
        elif isinstance(error, PermissionError):
            # PermissionError 需要在 OSError 之前检查，因为它是 OSError 的子类
            return {
                "failure_type": "resource",
                "recoverable": False,
                "strategy": "skip",
                "reasoning": "权限错误，无法自动恢复"
            }
        elif isinstance(error, ValueError):
            return {
                "failure_type": "resource",
                "recoverable": False,
                "strategy": "skip",
                "reasoning": "参数错误，无法自动恢复"
            }
        elif isinstance(error, (ConnectionError, OSError)):
            return {
                "failure_type": "network",
                "recoverable": True,
                "strategy": "retry",
                "reasoning": "网络或系统资源问题，可以尝试重试"
            }
        else:
            # 对于未知错误，返回最保守的策略
            return {
                "failure_type": "unknown",
                "recoverable": True,
                "strategy": "decompose",
                "reasoning": f"未识别的错误类型: {type(error).__name__}，尝试分解任务降低复杂度"
            }

    async def _retry_failed_task(self, task: Task) -> None:
        """重试失败的任务。"""
        retry_count = self.task_retry_counts[task.id]

        if retry_count >= self.config.max_task_retries:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now()
            await self.progress_manager.update_progress(
                task_id=task.id,
                status="failed",
                message="达到重试上限，任务已终止"
            )
            return

        next_retry = retry_count + 1
        self.task_retry_counts[task.id] = next_retry

        backoff_seconds = min(
            self.config.retry_backoff_max,
            self.config.retry_backoff_base ** retry_count,
        )

        await self.progress_manager.update_progress(
            task_id=task.id,
            status="pending",
            message=f"第 {next_retry} 次重试将在 {backoff_seconds} 秒后进行"
        )

        await asyncio.sleep(backoff_seconds)

        task.status = TaskStatus.PENDING
        task.updated_at = datetime.now()

    async def _modify_and_retry_task(self, task: Task, recovery_plan: dict) -> None:
        """修改任务后重试。"""

        modifications = recovery_plan.get("modifications", {})

        # 修改任务描述
        if "new_description" in modifications:
            original_description = task.description
            task.description = modifications["new_description"]

            await self.progress_manager.update_progress(
                task_id=task.id,
                status="pending",
                message=f"任务描述已修改：{original_description} -> {task.description}"
            )

        # 重置状态
        task.status = TaskStatus.PENDING
        task.updated_at = datetime.now()

    async def _decompose_failed_task(self, task: Task, recovery_plan: dict) -> None:
        """重新分解失败的任务。"""

        decomposition = recovery_plan.get("decomposition", [])

        if decomposition:
            # 创建子任务
            subtasks = []
            for i, subtask_desc in enumerate(decomposition):
                subtask = Task(
                    id=f"{task.id}_sub_{i+1}",
                    description=subtask_desc,
                    status=TaskStatus.PENDING,
                    subtasks=[],
                    result=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                subtasks.append(subtask)

            # 更新原任务
            task.subtasks = subtasks
            task.status = TaskStatus.PENDING
            task.updated_at = datetime.now()

            # 重建任务查找表
            self.progress_manager._rebuild_task_lookup()

            await self.progress_manager.update_progress(
                task_id=task.id,
                status="pending",
                message=f"任务已分解为 {len(subtasks)} 个子任务"
            )

    async def _skip_failed_task(self, task: Task, recovery_plan: dict) -> None:
        """跳过失败的任务。"""

        reasoning = recovery_plan.get("reasoning", "未知原因")

        # 标记为跳过
        task.status = TaskStatus.FAILED
        task.result = {
            "skipped": True,
            "reason": reasoning
        }
        task.updated_at = datetime.now()

        await self.progress_manager.update_progress(
            task_id=task.id,
            status="failed",
            message=f"任务已跳过：{reasoning}"
        )

    async def _trigger_replanning(self) -> None:
        """
        触发重新规划，当进度停滞时调用。
        """

        # 获取当前状态
        current_state = self.progress_manager.get_current_state()

        # 使用规划器重新分析和规划
        updated_tasks, next_task = await self.planner.plan_and_dispatch(
            goal=self.planner.goal or "继续执行剩余任务",
            current_tasks=self.progress_manager.task_tree,
            execution_history=self.execution_history[-20:],  # 最近20条历史
            user_feedback="系统检测到进度停滞，正在重新规划"
        )

        # 更新任务列表
        await self._update_task_list(updated_tasks)

        await self.progress_manager.update_progress(
            task_id="system",
            status="replanned",
            message="系统已重新规划任务"
        )

    def _all_tasks_completed(self, state: SystemState) -> bool:
        """
        检查是否所有任务都已完成。
        """
        return (
            state.pending_count == 0 and
            state.in_progress_count == 0 and
            len(self.active_agents) == 0
        )

    def _calculate_progress_ratio(self, state: SystemState) -> float:
        """
        计算进度比率。
        """
        total = state.task_count
        if total == 0:
            return 1.0

        completed = state.completed_count
        return completed / total

    def _extract_tasks_from_state(self, state: SystemState) -> list[Task]:
        """
        从系统状态中提取任务列表。
        
        现在直接从 ProgressManager 获取任务树。
        """
        return self.progress_manager.task_tree

    def _flatten_task_tree(self, tasks: list[Task]) -> list[Task]:
        """
        将任务树展平为列表。
        """
        result = []

        def traverse(task_list: list[Task]):
            for task in task_list:
                result.append(task)
                if task.subtasks:
                    traverse(task.subtasks)

        traverse(tasks)
        return result

    def _create_empty_state(self, message: str) -> SystemState:
        """
        创建空的系统状态。
        """
        return SystemState(
            timestamp=datetime.now(),
            session_id=self.session_id or "",
            task_count=0,
            completed_count=0,
            pending_count=0,
            in_progress_count=0,
            failed_count=0,
            active_agents=[],
            recent_events=[],
            overall_progress=0.0,
            estimated_completion=None,
            system_health="healthy"
        )

    async def shutdown(self) -> None:
        """
        优雅关闭系统。
        """
        # 取消所有活跃的智能体
        for agent_task in self.active_agents.values():
            agent_task.cancel()

        # 等待取消完成
        await self._wait_for_all_agents()

        # 清理资源
        self.active_agents.clear()
        self.completed_agents.clear()
        self.failed_agents.clear()
        self.execution_history.clear()

    def _find_blocked_tasks(self, tasks: list[Task]) -> list[Task]:
        """查找任务树中的 BLOCKED 任务。"""
        result: list[Task] = []

        def traverse(task_list: list[Task]):
            for t in task_list:
                if t.status == TaskStatus.BLOCKED:
                    result.append(t)
                if t.subtasks:
                    traverse(t.subtasks)

        traverse(tasks)
        return result

    async def _prompt_user_resolution(self, blocked_task: Task) -> dict[str, str] | None:
        """通过控制台异步读取用户分辨率与提示。"""
        prompt_text = (
            f"\n任务被阻塞: {blocked_task.description} ({blocked_task.id})\n"
            "选择处理方式: [resume/retry/replan] (默认 replan): "
        )
        # 非阻塞读取输入
        try:
            resolution_type = await asyncio.to_thread(input, prompt_text)
        except Exception:
            resolution_type = "replan"
        resolution_type = (resolution_type or "replan").strip().lower()

        scope = "subtree"
        if resolution_type == "replan":
            try:
                scope_in = await asyncio.to_thread(input, "重规划范围: [subtree/global] (默认 subtree): ")
                if scope_in and scope_in.strip().lower() in {"subtree", "global"}:
                    scope = scope_in.strip().lower()
            except Exception:
                scope = "subtree"

        try:
            user_hint = await asyncio.to_thread(input, "请输入用户提示(hint)，用于指导恢复/重规划: ")
        except Exception:
            user_hint = ""

        return {"resolution_type": resolution_type, "user_hint": user_hint or "", "scope": scope}

    async def resolve_blocked(self, task_id: str, resolution_type: str, user_hint: str, scope: str = "subtree") -> None:
        """处理被阻塞任务的恢复/重试/重规划。"""
        # 标记任务为 blocked 附带原因，生成简易 resume_token
        resume_token = uuid.uuid4().hex[:8]
        await self.progress_manager.update_progress(
            task_id=task_id,
            status="blocked",
            message=f"收到用户输入，准备执行 {resolution_type}",
            details={
                "blocked_reason": (self.progress_manager.get_task(task_id).blocked_reason if self.progress_manager.get_task(task_id) else None),
                "resume_token": resume_token,
            }
        )

        # 构造 user_feedback（用于 Planner 触发 replan 或其它恢复策略）
        user_feedback = None
        if resolution_type in {"replan", "retry", "resume"}:
            spec = {"resolution": resolution_type}
            if resolution_type == "replan":
                spec = {
                    "replan": {
                        "target_task_id": task_id,
                        "hint": user_hint,
                        "scope": scope,
                    }
                }
            user_feedback = json.dumps(spec, ensure_ascii=False)

        # 触发 Planner 更新（最小侵入：在现有任务树基础上传 user_feedback）
        updated_tasks, _ = await self.planner.plan_and_dispatch(
            goal=self.planner.goal or "继续执行剩余任务",
            current_tasks=self.progress_manager.task_tree,
            execution_history=self.execution_history[-20:],
            user_feedback=user_feedback,
        )

        # 更新任务树并广播
        await self._update_task_list(updated_tasks)
        await self.progress_manager.update_progress(
            task_id="system",
            status="replanned" if resolution_type == "replan" else "updated",
            message=f"用户 {resolution_type} 已应用，任务树已更新",
            details={"resume_token": resume_token}
        )
