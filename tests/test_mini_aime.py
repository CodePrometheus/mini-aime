"""
MiniAime 主控制器单元测试
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core import (
    MiniAime,
    MiniAimeConfig,
    PlannerConfig,
    Task,
    TaskStatus,
)


class SimpleMockLLMClient:
    """简单的模拟 LLM 客户端。"""

    def __init__(self):
        self.call_count = 0

    async def complete(self, prompt: str, **kwargs) -> str:
        """模拟简单的 LLM 调用。"""
        self.call_count += 1

        if "动态分析和规划" in prompt:
            return json.dumps(
                {
                    "updated_task_list": [
                        {
                            "id": "test_task_1",
                            "description": "测试任务1",
                            "status": "pending",
                            "subtasks": [],
                        }
                    ],
                    "next_action": {
                        "task_id": "test_task_1",
                        "action_type": "execute",
                        "reasoning": "执行测试任务",
                    },
                }
            )

        return "模拟响应"

    async def complete_with_context(self, messages: list[dict], **kwargs) -> str:
        """模拟带上下文的调用。"""
        return await self.complete("context", **kwargs)

    async def complete_with_functions(
        self, prompt: str, functions: list[dict], **kwargs
    ) -> dict[str, Any]:
        """模拟函数调用。"""
        return {"type": "text", "content": "模拟响应"}


@pytest.fixture
def simple_config():
    """简单的配置。"""
    return MiniAimeConfig(
        max_parallel_agents=1,
        agent_timeout=10,
        enable_auto_recovery=False,
        enable_persistence=False,
        planner_config=PlannerConfig(enable_user_interaction=False, max_parallel_tasks=1),
    )


@pytest.fixture
def mini_aime(simple_config):
    """创建 MiniAime 实例。"""
    llm_client = SimpleMockLLMClient()

    # 创建 MiniAime 实例，不初始化真实工具
    return MiniAime(llm_client, simple_config, initialize_tools=False)


class TestMiniAime:
    """MiniAime 主控制器测试。"""

    @pytest.mark.asyncio
    async def test_initialization(self, mini_aime):
        """测试初始化。"""
        assert mini_aime.planner is not None
        assert mini_aime.factory is not None
        assert mini_aime.progress_manager is not None
        assert mini_aime.config is not None

    @pytest.mark.asyncio
    async def test_task_to_specification_conversion(self, mini_aime):
        """测试任务到规格的转换。"""

        task = Task(
            id="test_task",
            description="测试任务描述",
            status=TaskStatus.PENDING,
            subtasks=[],
            result=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        spec = mini_aime._task_to_specification(task)

        assert spec.task_id == task.id
        assert spec.description == task.description
        assert spec.priority == "medium"  # 默认值

    @pytest.mark.asyncio
    async def test_task_tree_flattening(self, mini_aime):
        """测试任务树展平。"""

        subtask = Task(
            id="subtask_1",
            description="子任务1",
            status=TaskStatus.PENDING,
            subtasks=[],
            result=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        main_task = Task(
            id="main_task",
            description="主任务",
            status=TaskStatus.PENDING,
            subtasks=[subtask],
            result=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        flattened = mini_aime._flatten_task_tree([main_task])

        assert len(flattened) == 2
        assert any(t.id == "main_task" for t in flattened)
        assert any(t.id == "subtask_1" for t in flattened)

    @pytest.mark.asyncio
    async def test_progress_calculation(self, mini_aime):
        """测试进度计算。"""

        # 模拟系统状态
        from src.core.models import SystemState

        state = SystemState(
            timestamp=datetime.now(),
            session_id="test",
            task_count=10,
            completed_count=3,
            pending_count=5,
            in_progress_count=2,
            failed_count=0,
            active_agents=[],
            recent_events=[],
            overall_progress=0.3,
            estimated_completion=None,
            system_health="healthy",
        )

        progress = mini_aime._calculate_progress_ratio(state)
        assert progress == 0.3

    @pytest.mark.asyncio
    async def test_task_completion_check(self, mini_aime):
        """测试任务完成检查。"""

        from src.core.models import SystemState

        # 所有任务都完成
        completed_state = SystemState(
            timestamp=datetime.now(),
            session_id="test",
            task_count=5,
            completed_count=5,
            pending_count=0,
            in_progress_count=0,
            failed_count=0,
            active_agents=[],
            recent_events=[],
            overall_progress=1.0,
            estimated_completion=None,
            system_health="healthy",
        )

        assert mini_aime._all_tasks_completed(completed_state) is True

        # 还有未完成的任务
        incomplete_state = SystemState(
            timestamp=datetime.now(),
            session_id="test",
            task_count=5,
            completed_count=3,
            pending_count=2,
            in_progress_count=0,
            failed_count=0,
            active_agents=[],
            recent_events=[],
            overall_progress=0.6,
            estimated_completion=None,
            system_health="healthy",
        )

        assert mini_aime._all_tasks_completed(incomplete_state) is False

    @pytest.mark.asyncio
    async def test_simple_recovery_analysis(self, mini_aime):
        """测试简单的恢复分析。"""

        # 测试超时错误
        timeout_error = TimeoutError("Request timeout")
        analysis = mini_aime._simple_recovery_analysis(timeout_error)

        assert analysis["failure_type"] == "timeout"
        assert analysis["recoverable"] is True
        assert analysis["strategy"] == "retry"

        # 测试权限错误
        perm_error = PermissionError("Permission denied")
        analysis = mini_aime._simple_recovery_analysis(perm_error)

        # PermissionError 现在被归类为 resource 类型
        assert analysis["failure_type"] == "resource"
        assert analysis["recoverable"] is False
        assert analysis["strategy"] == "skip"

        # 测试 OSError（应该被归类为 network）
        os_error = OSError("Connection failed")
        analysis = mini_aime._simple_recovery_analysis(os_error)

        assert analysis["failure_type"] == "network"
        assert analysis["recoverable"] is True
        assert analysis["strategy"] == "retry"

    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(self, mini_aime):
        """测试智能体生命周期管理。"""

        # 初始状态
        assert len(mini_aime.active_agents) == 0
        assert len(mini_aime.completed_agents) == 0
        assert len(mini_aime.failed_agents) == 0

        # 模拟添加活跃智能体
        mini_aime.active_agents["agent_1"] = MagicMock()
        assert len(mini_aime.active_agents) == 1

        # 模拟智能体完成
        mini_aime.completed_agents.add("task_1")
        assert len(mini_aime.completed_agents) == 1

    @pytest.mark.asyncio
    async def test_execution_history_tracking(self, mini_aime):
        """测试执行历史跟踪。"""

        # 初始状态
        assert len(mini_aime.execution_history) == 0

        # 添加历史记录
        from src.core.models import ExecutionStep

        step = ExecutionStep(thought="测试思考", action="测试行动", observation="测试观察")

        mini_aime.execution_history.append(step)
        assert len(mini_aime.execution_history) == 1
        assert mini_aime.execution_history[0].thought == "测试思考"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
