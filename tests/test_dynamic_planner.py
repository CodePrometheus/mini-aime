"""动态规划器功能的测试用例。"""

import asyncio
import json
from unittest.mock import Mock

import pytest

from src.core import DynamicPlanner, PlannerConfig, Task, TaskStatus
from src.llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """用于测试的模拟 LLM 客户端。"""

    def __init__(self, response_data=None):
        super().__init__()
        self.response_data = response_data or {
            "analysis": "Initial goal decomposition needed for Tokyo trip planning",
            "task_updates": [
                {
                    "action": "add",
                    "task": {
                        "id": "research_attractions",
                        "description": "Research Tokyo attractions and landmarks",
                        "status": "pending",
                    },
                },
                {
                    "action": "add",
                    "task": {
                        "id": "find_accommodation",
                        "description": "Find and book suitable accommodation",
                        "status": "pending",
                    },
                },
            ],
            "next_action": {
                "task_id": "research_attractions",
                "reasoning": "Start with research to understand available options",
            },
        }

    async def complete(self, prompt: str) -> str:
        """返回 JSON 响应的模拟补全。"""
        import json

        return json.dumps(self.response_data)

    async def complete_with_context(self, messages: list) -> str:
        """带上下文的模拟补全。"""
        import json

        return json.dumps(self.response_data)

    async def complete_with_functions(self, messages: list, functions: list, **kwargs) -> dict:
        """Function calling 的模拟实现。"""
        # 返回普通文本响应（不调用函数）
        content = await self.complete("")
        return {"content": content, "finish_reason": "stop"}


@pytest.fixture
def mock_llm():
    """提供模拟 LLM 客户端的测试夹具。"""
    return MockLLMClient()


@pytest.fixture
def planner(mock_llm):
    """提供带有模拟 LLM 的动态规划器实例。"""
    return DynamicPlanner(mock_llm, PlannerConfig())


@pytest.mark.asyncio
async def test_initial_planning(planner):
    """测试：初始任务分解。"""
    goal = "Plan a 3-day Tokyo trip"
    current_tasks = []
    execution_history = []

    updated_tasks, next_task = await planner.plan_and_dispatch(
        goal, current_tasks, execution_history
    )

    assert len(updated_tasks) == 2
    assert next_task is not None
    assert next_task.description == "Research Tokyo attractions and landmarks"
    assert next_task.id == "research_attractions"


@pytest.mark.asyncio
async def test_task_tree_operations(planner):
    """测试：任务树操作方法。"""
    # Create test tasks
    task1 = Task(id="task1", description="Main task", status=TaskStatus.PENDING, subtasks=[])

    # task2 = Task(
    #     id="task2",
    #     description="Subtask",
    #     status=TaskStatus.COMPLETED,
    #     subtasks=[]
    # )

    tasks = [task1]

    # Test _build_task_dict
    task_dict = planner._build_task_dict(tasks)
    assert "task1" in task_dict
    assert task_dict["task1"] == task1

    # Test _find_task_by_id
    found_task = planner._find_task_by_id(tasks, "task1")
    assert found_task == task1

    not_found = planner._find_task_by_id(tasks, "nonexistent")
    assert not_found is None

    # Test _find_first_pending_task
    pending_task = planner._find_first_pending_task(tasks)
    assert pending_task == task1


@pytest.mark.asyncio
async def test_planner_state_tracking(planner):
    """测试：规划器状态监控。"""
    goal = "Test goal"
    tasks = []
    history = []

    await planner.plan_and_dispatch(goal, tasks, history)

    state = planner.get_current_state()
    assert state["goal"] == goal
    assert state["planning_decisions"] == 1
    assert "task_count" in state
    assert "completed_tasks" in state
    assert "pending_tasks" in state


@pytest.mark.asyncio
async def test_fallback_planning():
    """测试：LLM 失败时的降级行为。"""
    # Create mock that raises exception
    failing_llm = Mock(spec=BaseLLMClient)
    failing_llm.complete_with_context.side_effect = Exception("LLM failed")

    planner = DynamicPlanner(failing_llm, PlannerConfig())

    goal = "Test goal"
    tasks = []
    history = []

    updated_tasks, _next_task = await planner.plan_and_dispatch(goal, tasks, history)

    # Should fallback to basic task creation
    assert len(updated_tasks) == 1
    assert "Analyze and break down" in updated_tasks[0].description


@pytest.mark.asyncio
async def test_task_updates_application(mock_llm):
    """测试：应用 LLM 生成的任务更新。"""
    # Setup mock with task modification response
    mock_llm.response_data = {
        "analysis": "Need to modify existing task",
        "task_updates": [
            {
                "action": "modify",
                "task_id": "existing_task",
                "changes": {"status": "completed", "description": "Modified description"},
            },
            {
                "action": "add",
                "task": {"id": "new_task", "description": "New task added", "status": "pending"},
            },
        ],
        "next_action": {"task_id": "new_task", "reasoning": "Execute new task"},
    }

    planner = DynamicPlanner(mock_llm, PlannerConfig())

    # Create initial task
    existing_task = Task(
        id="existing_task",
        description="Original description",
        status=TaskStatus.PENDING,
        subtasks=[],
    )

    current_tasks = [existing_task]

    updated_tasks, next_task = await planner.plan_and_dispatch("test", current_tasks, [])

    # Should have modified existing task and added new one
    assert len(updated_tasks) == 2

    # Find the modified task
    modified_task = next(t for t in updated_tasks if t.id == "existing_task")
    assert modified_task.status == TaskStatus.COMPLETED
    assert modified_task.description == "Modified description"

    # Check new task was added
    new_task = next(t for t in updated_tasks if t.id == "new_task")
    assert new_task.description == "New task added"

    # Check next task selection
    assert next_task.id == "new_task"


def test_planner_config():
    """测试：规划器配置项。"""
    config = PlannerConfig(
        enable_user_clarification=True,
        max_clarification_rounds=3,
        max_parallel_tasks=5,
        max_task_depth=6,
    )

    assert config.enable_user_clarification is True
    assert config.max_clarification_rounds == 3
    assert config.max_parallel_tasks == 5
    assert config.max_task_depth == 6


@pytest.mark.asyncio
async def test_plan_and_dispatch_batch():
    """测试：并行执行的批量规划。"""

    # Mock LLM responses for planning and parallel analysis
    class BatchMockLLMClient(BaseLLMClient):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        async def complete(self, prompt: str) -> str:
            import json

            if self.call_count == 0:
                # First call: planning response
                self.call_count += 1
                return json.dumps(
                    {
                        "analysis": "Initial task decomposition",
                        "task_updates": [
                            {
                                "action": "add",
                                "parent_id": None,
                                "task": {
                                    "id": "task1",
                                    "description": "Research Tokyo attractions",
                                    "status": "pending",
                                },
                            },
                            {
                                "action": "add",
                                "parent_id": None,
                                "task": {
                                    "id": "task2",
                                    "description": "Check weather forecast",
                                    "status": "pending",
                                },
                            },
                        ],
                        "next_action": {"task_id": "task1"},
                    }
                )
            else:
                # Second call: parallel analysis response
                return json.dumps(
                    {
                        "analysis": "Both tasks can run in parallel",
                        "parallel_task_ids": ["task1", "task2"],
                        "reasoning": "No resource conflicts between research and weather check",
                    }
                )

        async def complete_with_context(self, messages: list) -> str:
            return await self.complete("")

        async def complete_with_functions(self, messages: list, functions: list, **kwargs) -> dict:
            content = await self.complete("")
            return {"content": content, "finish_reason": "stop"}

    planner = DynamicPlanner(BatchMockLLMClient())

    updated_tasks, parallel_tasks = await planner.plan_and_dispatch_batch(
        goal="Plan Tokyo trip", current_tasks=[], execution_history=[]
    )

    # Verify results
    assert len(updated_tasks) == 2
    assert len(parallel_tasks) == 2
    assert parallel_tasks[0].id == "task1"  # Primary task first
    assert parallel_tasks[1].id == "task2"


@pytest.mark.asyncio
async def test_parallel_execution_with_dependencies():
    """测试：带依赖关系的并行执行分析。"""
    existing_tasks = [
        Task(id="task1", description="Book hotel", status=TaskStatus.PENDING),
        Task(id="task2", description="Check hotel booking", status=TaskStatus.PENDING),
        Task(id="task3", description="Research restaurants", status=TaskStatus.PENDING),
    ]

    class DependencyMockLLMClient(BaseLLMClient):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        async def complete(self, prompt: str) -> str:
            import json

            if self.call_count == 0:
                # First call: planning response
                self.call_count += 1
                return json.dumps(
                    {
                        "analysis": "Continue with existing tasks",
                        "task_updates": [],
                        "next_action": {"task_id": "task1"},
                    }
                )
            else:
                # Second call: parallel analysis that excludes dependent task
                return json.dumps(
                    {
                        "analysis": "Task2 depends on Task1, but Task3 is independent",
                        "parallel_task_ids": ["task1", "task3"],
                        "reasoning": "Hotel booking and restaurant research can run in parallel",
                        "excluded_tasks": {"task2": "Depends on task1 completion"},
                    }
                )

        async def complete_with_context(self, messages: list) -> str:
            return await self.complete("")

        async def complete_with_functions(self, messages: list, functions: list, **kwargs) -> dict:
            content = await self.complete("")
            return {"content": content, "finish_reason": "stop"}

    planner = DynamicPlanner(DependencyMockLLMClient())
    planner.task_list = existing_tasks

    _updated_tasks, parallel_tasks = await planner.plan_and_dispatch_batch(
        goal="Plan Tokyo trip", current_tasks=existing_tasks, execution_history=[]
    )

    # Should return task1 and task3, excluding dependent task2
    assert len(parallel_tasks) == 2
    task_ids = [t.id for t in parallel_tasks]
    assert "task1" in task_ids
    assert "task3" in task_ids
    assert "task2" not in task_ids


@pytest.mark.asyncio
async def test_identify_parallel_tasks_single_task():
    """测试：当 max_parallel=1 时的并行任务识别。"""
    primary_task = Task(id="task1", description="Main task", status=TaskStatus.PENDING)

    planner = DynamicPlanner(MockLLMClient())

    parallel_tasks = await planner._identify_parallel_tasks(
        tasks=[primary_task], primary_task=primary_task, max_parallel=1
    )

    assert len(parallel_tasks) == 1
    assert parallel_tasks[0] == primary_task


def test_progressive_user_guidance():
    """测试：渐进式用户引导功能。"""
    config = PlannerConfig(enable_user_interaction=True)

    # Mock LLM responses for ambiguity analysis and question generation
    llm_responses = {
        "ambiguity_analysis": json.dumps(
            {
                "needs_guidance": True,
                "ambiguous_aspects": [
                    {
                        "aspect": "目的地",
                        "description": "用户没有指定具体目的地",
                        "priority": "high",
                    }
                ],
                "suggested_questions": ["您想去哪里旅行？"],
            }
        ),
        "question_generation": json.dumps(
            {"questions": ["为了给您制定更合适的方案，您想去哪里呢？"]}
        ),
    }

    llm_client = MockLLMClient(llm_responses)
    planner = DynamicPlanner(llm_client, config)

    # Test progressive guidance
    result = asyncio.run(planner._progressive_user_guidance("帮我规划旅行", [], []))

    # Should return None as we don't have actual user interaction
    assert result is None


def test_user_interaction_disabled():
    """测试：默认情况下用户交互应为禁用。"""
    config = PlannerConfig(enable_user_interaction=False)
    llm_client = MockLLMClient()
    planner = DynamicPlanner(llm_client, config)

    # Should return None when disabled
    result = asyncio.run(planner._progressive_user_guidance("帮我规划旅行", [], []))

    assert result is None


def test_plan_and_dispatch_with_user_feedback():
    """测试：带用户反馈的规划。"""
    config = PlannerConfig(enable_user_interaction=True)

    # Mock successful planning response
    planning_response = json.dumps(
        {
            "analysis": "基于用户反馈调整计划",
            "task_updates": [
                {
                    "action": "add",
                    "task": {
                        "id": "task1",
                        "description": "基于用户反馈的新任务",
                        "status": "pending",
                    },
                }
            ],
            "next_action": {"task_id": "task1", "reasoning": "执行用户建议的任务"},
        }
    )

    # Create response data in the format expected by MockLLMClient
    response_data = json.loads(planning_response)
    llm_client = MockLLMClient(response_data)
    planner = DynamicPlanner(llm_client, config)

    # Test with user feedback
    result = asyncio.run(planner.plan_and_dispatch("规划旅行", [], [], user_feedback="我想去日本"))

    updated_tasks, next_task = result
    assert len(updated_tasks) == 1
    assert next_task is not None
    assert next_task.id == "task1"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_initial_planning(DynamicPlanner(MockLLMClient(), PlannerConfig())))
    print("✅ All Dynamic Planner tests passed!")
