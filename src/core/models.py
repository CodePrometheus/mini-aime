"""mini-aime 系统的核心数据模型。"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from src.config.settings import settings


class TaskStatus(Enum):
    """任务执行状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"


@dataclass
class Task:
    """支持层级结构的任务数据结构。"""

    id: str
    description: str
    status: TaskStatus
    subtasks: list["Task"] | None = None
    result: str | None = None
    # 人机环/重规划相关元数据
    blocked_reason: str | None = None
    resume_token: str | None = None
    subtree_revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """当子任务列表为 None 时进行初始化。"""
        if self.subtasks is None:
            self.subtasks = []


@dataclass
class ExecutionStep:
    """ReAct 循环中的单步执行记录。"""

    thought: str
    action: str
    observation: str
    timestamp: datetime = field(default_factory=datetime.now)
    step_id: str | None = None


@dataclass
class AgentConfig:
    """智能体配置（工具与人格设定）。"""

    task: Task
    persona: str
    tools: list[str]
    knowledge: str = ""
    system_prompt: str = ""
    agent_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskRequest(BaseModel):
    """任务请求模型（用于 API）。"""

    goal: str = Field(..., description="User goal description")
    max_parallel_agents: int = Field(default=None, ge=1, le=10, description="Maximum parallel agents")
    timeout_seconds: int = Field(default=None, ge=30, le=3600, description="Task timeout in seconds")


class TaskResponse(BaseModel):
    """任务响应模型（用于 API）。"""

    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    result: dict[str, Any] | None = Field(None, description="Execution result")


class ProgressUpdate(BaseModel):
    """进度更新模型（用于实时状态跟踪）。"""

    task_id: str = Field(..., description="Task ID")
    agent_id: str | None = Field(None, description="Agent ID")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Progress message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    details: dict[str, Any] | None = Field(None, description="Additional details")


class SystemState(BaseModel):
    """系统状态模型（用于整体进度监控）。"""

    timestamp: datetime = Field(default_factory=datetime.now, description="Snapshot timestamp")
    session_id: str = Field(default="", description="Current session identifier")

    task_count: int = Field(default=0, description="Total number of tasks")
    active_agents: list[str] = Field(default_factory=list, description="Active agent IDs")
    completed_count: int = Field(default=0, description="Number of completed tasks")
    pending_count: int = Field(default=0, description="Number of pending tasks")
    in_progress_count: int = Field(default=0, description="Number of tasks in progress")
    failed_count: int = Field(default=0, description="Number of failed tasks")

    recent_events: list[str] = Field(default_factory=list, description="Recent progress events")
    overall_progress: float = Field(default=0.0, description="Overall task completion ratio [0,1]")
    estimated_completion: datetime | None = Field(
        default=None, description="Estimated completion time"
    )
    system_health: str = Field(default="healthy", description="System health status")

    total_agents_created: int = Field(default=0, description="Total agents created")
    system_uptime: float = Field(default=0.0, description="System uptime in seconds")


class AgentReport(BaseModel):
    """智能体最终执行报告。"""

    agent_id: str = Field(..., description="Agent ID")
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Final status")
    execution_steps: list[ExecutionStep] = Field(
        default_factory=list, description="Execution steps"
    )
    final_result: str = Field(..., description="Final execution result")
    error_message: str | None = Field(None, description="Error message if any")
    execution_time: float = Field(..., description="Execution time in seconds")
    tokens_used: int | None = Field(None, description="Number of tokens used")


class ToolBundle(BaseModel):
    """工具包配置模型。"""

    name: str = Field(..., description="Bundle name")
    description: str = Field(..., description="Bundle description")
    tools: list[str] = Field(..., description="List of included tools")
    required_permissions: list[str] = Field(
        default_factory=list, description="Required permissions"
    )


# Default tool bundle configurations
DEFAULT_TOOL_BUNDLES = {
    "web_research": ToolBundle(
        name="web_research",
        description="Web search, browse pages, extract information",
        tools=["search", "browse", "extract_content"],
        required_permissions=["internet_access"],
    ),
    "file_operations": ToolBundle(
        name="file_operations",
        description="Read/write files, directory operations",
        tools=["read_file", "write_file", "list_directory", "create_directory"],
        required_permissions=["file_system_access"],
    ),
    "code_analysis": ToolBundle(
        name="code_analysis",
        description="Code analysis, testing, debugging",
        tools=["analyze_code", "run_tests", "debug_code", "format_code"],
        required_permissions=["code_execution"],
    ),
    "data_processing": ToolBundle(
        name="data_processing",
        description="Data parsing, database queries, data transformation",
        tools=["parse_json", "query_database", "transform_data", "generate_report"],
        required_permissions=["data_access"],
    ),
    "research_integration": ToolBundle(
        name="research_integration",
        description="Integrate and analyze research files, generate comprehensive reports",
        tools=["integrate_research"],
        required_permissions=["file_system_access"],
    ),
}


class UserEventType(Enum):
    """用户可见的事件类型（基于 ReAct 框架）"""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    PLANNING = "planning"
    TASK_UPDATE = "task_update"
    MILESTONE = "milestone"
    ERROR = "error"
    COMPLETION = "completion"


class UserEvent(BaseModel):
    """面向用户的结构化事件"""

    event_type: UserEventType
    title: str
    content: str
    timestamp: datetime
    agent_id: str | None = None
    task_id: str | None = None
    level: str = "info"
    collapsible: bool = False
    icon: str | None = None
    details: dict[str, Any] | None = None

    def to_display_dict(self) -> dict:
        """转换为前端展示格式"""
        return {
            "type": self.event_type.value,
            "title": self.title,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "collapsible": self.collapsible,
            "icon": self.icon or self._default_icon(),
            "metadata": {"agent_id": self.agent_id, "task_id": self.task_id},
            "details": self.details or {},
        }

    def _default_icon(self) -> str:
        """默认图标映射"""
        icon_map = {
            UserEventType.THOUGHT: "💭",
            UserEventType.ACTION: "⚡",
            UserEventType.OBSERVATION: "👁️",
            UserEventType.PLANNING: "🎯",
            UserEventType.TASK_UPDATE: "📋",
            UserEventType.MILESTONE: "🎉",
            UserEventType.ERROR: "⚠️",
            UserEventType.COMPLETION: "✅",
        }
        return icon_map.get(self.event_type, "📌")
