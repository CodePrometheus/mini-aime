"""Core data models for the mini-aime system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task data structure with hierarchical support."""

    id: str
    description: str
    status: TaskStatus
    subtasks: list["Task"] | None = None
    result: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize subtasks list if None."""
        if self.subtasks is None:
            self.subtasks = []


@dataclass
class ExecutionStep:
    """Single execution step in ReAct cycle."""

    thought: str
    action: str
    observation: str
    timestamp: datetime = field(default_factory=datetime.now)
    step_id: str | None = None


@dataclass
class AgentConfig:
    """Agent configuration with tools and persona."""

    task: Task
    persona: str
    tools: list[str]
    knowledge: str = ""
    system_prompt: str = ""
    agent_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskRequest(BaseModel):
    """Task request model for API endpoints."""

    goal: str = Field(..., description="User goal description")
    max_parallel_agents: int = Field(default=3, ge=1, le=10, description="Maximum parallel agents")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Task timeout in seconds")


class TaskResponse(BaseModel):
    """Task response model for API endpoints."""

    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    result: dict[str, Any] | None = Field(None, description="Execution result")


class ProgressUpdate(BaseModel):
    """Progress update model for real-time status tracking."""

    task_id: str = Field(..., description="Task ID")
    agent_id: str | None = Field(None, description="Agent ID")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Progress message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    details: dict[str, Any] | None = Field(None, description="Additional details")


class SystemState(BaseModel):
    """System state model for monitoring overall progress."""

    task_count: int = Field(default=0, description="Total number of tasks")
    active_agents: list[str] = Field(default_factory=list, description="Active agent IDs")
    completed_count: int = Field(default=0, description="Number of completed tasks")
    pending_count: int = Field(default=0, description="Number of pending tasks")
    in_progress_count: int = Field(default=0, description="Number of tasks in progress")
    failed_count: int = Field(default=0, description="Number of failed tasks")
    total_agents_created: int = Field(default=0, description="Total agents created")
    system_uptime: float = Field(default=0.0, description="System uptime in seconds")


class AgentReport(BaseModel):
    """Agent final execution report."""

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
    """Tool bundle configuration model."""

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
}
