"""实时进度管理器，提供系统级状态跟踪和事件通知。"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import ProgressUpdate, SystemState, Task, TaskStatus


class ProgressManager:
    """
    集中式进度管理器，负责跟踪系统状态和任务进展。
    
    作为论文中的"单一真相源"，直接管理 Task 对象，
    确保所有组件都能看到一致的任务状态。
    """
    
    def __init__(self):
        # 核心数据：任务树（单一真相源）
        self.task_tree: List[Task] = []
        self.task_lookup: Dict[str, Task] = {}  # 快速查找
        
        # 智能体状态
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        
        # 历史记录
        self.progress_history: List[ProgressUpdate] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # 系统状态
        self.system_start_time = datetime.now()
        self.subscribers: List[callable] = []
        
    def set_task_tree(self, tasks: List[Task]) -> None:
        """
        设置任务树（单一真相源）。
        
        Args:
            tasks: 完整的任务树
        """
        self.task_tree = tasks
        self._rebuild_task_lookup()
    
    def _rebuild_task_lookup(self) -> None:
        """重建任务查找表。"""
        self.task_lookup.clear()
        
        def add_tasks(task_list: List[Task]):
            for task in task_list:
                self.task_lookup[task.id] = task
                if task.subtasks:
                    add_tasks(task.subtasks)
        
        add_tasks(self.task_tree)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取指定ID的任务对象。"""
        return self.task_lookup.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus, result: Any = None) -> bool:
        """
        直接更新任务状态。
        
        Args:
            task_id: 任务ID
            status: 新状态
            result: 任务结果（可选）
            
        Returns:
            是否成功更新
        """
        task = self.get_task(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.now()
            if result is not None:
                task.result = result
            return True
        return False
    
    async def update_progress(
        self,
        task_id: str,
        agent_id: Optional[str] = None,
        status: str = "in_progress",
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """更新任务进度并发送事件通知。"""
        
        # 创建进度更新记录
        progress_update = ProgressUpdate(
            task_id=task_id,
            agent_id=agent_id,
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        # 更新 Task 对象状态
        if status in ["pending", "in_progress", "completed", "failed"]:
            task_status = TaskStatus(status)
            self.update_task_status(task_id, task_status)
        
        # 更新智能体状态
        if agent_id:
            if agent_id not in self.active_agents:
                self.active_agents[agent_id] = {
                    "task_id": task_id,
                    "created_at": datetime.now(),
                    "status": "active"
                }
            self.active_agents[agent_id]["last_activity"] = datetime.now()
            self.active_agents[agent_id]["current_status"] = status
        
        # 记录历史
        self.progress_history.append(progress_update)
        
        # 限制历史记录长度
        if len(self.progress_history) > 1000:
            self.progress_history = self.progress_history[-800:]  # 保留最近800条
        
        # 发送事件通知
        await self.event_queue.put({
            "type": "progress_update",
            "data": progress_update.model_dump(),
            "timestamp": datetime.now().isoformat()
        })
        
        # 通知订阅者
        for subscriber in self.subscribers:
            try:
                await subscriber(progress_update)
            except Exception:
                pass  # 忽略订阅者错误
        
        # 如果任务完成或失败，更新智能体状态
        if status in ["completed", "failed"]:
            if agent_id and agent_id in self.active_agents:
                self.active_agents[agent_id]["status"] = "completed"
    
    async def submit_final_report(
        self,
        task_id: str,
        agent_id: str,
        report: Dict[str, Any]
    ):
        """提交最终执行报告。"""
        
        # 更新任务结果
        task = self.get_task(task_id)
        if task:
            task.result = report
            task.status = TaskStatus.COMPLETED if report.get("status_update", {}).get("completed", False) else TaskStatus.FAILED
            task.updated_at = datetime.now()
        
        await self.update_progress(
            task_id=task_id,
            agent_id=agent_id,
            status="completed" if task and task.status == TaskStatus.COMPLETED else "failed",
            message=f"任务完成，结果: {report.get('conclusion_summary', {}).get('final_outcome', 'Unknown')[:100]}",
            details={
                "final_report": report,
                "completion_time": datetime.now().isoformat()
            }
        )
        
        # 发送完成事件
        await self.event_queue.put({
            "type": "task_completed",
            "data": {
                "task_id": task_id,
                "agent_id": agent_id,
                "report": report
            },
            "timestamp": datetime.now().isoformat()
        })
    
    def get_current_state(self) -> SystemState:
        """获取当前系统状态快照。"""
        
        # 基于任务树统计状态
        status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        
        def count_tasks(task_list: List[Task]):
            for task in task_list:
                status_key = task.status.value
                if status_key in status_counts:
                    status_counts[status_key] += 1
                
                if task.subtasks:
                    count_tasks(task.subtasks)
        
        count_tasks(self.task_tree)
        
        # 获取活跃智能体列表（状态不是completed的智能体）
        active_agent_ids = [
            agent_id for agent_id, agent_info in self.active_agents.items()
            if agent_info.get("status") != "completed"
        ]
        
        # 获取最近的事件
        recent_events = [
            f"[{update.timestamp.strftime('%H:%M:%S')}] {update.message}"
            for update in self.progress_history[-10:]  # 最近10条
        ]
        
        # 计算整体进度
        total_tasks = sum(status_counts.values())
        completed_tasks = status_counts["completed"]
        overall_progress = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return SystemState(
            timestamp=datetime.now(),
            session_id="current",  # TODO: 从外部传入
            task_count=total_tasks,
            completed_count=completed_tasks,
            pending_count=status_counts["pending"],
            in_progress_count=status_counts["in_progress"],
            failed_count=status_counts["failed"],
            active_agents=active_agent_ids,
            recent_events=recent_events,
            overall_progress=overall_progress,
            estimated_completion=None,  # TODO: 计算预估完成时间
            system_health="healthy"  # TODO: 基于错误率计算
        )
    
    async def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取特定任务的详细进度信息。"""
        
        task = self.get_task(task_id)
        if not task:
            return None
        
        # 获取该任务的进度历史
        task_history = [
            update for update in self.progress_history
            if update.task_id == task_id
        ]
        
        return {
            "task_id": task_id,
            "current_status": task.status.value,
            "description": task.description,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "last_update": task.updated_at.isoformat() if task.updated_at else None,
            "result": task.result,
            "subtasks": len(task.subtasks) if task.subtasks else 0,
            "progress_history": [update.model_dump() for update in task_history[-10:]]  # 最近10条
        }
    
    async def get_events(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取下一个事件（用于实时通知）。"""
        
        try:
            event = await asyncio.wait_for(self.event_queue.get(), timeout=timeout)
            return event
        except asyncio.TimeoutError:
            return None
    
    def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """清理已完成的旧任务记录。"""
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        # 清理已完成的旧任务
        tasks_to_remove = []
        for task in self.get_all_tasks():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and 
                task.updated_at and task.updated_at < cutoff_time):
                tasks_to_remove.append(task.id)
        
        # 清理智能体状态
        agents_to_remove = []
        for agent_id, agent_info in self.active_agents.items():
            if (agent_info["status"] == "completed" and
                agent_info.get("last_activity", datetime.now()).timestamp() < cutoff_time):
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            del self.active_agents[agent_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统运行统计信息。"""
        
        current_state = self.get_current_state()
        
        # 计算平均任务执行时间
        completed_tasks = self.find_tasks_by_status(TaskStatus.COMPLETED)
        
        avg_execution_time = 0.0
        if completed_tasks:
            valid_tasks = [
                task for task in completed_tasks 
                if task.created_at and task.updated_at
            ]
            if valid_tasks:
                total_time = sum([
                    (task.updated_at - task.created_at).total_seconds()
                    for task in valid_tasks
                ])
                avg_execution_time = total_time / len(valid_tasks)
        
        return {
            "total_tasks_processed": len(self.get_all_tasks()),
            "success_rate": (
                len(self.find_tasks_by_status(TaskStatus.COMPLETED)) / max(len(self.get_all_tasks()), 1) * 100
            ),
            "average_execution_time": avg_execution_time,
            "active_agents": len([a for a in self.active_agents.values() if a.get("status") != "completed"]),
            "total_progress_updates": len(self.progress_history),
            "current_queue_size": self.event_queue.qsize()
        }
    
    def subscribe(self, callback: callable) -> None:
        """订阅进度事件。"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: callable) -> None:
        """取消订阅进度事件。"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_all_tasks(self) -> List[Task]:
        """获取所有任务的扁平列表。"""
        all_tasks = []
        
        def collect_tasks(task_list: List[Task]):
            for task in task_list:
                all_tasks.append(task)
                if task.subtasks:
                    collect_tasks(task.subtasks)
        
        collect_tasks(self.task_tree)
        return all_tasks
    
    def find_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """查找指定状态的所有任务。"""
        return [task for task in self.get_all_tasks() if task.status == status]
