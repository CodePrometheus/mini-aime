from __future__ import annotations

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, Body, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from src.config.logging import setup_logging
from src.config.settings import settings
from src.core.mini_aime import MiniAime, MiniAimeConfig
from src.llm.base import OpenAICompatibleClient


# 全局任务集合，用于存储后台任务引用
background_tasks_set: set[asyncio.Task] = set()


try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore


setup_logging()
logger = logging.getLogger("mini_aime.api")

# 存储活跃的任务
active_tasks = {}

# 任务状态持久化文件路径
ACTIVE_TASKS_FILE = "logs/active_tasks.json"


def load_active_tasks():
    """从文件加载活跃任务状态"""
    global active_tasks
    try:
        if os.path.exists(ACTIVE_TASKS_FILE):
            with open(ACTIVE_TASKS_FILE, encoding="utf-8") as f:
                loaded_tasks = json.load(f)
                active_tasks.update(loaded_tasks)  # 使用 update 而不是直接赋值
            logger.info(f"Loaded {len(loaded_tasks)} active tasks from file")
    except Exception as e:
        logger.warning(f"Failed to load active tasks: {e}")
        # 不要清空 active_tasks，保持现有状态


def save_active_tasks():
    """保存活跃任务状态到文件"""
    try:
        os.makedirs(os.path.dirname(ACTIVE_TASKS_FILE), exist_ok=True)

        # 创建可序列化的任务副本（排除不可序列化的对象）
        serializable_tasks = {}
        for task_id, task_info in active_tasks.items():
            serializable_task = {k: v for k, v in task_info.items() if k != "progress_manager"}
            serializable_tasks[task_id] = serializable_task

        with open(ACTIVE_TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_tasks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save active tasks: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器。"""
    # 启动时执行
    logger.info("API starting up", extra={"debug": settings.debug})
    load_active_tasks()  # 加载持久化的任务状态
    yield
    # 关闭时执行（如果需要的话）
    save_active_tasks()  # 保存任务状态
    logger.info("API shutting down")


app = FastAPI(title="Mini-Aime API", debug=settings.debug, lifespan=lifespan)


@app.get("/api/health")
async def health_check():
    """健康检查端点。"""
    return {"status": "healthy", "service": "mini-aime-api"}


@app.get("/api/tasks/active")
async def get_active_tasks():
    """获取当前活跃的任务列表。"""
    active_task_list = []
    for task_id, task_info in active_tasks.items():
        active_task_list.append(
            {
                "task_id": task_id,
                "status": task_info["status"],
                "goal": task_info.get("goal", ""),
                "created_at": task_info.get("created_at", ""),
            }
        )

    return {"active_tasks": active_task_list, "count": len(active_task_list)}


async def execute_task_background(task_id: str, goal: str, log_file: Path):
    """后台执行任务。"""
    llm_client = None
    file_handler = None
    
    try:
        # 配置日志输出到文件
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)

        # 添加文件handler到所有相关logger
        loggers_to_configure = [
            logging.getLogger("mini_aime"),
            logging.getLogger("src.core"),
            logging.getLogger("src.llm"),
        ]
        for log in loggers_to_configure:
            log.addHandler(file_handler)

        # 创建并执行任务
        llm_client = OpenAICompatibleClient()
        config = MiniAimeConfig()
        mini_aime = MiniAime(llm_client=llm_client, config=config)

        # 保存 progress_manager 引用供事件流使用
        active_tasks[task_id]["progress_manager"] = mini_aime.progress_manager
        active_tasks[task_id]["status"] = "running"
        save_active_tasks()  # 保存状态更新

        logger.info(f"Starting task {task_id}: {goal}")

        # 将 task_id 作为 session_id 传入，确保目录一致
        await mini_aime.execute_task(user_goal=goal, session_id=task_id)

        active_tasks[task_id]["status"] = "completed"
        save_active_tasks()  # 保存状态更新
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        save_active_tasks()  # 保存状态更新
    finally:
        # 关闭 LLM 客户端
        if llm_client:
            try:
                await llm_client.close()
            except Exception as e:
                logger.warning(f"Failed to close LLM client: {e}")
        
        # 移除文件handler
        if file_handler:
            for log in loggers_to_configure:
                log.removeHandler(file_handler)
            file_handler.close()


@app.get("/health")
async def health() -> dict:
    logger.debug("Health check called")
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():  # type: ignore[override]
    if not settings.enable_metrics:
        return {"detail": "metrics disabled"}
    if generate_latest is None:
        return {"detail": "prometheus-client not installed"}

    data = generate_latest()
    from fastapi import Response

    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/agent/resolution")
async def submit_resolution(
    task_id: str = Body(..., embed=True),
    resolution_type: str = Body(..., embed=True, description="resume|retry|replan"),
    user_hint: str | None = Body(None, embed=True),
) -> dict:
    """提交用户指引用于恢复/重试/重规划。"""
    if not settings.human_in_loop_enabled:
        return {"detail": "human-in-loop disabled"}

    return {
        "task_id": task_id,
        "resolution_type": resolution_type,
        "user_hint": (user_hint or "").strip(),
        "status": "accepted",
    }


@app.post("/api/tasks/submit")
async def submit_task(
    goal: str = Body(..., embed=True), background_tasks: BackgroundTasks = None
) -> dict:
    """提交新任务并返回任务ID。"""
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 创建日志文件
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{task_id}.log"

    active_tasks[task_id] = {
        "goal": goal,
        "status": "pending",
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(),
    }
    save_active_tasks()  # 保存新任务

    # 在后台执行任务
    if background_tasks:
        background_tasks.add_task(execute_task_background, task_id, goal, log_file)
    else:
        task = asyncio.create_task(execute_task_background(task_id, goal, log_file))
        background_tasks_set.add(task)
        task.add_done_callback(background_tasks_set.discard)

    logger.info(f"Task submitted: {task_id}, goal: {goal}")

    return {"task_id": task_id, "status": "pending", "message": "Task submitted successfully"}


@app.get("/api/tasks/{task_id}/logs")
async def stream_logs(task_id: str):
    """实时流式传输任务执行日志（SSE）。"""

    async def log_generator() -> AsyncGenerator[str, None]:
        if task_id not in active_tasks:
            yield "data: {'error': 'Task not found'}\n\n"
            return

        log_file = Path(active_tasks[task_id]["log_file"])

        # 等待日志文件创建
        retry_count = 0
        while not log_file.exists() and retry_count < 100:
            await asyncio.sleep(0.1)
            retry_count += 1

        if not log_file.exists():
            yield "data: {'message': 'Waiting for task to start...'}\n\n"

        # 读取并流式传输日志
        last_position = 0
        no_new_data_count = 0
        first_read = True

        while no_new_data_count < 60:  # 60秒没有新数据就断开
            try:
                if log_file.exists():
                    with open(log_file, encoding="utf-8") as f:
                        if first_read:
                            # 首次连接时，读取所有现有日志
                            all_lines = f.readlines()
                            for line in all_lines:
                                line_clean = line.rstrip("\n\r").replace('"', '\\"')
                                yield f'data: {{"log": "{line_clean}"}}\n\n'
                            last_position = f.tell()
                            first_read = False
                        else:
                            # 后续只读取新增的日志
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = f.tell()

                            if new_lines:
                                no_new_data_count = 0
                                for line in new_lines:
                                    # 移除换行符并转义引号
                                    line_clean = line.rstrip("\n\r").replace('"', '\\"')
                                    yield f'data: {{"log": "{line_clean}"}}\n\n'
                            else:
                                no_new_data_count += 1
                else:
                    no_new_data_count += 1

                # 检查任务是否完成
                if task_id in active_tasks and active_tasks[task_id]["status"] in (
                    "completed",
                    "failed",
                ):
                    # 再读取一次确保所有日志都发送了
                    await asyncio.sleep(1)
                    if log_file.exists():
                        with open(log_file, encoding="utf-8") as f:
                            f.seek(last_position)
                            final_lines = f.readlines()
                            for line in final_lines:
                                line_clean = line.rstrip("\n\r").replace('"', '\\"')
                                yield f'data: {{"log": "{line_clean}"}}\n\n'
                    break

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error reading log file: {e}")
                yield 'data: {"error": "Error reading logs"}\n\n'
                break

        yield 'data: {"message": "Stream ended"}\n\n'

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/tasks/{task_id}/events")
async def stream_user_events(task_id: str):
    """实时流式传输用户友好的事件（SSE）- 替代原始日志流。"""

    async def event_generator() -> AsyncGenerator[str, None]:
        if task_id not in active_tasks:
            yield 'data: {"error": "Task not found"}\n\n'
            return

        # 获取 progress_manager 引用
        progress_manager = active_tasks[task_id].get("progress_manager")

        if not progress_manager:
            # 如果 progress_manager 不存在（比如从文件恢复的任务），尝试重新创建
            logger.info(f"Progress manager not found for task {task_id}, attempting to recreate...")
            try:
                # 重新创建 MiniAime 实例来获取 progress_manager
                llm_client = OpenAICompatibleClient()
                config = MiniAimeConfig()
                mini_aime = MiniAime(llm_client=llm_client, config=config)

                # 设置相同的 session_id
                mini_aime.progress_manager.set_session_id(task_id)

                # 更新 active_tasks
                active_tasks[task_id]["progress_manager"] = mini_aime.progress_manager
                progress_manager = mini_aime.progress_manager

                logger.info(f"Successfully recreated progress manager for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to recreate progress manager for task {task_id}: {e}")
                yield 'data: {"error": "Failed to recreate progress manager"}\n\n'
                return

        # 发送连接确认
        yield f'data: {{"type": "connected", "task_id": "{task_id}"}}\n\n'

        # 首次连接：发送所有历史事件（支持页面刷新后恢复状态）
        history_events = progress_manager.get_user_event_history()
        if history_events:
            logger.info(
                f"Sending {len(history_events)} historical events to client for task {task_id}"
            )
            for event in history_events:
                event_json = json.dumps(event, ensure_ascii=False)
                yield f"data: {event_json}\n\n"

        # 持续从 user_event_queue 读取事件
        no_event_count = 0
        max_wait = 120  # 2分钟无事件则断开

        while no_event_count < max_wait:
            # 从 ProgressManager 获取用户事件
            user_event = await progress_manager.get_user_event(timeout=1.0)

            if user_event:
                no_event_count = 0
                # 发送事件到前端
                event_json = json.dumps(user_event, ensure_ascii=False)
                yield f"data: {event_json}\n\n"
            else:
                no_event_count += 1

            # 检查任务是否完成
            if task_id in active_tasks:
                status = active_tasks[task_id].get("status")
                if status in ("completed", "failed"):
                    # 再等待一会儿确保所有事件都发送了
                    await asyncio.sleep(2)

                    # 尝试读取剩余事件
                    for _ in range(5):
                        event = await progress_manager.get_user_event(timeout=0.5)
                        if event:
                            event_json = json.dumps(event, ensure_ascii=False)
                            yield f"data: {event_json}\n\n"
                        else:
                            break

                    # 发送完成信号
                    yield f'data: {{"type": "stream_ended", "task_status": "{status}"}}\n\n'
                    break

        # 超时
        if no_event_count >= max_wait:
            yield 'data: {"type": "timeout"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> dict:
    """获取任务的最终结果文档。

    根据论文，最终报告应该是：
    1. 各个子任务完成时生成子任务报告
    2. 所有任务完成后，汇总生成最终整合报告

    返回优先级：
    1. final_report.md (汇总报告) - 最终用户应看到的结果
    2. final_report_task_*.md (子任务报告) - 仅在汇总报告不存在时返回
    """

    # 检查任务是否存在
    if task_id not in active_tasks:
        return {"error": f"Task {task_id} not found"}

    task_info = active_tasks[task_id]
    task_status = task_info.get("status")

    # 如果任务还在运行，返回提示
    if task_status in ("pending", "running"):
        return {
            "task_id": task_id,
            "status": task_status,
            "message": "Task is still running, result not available yet",
        }

    # 任务完成，查找结果文件
    # 策略1: 优先查找任务专属目录中的汇总报告 docs/{task_id}/final_report.md
    task_docs_dir = Path("docs") / task_id

    if task_docs_dir.exists():
        # 优先查找精确的汇总报告文件名
        summary_report = task_docs_dir / "final_report.md"
        if summary_report.exists():
            try:
                content = summary_report.read_text(encoding="utf-8")
                return {
                    "task_id": task_id,
                    "filename": summary_report.name,
                    "content": content,
                    "generated_at": datetime.fromtimestamp(
                        summary_report.stat().st_mtime
                    ).isoformat(),
                    "type": "summary_report",  # 明确标识为汇总报告
                    "is_final": True,
                    "source": "task_directory",
                }
            except Exception as e:
                logger.error(f"Error reading summary report {summary_report}: {e}")

        # 查找其他可能的汇总报告文件名
        alternative_summary_names = ["final_learning_plan.md", "learning_plan.md", "综合报告.md"]
        for alt_name in alternative_summary_names:
            alt_report = task_docs_dir / alt_name
            if alt_report.exists():
                try:
                    content = alt_report.read_text(encoding="utf-8")
                    return {
                        "task_id": task_id,
                        "filename": alt_report.name,
                        "content": content,
                        "generated_at": datetime.fromtimestamp(
                            alt_report.stat().st_mtime
                        ).isoformat(),
                        "type": "summary_report",
                        "is_final": True,
                        "source": "task_directory",
                    }
                except Exception as e:
                    logger.error(f"Error reading alternative summary {alt_report}: {e}")

        # 查找任意以 final_report 开头的报告文件（不限文件名）
        # 但要排除明显的子任务报告（包含 _T[数字]_ 模式）
        all_reports = list(task_docs_dir.glob("final_report_*.md"))
        if all_reports:
            import re

            # 过滤掉子任务报告（文件名包含 _T数字_ 模式，如 final_report_T1_, final_report_T7_）
            summary_candidates = [f for f in all_reports if not re.search(r"_T\d+_", f.name)]

            if summary_candidates:
                latest_file = max(summary_candidates, key=lambda p: p.stat().st_mtime)
                try:
                    content = latest_file.read_text(encoding="utf-8")
                    return {
                        "task_id": task_id,
                        "filename": latest_file.name,
                        "content": content,
                        "generated_at": datetime.fromtimestamp(
                            latest_file.stat().st_mtime
                        ).isoformat(),
                        "type": "summary_report",
                        "is_final": True,
                        "source": "task_directory",
                    }
                except Exception as e:
                    logger.error(f"Error reading summary report {latest_file}: {e}")
            else:
                # 如果都是子任务报告，选择最新的（但标记为非最终）
                latest_file = max(all_reports, key=lambda p: p.stat().st_mtime)
                try:
                    content = latest_file.read_text(encoding="utf-8")
                    return {
                        "task_id": task_id,
                        "filename": latest_file.name,
                        "content": content,
                        "generated_at": datetime.fromtimestamp(
                            latest_file.stat().st_mtime
                        ).isoformat(),
                        "type": "subtask_report",
                        "is_final": False,
                        "warning": "显示的是最新的子任务报告，汇总报告可能尚未生成",
                        "source": "task_directory",
                    }
                except Exception as e:
                    logger.error(f"Error reading report {latest_file}: {e}")

    # 策略2: 兜底 - 在旧的 docs/ 目录中按时间过滤查找
    docs_dir = Path("docs")
    task_start_time_str = task_info.get("started_at")
    if task_start_time_str and docs_dir.exists():
        task_start_time = datetime.fromisoformat(task_start_time_str).timestamp()
        result_files = []
        for pattern in ["final_report_*.md", "docs/final_report_*.md"]:
            for file in docs_dir.glob(pattern):
                file_mtime = file.stat().st_mtime
                if task_start_time <= file_mtime:
                    result_files.append(file)

        if result_files:
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            try:
                content = latest_file.read_text(encoding="utf-8")
                return {
                    "task_id": task_id,
                    "filename": latest_file.name,
                    "content": content,
                    "generated_at": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
                    "type": "task_result",
                    "source": "legacy_time_filter",
                }
            except Exception as e:
                logger.error(f"Error reading result file {latest_file}: {e}")

    # 策略3: 最后兜底 - 查找任何最新的报告文件
    if docs_dir.exists():
        all_reports = list(docs_dir.glob("**/*.md"))
        if all_reports:
            latest_file = max(all_reports, key=lambda p: p.stat().st_mtime)
            try:
                content = latest_file.read_text(encoding="utf-8")
                return {
                    "task_id": task_id,
                    "filename": latest_file.name,
                    "content": content,
                    "generated_at": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
                    "type": "fallback",
                    "warning": "Using latest available report as fallback",
                }
            except Exception as e:
                logger.error(f"Error reading fallback file: {e}")

    return {"error": f"No result found for task {task_id}"}


@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> dict:
    """获取任务状态。"""
    if task_id not in active_tasks:
        return {"error": "Task not found"}

    return active_tasks[task_id]


# 挂载前端静态文件
frontend_dir = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(frontend_dir / "index.html")

    # 处理 favicon
    @app.get("/vite.svg")
    async def serve_favicon():
        favicon_path = frontend_dir / "vite.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return {"error": "Favicon not found"}
else:

    @app.get("/")
    async def serve_placeholder():
        return {"message": "Frontend not built. Run 'cd frontend && npm run build'"}


def main():
    """启动 Mini-Aime API 服务器的主函数"""
    import uvicorn
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
