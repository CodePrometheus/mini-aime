"""
Mini-Aime 系统使用示例

这个示例展示了如何使用 Mini-Aime 系统来执行一个简单的任务。
演示了论文中描述的动态多智能体系统的核心能力。
"""

import asyncio
import os
from datetime import datetime

from openai import AsyncOpenAI

from src.core import MiniAime, MiniAimeConfig, PlannerConfig
from src.llm import OpenAICompatibleClient


async def simple_task_example():
    """
    执行一个简单的任务示例：计划一个3天的东京旅行。
    
    这个例子展示了：
    1. 动态任务分解
    2. 智能体按需创建
    3. 并行执行
    4. 实时进度跟踪
    5. 结构化报告生成
    """
    
    # 1. 初始化 LLM 客户端（使用 DeepSeek）
    api_key = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
    
    llm_client = OpenAICompatibleClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat"
    )
    
    # 2. 配置系统
    config = MiniAimeConfig(
        max_parallel_agents=3,  # 最多3个智能体并行
        agent_timeout=60,  # 每个智能体60秒超时
        enable_auto_recovery=True,  # 启用自动错误恢复
        planner_config=PlannerConfig(
            enable_user_clarification=False,  # 使用论文原生模式
            max_parallel_tasks=3
        )
    )
    
    # 3. 创建 MiniAime 实例
    aime = MiniAime(llm_client, config)
    
    # 4. 定义用户目标
    user_goal = """
    帮我计划一个3天的东京旅行。
    我喜欢动漫文化和美食，希望体验地道的日本文化。
    预算大约每天1000元人民币。
    """
    
    print("=" * 60)
    print("Mini-Aime 系统启动")
    print("=" * 60)
    print(f"用户目标：{user_goal}")
    print("-" * 60)
    
    # 5. 执行任务
    start_time = datetime.now()
    
    try:
        # 执行主循环
        final_state = await aime.execute_task(user_goal)
        
        # 6. 显示执行结果
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("任务执行完成")
        print("=" * 60)
        
        # 显示系统状态
        print(f"\n📊 系统状态：")
        print(f"  - 总任务数：{final_state.task_count}")
        print(f"  - 已完成：{final_state.completed_count}")
        print(f"  - 失败：{final_state.failed_count}")
        print(f"  - 执行时间：{execution_time:.1f} 秒")
        print(f"  - 整体进度：{final_state.overall_progress * 100:.1f}%")
        
        # 显示活跃智能体
        if final_state.active_agents:
            print(f"\n👥 活跃智能体：")
            for agent in final_state.active_agents:
                print(f"  - {agent}")
        
        # 显示最近事件
        if final_state.recent_events:
            print(f"\n📝 最近事件：")
            for event in final_state.recent_events[-5:]:  # 显示最后5个事件
                print(f"  - {event}")
        
        print("\n" + "=" * 60)
        print("✅ 任务成功完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 任务执行失败：{str(e)}")
        
    finally:
        # 7. 清理资源
        await aime.shutdown()


async def advanced_example_with_monitoring():
    """
    高级示例：带实时监控的任务执行。
    
    展示如何：
    1. 实时监控任务进度
    2. 订阅事件通知
    3. 动态调整策略
    """
    
    # 初始化系统（同上）
    api_key = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
    llm_client = OpenAICompatibleClient(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # 启用用户交互的配置
    config = MiniAimeConfig(
        max_parallel_agents=5,
        planner_config=PlannerConfig(
            enable_user_interaction=True,  # 启用渐进式用户引导
            enable_user_clarification=True,  # 启用初始澄清
            max_clarification_rounds=2
        )
    )
    
    aime = MiniAime(llm_client, config)
    
    # 定义监控任务
    async def monitor_progress():
        """异步监控任务进度"""
        while True:
            await asyncio.sleep(5)  # 每5秒检查一次
            
            # 获取当前状态
            state = await aime.progress_manager.get_current_state()
            
            # 显示进度条
            progress = state.overall_progress * 100
            bar_length = 40
            filled = int(bar_length * state.overall_progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\r进度: [{bar}] {progress:.1f}% ", end="", flush=True)
            
            # 检查是否完成
            if state.pending_count == 0 and state.in_progress_count == 0:
                break
    
    # 执行复杂任务
    complex_goal = """
    创建一个完整的个人博客网站方案，包括：
    1. 技术栈选择和架构设计
    2. 功能规划和用户体验设计
    3. 部署方案和成本估算
    4. SEO优化策略
    """
    
    print("🚀 开始执行复杂任务...")
    print(f"目标：{complex_goal}\n")
    
    # 并行执行任务和监控
    task_future = asyncio.create_task(aime.execute_task(complex_goal))
    monitor_future = asyncio.create_task(monitor_progress())
    
    # 等待任务完成
    final_state = await task_future
    await monitor_future
    
    print("\n\n✅ 复杂任务执行完成！")
    
    # 显示详细报告
    print("\n📊 详细执行报告：")
    
    # 获取任务进度历史
    for task_id in aime.completed_agents:
        progress_history = await aime.progress_manager.get_task_progress(task_id)
        if progress_history:
            print(f"\n任务 {task_id}:")
            for update in progress_history[-3:]:  # 显示最后3条更新
                print(f"  - [{update.timestamp.strftime('%H:%M:%S')}] {update.message}")
    
    # 清理
    await aime.shutdown()


async def example_with_error_recovery():
    """
    错误恢复示例：展示系统的自适应能力。
    """
    
    # 配置启用错误恢复
    config = MiniAimeConfig(
        enable_auto_recovery=True,
        max_parallel_agents=2
    )
    
    # ... 初始化系统 ...
    
    # 执行可能失败的任务
    risky_goal = """
    从网络获取实时股票数据并生成投资报告。
    注意：这个任务可能因为网络问题或API限制而失败。
    """
    
    print("⚠️ 执行具有风险的任务...")
    
    try:
        final_state = await aime.execute_task(risky_goal)
        
        # 检查失败的任务
        if aime.failed_agents:
            print("\n❌ 部分任务失败：")
            for task_id, error in aime.failed_agents.items():
                print(f"  - 任务 {task_id}: {str(error)}")
            
            print("\n♻️ 系统自动恢复策略：")
            # 系统会自动尝试恢复或重新规划
        
    except Exception as e:
        print(f"系统级错误：{str(e)}")


def main():
    """主函数：运行所有示例"""
    
    print("🎯 Mini-Aime 系统示例")
    print("=" * 60)
    
    # 选择要运行的示例
    examples = {
        "1": ("简单任务示例", simple_task_example),
        "2": ("带监控的高级示例", advanced_example_with_monitoring),
        "3": ("错误恢复示例", example_with_error_recovery)
    }
    
    print("\n请选择要运行的示例：")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice in examples:
        name, example_func = examples[choice]
        print(f"\n运行：{name}")
        print("-" * 60)
        
        # 运行选中的示例
        asyncio.run(example_func())
    else:
        print("无效的选择")


if __name__ == "__main__":
    # 确保有 API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️ 警告：未设置 DEEPSEEK_API_KEY 环境变量")
        print("请设置：export DEEPSEEK_API_KEY='your-api-key'")
        print()
    
    main()
