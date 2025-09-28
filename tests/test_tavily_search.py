#!/usr/bin/env python3
"""测试 Tavily 搜索工具功能。"""

import asyncio
import os
import sys

import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.actor_factory import ActorFactory
from src.llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """模拟 LLM 客户端用于测试。"""
    
    async def complete(self, prompt: str, **kwargs) -> str:
        return "{}"
    
    async def complete_with_context(self, messages: list, **kwargs) -> str:
        return "{}"
    
    async def complete_with_functions(self, messages: list, functions: list, **kwargs) -> dict:
        return {}


@pytest.mark.asyncio
async def test_tavily_search_tool():
    """测试 Tavily 搜索工具功能。"""
    
    print("🔍 测试 Tavily 搜索工具功能")
    print("=" * 50)
    
    try:
        # 创建工厂实例并初始化真实工具
        factory = ActorFactory(MockLLMClient(), initialize_tools=True)
        
        # 检查 web_research 工具包是否存在
        if "web_research" in factory.tool_bundles:
            print("✅ Web 研究工具包已成功加载")
            
            web_tools = factory.tool_bundles["web_research"]["tools"]
            print(f"📦 Web 研究工具包包含 {len(web_tools)} 个工具:")
            
            for tool in web_tools:
                print(f"   - {tool.name}: {tool.description}")
            
            # 测试 Tavily 搜索功能
            print("\n🔍 测试 Tavily 搜索功能:")
            
            # 获取 Tavily 搜索工具
            tavily_tool = web_tools[0]  # 第一个工具应该是 TavilySearchResults
            
            # 测试搜索查询
            test_queries = [
                "Python programming best practices",
                "latest AI developments 2024"
            ]
            
            for query in test_queries:
                try:
                    print(f"\n📍 搜索: {query}")
                    # 调用 Tavily 搜索
                    result = tavily_tool.run(query)
                    
                    print(f"   结果类型: {type(result)}")
                    
                    # 打印完整的搜索结果
                    if isinstance(result, dict):
                        print(f"   搜索结果字典键: {list(result.keys())}")
                        
                        # 打印每个键的内容
                        for key, value in result.items():
                            print(f"   📋 {key}:")
                            if isinstance(value, list):
                                print(f"      类型: list, 长度: {len(value)}")
                                if value:  # 如果列表不为空
                                    print(f"      第一项类型: {type(value[0])}")
                                    if isinstance(value[0], dict):
                                        print(f"      第一项键: {list(value[0].keys())}")
                                        # 打印第一个搜索结果的详细信息
                                        first_result = value[0]
                                        for k, v in first_result.items():
                                            if isinstance(v, str) and len(v) > 100:
                                                preview = v[:100] + "..."
                                            else:
                                                preview = v
                                            print(f"         {k}: {preview}")
                                    else:
                                        print(f"      第一项内容: {value[0]}")
                            elif isinstance(value, str):
                                preview = value[:200] + "..." if len(value) > 200 else value
                                print(f"      内容: {preview}")
                            else:
                                print(f"      值: {value}")
                    else:
                        # 如果不是字典，直接打印内容
                        if isinstance(result, str):
                            preview = result[:500] + "..." if len(result) > 500 else result
                            print(f"   完整结果: {preview}")
                        else:
                            print(f"   完整结果: {result}")
                        
                except Exception as e:
                    print(f"   ❌ 搜索失败: {str(e)}")
        
        else:
            print("❌ Web 研究工具包未找到")
    
    except Exception as e:
        print(f"❌ 初始化工具时发生错误: {str(e)}")
        # 这可能是由于缺少 API Key 或网络问题
        print("💡 提示: 请确保设置了 TAVILY_API_KEY 环境变量")
    
    print("\n" + "=" * 50)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(test_tavily_search_tool())
