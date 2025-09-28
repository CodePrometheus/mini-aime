#!/usr/bin/env python3
"""测试 Brave Search 工具功能。"""

import asyncio
import os
import sys

import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.web_tools import BraveSearchTool


@pytest.mark.asyncio
async def test_brave_search_tool():
    """测试 Brave Search 工具功能。"""
    
    print("🔍 测试 Brave Search 工具功能")
    print("=" * 50)
    
    # 检查是否设置了 API Key
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        print("⚠️  未设置 BRAVE_SEARCH_API_KEY 环境变量")
        print("💡 请在 .env 文件中设置 BRAVE_SEARCH_API_KEY")
        return
    
    try:
        # 创建 Brave Search 工具实例
        brave_tool = BraveSearchTool(api_key=api_key, max_results=5)
        print("✅ Brave Search 工具初始化成功")
        
        # 测试搜索查询
        test_queries = [
            "Python programming best practices",
            "latest AI developments 2024",
            "machine learning tutorials"
        ]
        
        for query in test_queries:
            try:
                print(f"\n📍 搜索: {query}")
                
                # 执行搜索
                result = await brave_tool.execute(query)
                
                print(f"   结果类型: {type(result)}")
                print(f"   搜索结果键: {list(result.keys())}")
                
                # 显示基本信息
                print(f"   查询: {result.get('query', 'N/A')}")
                print(f"   总结果数: {result.get('total_results', 0):,}")
                print(f"   响应时间: {result.get('response_time', 0):.2f}秒")
                
                # 显示网页搜索结果
                web_results = result.get('web', {}).get('results', [])
                if web_results:
                    print(f"   网页结果数: {len(web_results)}")
                    
                    # 显示前3个结果的详细信息
                    for i, web_result in enumerate(web_results[:3], 1):
                        print(f"\n   📋 结果 {i}:")
                        print(f"      标题: {web_result.get('title', 'N/A')}")
                        print(f"      URL: {web_result.get('url', 'N/A')}")
                        
                        description = web_result.get('description', '')
                        if description:
                            # 截断长描述
                            if len(description) > 100:
                                description = description[:100] + "..."
                            print(f"      描述: {description}")
                        
                        # 显示其他可用字段
                        for key, value in web_result.items():
                            if key not in ['title', 'url', 'description'] and value:
                                if isinstance(value, str) and len(value) > 50:
                                    value = value[:50] + "..."
                                print(f"      {key}: {value}")
                
                # 显示信息框（如果有）
                infobox = result.get('infobox')
                if infobox:
                    print(f"\n   📋 信息框:")
                    print(f"      标题: {infobox.get('title', 'N/A')}")
                    description = infobox.get('description', '')
                    if description:
                        if len(description) > 150:
                            description = description[:150] + "..."
                        print(f"      描述: {description}")
                
                # 显示新闻结果（如果有）
                news_results = result.get('news', {}).get('results', [])
                if news_results:
                    print(f"\n   📰 新闻结果数: {len(news_results)}")
                    for i, news in enumerate(news_results[:2], 1):
                        print(f"      新闻 {i}: {news.get('title', 'N/A')}")
                        print(f"         URL: {news.get('url', 'N/A')}")
                        if news.get('age'):
                            print(f"         时间: {news['age']}")
                
                # 测试格式化结果
                print(f"\n   📄 格式化结果预览:")
                formatted = brave_tool.format_results(result, max_length=500)
                print(f"   {formatted[:200]}..." if len(formatted) > 200 else f"   {formatted}")
                
            except Exception as e:
                print(f"   ❌ 搜索失败: {str(e)}")
                print(f"   错误类型: {type(e).__name__}")
        
        # 测试同步版本
        print(f"\n🔄 测试同步搜索:")
        try:
            sync_result = brave_tool.execute_sync("test sync search")
            print(f"   同步搜索成功，结果类型: {type(sync_result)}")
            print(f"   同步搜索总结果数: {sync_result.get('total_results', 0)}")
        except Exception as e:
            print(f"   ❌ 同步搜索失败: {str(e)}")
        
        # 测试错误处理
        print(f"\n🚫 测试错误处理:")
        try:
            await brave_tool.execute("")  # 空查询
        except Exception as e:
            print(f"   ✅ 空查询错误处理正确: {str(e)}")
        
        try:
            await brave_tool.execute("   ")  # 空白查询
        except Exception as e:
            print(f"   ✅ 空白查询错误处理正确: {str(e)}")
    
    except Exception as e:
        print(f"❌ 初始化 Brave Search 工具时发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print("💡 请检查 API Key 是否正确设置")
    
    print("\n" + "=" * 50)
    print("测试完成！")


@pytest.mark.asyncio
async def test_brave_search_with_parameters():
    """测试 Brave Search 工具的参数配置。"""
    
    print("\n🔧 测试 Brave Search 参数配置")
    print("=" * 50)
    
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        print("⚠️  跳过参数测试：未设置 BRAVE_SEARCH_API_KEY")
        return
    
    try:
        # 测试不同的参数配置
        configs = [
            {
                "max_results": 3,
                "country": "CN",
                "search_lang": "zh",
                "ui_lang": "zh-CN",
                "safe_search": "strict"
            },
            {
                "max_results": 10,
                "country": "US",
                "search_lang": "en",
                "freshness": "pd",  # past day
                "text_decorations": False
            }
        ]
        
        for i, config in enumerate(configs, 1):
            print(f"\n📋 配置 {i}: {config}")
            
            try:
                brave_tool = BraveSearchTool(api_key=api_key, **config)
                result = await brave_tool.execute("artificial intelligence")
                
                print(f"   ✅ 配置 {i} 测试成功")
                print(f"   结果数: {len(result.get('web', {}).get('results', []))}")
                print(f"   响应时间: {result.get('response_time', 0):.2f}秒")
                
            except Exception as e:
                print(f"   ❌ 配置 {i} 测试失败: {str(e)}")
    
    except Exception as e:
        print(f"❌ 参数测试失败: {str(e)}")
    
    print("\n" + "=" * 50)
    print("参数测试完成！")


def test_brave_search_initialization():
    """测试 Brave Search 工具初始化。"""
    
    print("\n🏗️  测试 Brave Search 工具初始化")
    print("=" * 50)
    
    # 测试无 API Key 初始化
    try:
        # 临时清除环境变量
        original_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if "BRAVE_SEARCH_API_KEY" in os.environ:
            del os.environ["BRAVE_SEARCH_API_KEY"]
        
        BraveSearchTool()
        print("❌ 应该抛出 API Key 错误")
        
    except Exception as e:
        print(f"✅ 正确处理缺少 API Key: {str(e)}")
        
    finally:
        # 恢复环境变量
        if original_key:
            os.environ["BRAVE_SEARCH_API_KEY"] = original_key
    
    # 测试有效初始化
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if api_key:
        try:
            brave_tool = BraveSearchTool(api_key=api_key)
            print("✅ 有效 API Key 初始化成功")
            print(f"   工具名称: {brave_tool.name}")
            print(f"   工具描述: {brave_tool.description}")
            print(f"   最大结果数: {brave_tool.max_results}")
            print(f"   国家: {brave_tool.country}")
            print(f"   搜索语言: {brave_tool.search_lang}")
            
        except Exception as e:
            print(f"❌ 有效 API Key 初始化失败: {str(e)}")
    else:
        print("⚠️  跳过有效初始化测试：未设置 BRAVE_SEARCH_API_KEY")
    
    print("\n" + "=" * 50)
    print("初始化测试完成！")


if __name__ == "__main__":
    # 运行所有测试
    print("🚀 开始 Brave Search 工具测试")
    print("=" * 60)
    
    # 初始化测试
    test_brave_search_initialization()
    
    # 基本功能测试
    asyncio.run(test_brave_search_tool())
    
    # 参数配置测试
    asyncio.run(test_brave_search_with_parameters())
    
    print("\n🎉 所有测试完成！")
