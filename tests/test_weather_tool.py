#!/usr/bin/env python3
"""测试天气工具功能的简单脚本。"""

import asyncio
import os
import sys

import pytest


# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
async def test_weather_tool():
    """测试天气工具功能。"""

    print("🌤️  测试天气工具功能")
    print("=" * 50)

    # 创建工厂实例 (不初始化外部API工具，避免API Key问题)
    factory = ActorFactory(MockLLMClient(), initialize_tools=False)

    # 手动初始化天气工具进行测试
    def get_weather_func(city: str) -> str:
        """获取指定城市的当前天气信息。"""
        import os

        import requests

        # 使用 WeatherAPI 的 API 来获取天气信息
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            return "错误：未设置 WEATHER_API_KEY 环境变量，请在 .env 文件中配置"

        base_url = "http://api.weatherapi.com/v1/current.json"
        params = {
            "key": api_key,
            "q": city,
            "aqi": "no",  # 不需要空气质量数据
        }

        try:
            # 调用天气 API
            response = requests.get(base_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # 拿到天气和温度
                weather = data["current"]["condition"]["text"]
                temperature = data["current"]["temp_c"]
                humidity = data["current"]["humidity"]
                wind_kph = data["current"]["wind_kph"]
                return f"城市 {city} 当前天气：{weather}，温度 {temperature}°C，湿度 {humidity}%，风速 {wind_kph} km/h"
            else:
                return f"无法检索 {city} 的天气信息，API返回状态码: {response.status_code}"

        except requests.exceptions.RequestException as e:
            return f"获取 {city} 天气信息时网络请求失败: {e!s}"
        except Exception as e:
            return f"获取 {city} 天气信息时发生错误: {e!s}"

    from langchain_core.tools import Tool

    # 手动创建天气工具包进行测试
    weather_tool = Tool(
        name="get_weather",
        func=get_weather_func,
        description="获取指定城市的实时天气信息，包括温度、湿度、风速等",
    )

    factory.tool_bundles["weather_services"] = {
        "tools": [weather_tool],
        "description": "天气信息查询和分析",
        "use_cases": ["天气查询", "出行规划", "气候分析", "活动安排"],
    }

    # 检查天气工具包是否存在
    if "weather_services" in factory.tool_bundles:
        print("✅ 天气工具包已成功加载")

        weather_tools = factory.tool_bundles["weather_services"]["tools"]
        print(f"📦 天气工具包包含 {len(weather_tools)} 个工具:")

        for tool in weather_tools:
            print(f"   - {tool.name}: {tool.description}")

        # 测试天气查询功能
        print("\n🔍 测试天气查询功能:")

        # 获取天气工具
        get_weather_tool = weather_tools[0]  # 第一个工具应该是 get_weather

        # 测试几个城市
        test_cities = ["北京", "上海", "广州", "深圳"]

        for city in test_cities:
            try:
                print(f"\n📍 查询 {city} 的天气...")
                result = get_weather_tool.func(city)
                print(f"   结果: {result}")
            except Exception as e:
                print(f"   ❌ 查询失败: {e!s}")

    else:
        print("❌ 天气工具包未找到")

    print("\n" + "=" * 50)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(test_weather_tool())
