"""测试工具系统。"""

import os
import tempfile

import pytest

from src.tools.base import BaseTool, ToolError, ToolRegistry
from src.tools.code_executor import CodeExecutorTool
from src.tools.file_tools import DirectoryListTool, FileReadTool, FileWriteTool


class MockTool(BaseTool):
    """模拟工具用于测试。"""

    def __init__(self):
        super().__init__(
            name="mock_tool",
            description="模拟工具用于测试",
            required_permissions=["test_permission"],
        )

    async def execute(self, **kwargs):
        return f"Mock result: {kwargs}"


class TestBaseTool:
    """BaseTool 基类测试。"""

    @pytest.mark.asyncio
    async def test_tool_creation(self):
        """测试：工具创建。"""
        tool = MockTool()

        assert tool.name == "mock_tool"
        assert tool.description == "模拟工具用于测试"
        assert tool.required_permissions == ["test_permission"]
        assert tool.call_count == 0
        assert tool.error_count == 0

    @pytest.mark.asyncio
    async def test_permission_validation(self):
        """测试：权限验证。"""
        tool = MockTool()

        # 有权限
        assert await tool.validate_permissions(["test_permission", "other_permission"])

        # 无权限
        assert not await tool.validate_permissions(["other_permission"])
        assert not await tool.validate_permissions([])

    @pytest.mark.asyncio
    async def test_safe_execute_success(self):
        """测试：安全执行成功。"""
        tool = MockTool()

        result = await tool.safe_execute(param1="value1", param2="value2")

        assert result["success"] is True
        assert result["tool_name"] == "mock_tool"
        assert result["error"] is None
        assert "Mock result" in str(result["result"])
        assert tool.call_count == 1
        assert tool.error_count == 0

    def test_get_stats(self):
        """测试：获取统计信息。"""
        tool = MockTool()
        tool.call_count = 5
        tool.error_count = 1

        stats = tool.get_stats()

        assert stats["name"] == "mock_tool"
        assert stats["call_count"] == 5
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.2


class TestToolRegistry:
    """工具注册表测试。"""

    def test_tool_registration(self):
        """测试：工具注册。"""
        registry = ToolRegistry()
        tool = MockTool()

        registry.register_tool(tool)

        assert "mock_tool" in registry.list_tools()
        assert registry.get_tool("mock_tool") is tool

    def test_tool_class_registration(self):
        """测试：工具类注册。"""
        registry = ToolRegistry()

        registry.register_tool_class(MockTool, "mock_tool_class")

        assert "mock_tool_class" in registry.list_tool_classes()

        # 创建工具实例
        tool = registry.create_tool("mock_tool_class")
        assert tool is not None
        assert tool.name == "mock_tool"

    def test_permission_filtering(self):
        """测试：权限筛选。"""
        registry = ToolRegistry()

        tool1 = MockTool()
        tool2 = MockTool()
        tool2.name = "mock_tool_2"
        tool2.required_permissions = ["other_permission"]

        registry.register_tool(tool1)
        registry.register_tool(tool2)

        # 筛选有权限的工具
        available_tools = registry.get_tools_by_permission(["test_permission"])
        assert len(available_tools) == 1
        assert available_tools[0].name == "mock_tool"

    def test_registry_stats(self):
        """测试：注册表统计。"""
        registry = ToolRegistry()
        tool = MockTool()
        tool.call_count = 3
        tool.error_count = 1

        registry.register_tool(tool)

        stats = registry.get_registry_stats()

        assert stats["total_tools"] == 1
        assert stats["total_calls"] == 3
        assert stats["total_errors"] == 1


class TestFileTools:
    """文件工具测试。"""

    @pytest.mark.asyncio
    async def test_file_read_write(self):
        """测试：文件读写。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建工具
            write_tool = FileWriteTool(allowed_paths=[temp_dir])
            read_tool = FileReadTool(allowed_paths=[temp_dir])

            # 测试文件路径
            test_file = os.path.join(temp_dir, "test.txt")
            test_content = "Hello, World!\n这是测试内容。"

            # 写入文件
            write_result = await write_tool.execute(test_file, test_content)
            assert "成功写入文件" in write_result

            # 读取文件
            read_result = await read_tool.execute(test_file)
            assert read_result == test_content

    @pytest.mark.asyncio
    async def test_default_write_to_docs_directory(self):
        """当未提供 allowed_paths 时，写入应默认落到项目根 docs 目录。"""
        # 不传 allowed_paths，启用默认 docs 策略
        write_tool = FileWriteTool()

        # 相对路径应解析到项目根 docs 目录
        relative_name = "test_docs_write.txt"
        content = "doc content"

        await write_tool.execute(relative_name, content)

        # 解析出实际路径并断言
        # 从返回消息中提取文件路径或重建预期路径
        # 我们直接重建：项目根 -> docs -> relative_name
        # 与工具内部保持一致（通过 pyproject/git/README 探测）
        from src.tools.file_tools import _find_project_root  # type: ignore

        project_root = _find_project_root()
        docs_file = os.path.join(project_root, "docs", relative_name)

        assert os.path.exists(docs_file)
        with open(docs_file, encoding="utf-8") as f:
            assert f.read() == content

    @pytest.mark.asyncio
    async def test_directory_list(self):
        """测试：目录列表。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # 创建工具
            list_tool = DirectoryListTool(allowed_paths=[temp_dir])

            # 列出目录
            result = await list_tool.execute(temp_dir)

            assert "test.txt" in result
            assert "目录" in result

    @pytest.mark.asyncio
    async def test_path_restriction(self):
        """测试：路径限制。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            read_tool = FileReadTool(allowed_paths=[temp_dir])

            # 尝试访问受限路径
            with pytest.raises(ToolError) as exc_info:
                await read_tool.execute("/etc/passwd")

            assert "Access denied" in str(exc_info.value)


class TestCodeExecutor:
    """代码执行工具测试。"""

    @pytest.mark.asyncio
    async def test_simple_python_execution(self):
        """测试：简单 Python 代码执行。"""
        executor = CodeExecutorTool(timeout=5)

        code = """
print("Hello, World!")
result = 2 + 3
print(f"2 + 3 = {result}")
"""

        result = await executor.execute(code)

        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert "2 + 3 = 5" in result["output"]
        assert result["error"] is None or result["error"] == ""

    @pytest.mark.asyncio
    async def test_math_calculation(self):
        """测试：数学计算。"""
        executor = CodeExecutorTool()

        code = """
import math
result = math.sqrt(16) + math.pi
print(f"计算结果: {result:.2f}")
"""

        result = await executor.execute(code)

        assert result["success"] is True
        assert "计算结果" in result["output"]

    @pytest.mark.asyncio
    async def test_restricted_function(self):
        """测试：受限函数检查。"""
        executor = CodeExecutorTool()

        # 尝试使用受限函数
        dangerous_code = """
import os
os.system("ls")
"""

        with pytest.raises(ToolError) as exc_info:
            await executor.execute(dangerous_code)

        assert "not allowed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        """测试：语法错误处理。"""
        executor = CodeExecutorTool()

        invalid_code = """
print("Hello"
# 缺少右括号
"""

        with pytest.raises(ToolError) as exc_info:
            await executor.execute(invalid_code)

        assert "Syntax error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_runtime_error(self):
        """测试：运行时错误处理。"""
        executor = CodeExecutorTool()

        error_code = """
result = 1 / 0  # 除零错误
"""

        result = await executor.execute(error_code)

        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]
