"""基础工具抽象类和工具注册系统。"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# 统一的工具日志前缀
TOOL_LOG_PREFIX = "MiniAime|Tool|"

# 若未配置处理器，默认添加一个控制台处理器，确保日志可见
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class ToolError(Exception):
    """工具执行错误基类。"""
    pass


class BaseTool(ABC):
    """
    所有工具的抽象基类。
    
    提供统一的工具接口，支持同步和异步执行，
    包含权限检查、错误处理和日志记录功能。
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        required_permissions: Optional[List[str]] = None,
        **kwargs
    ):
        self.name = name
        self.description = description
        self.required_permissions = required_permissions or []
        self.metadata = kwargs
        
        # 统计信息
        self.call_count = 0
        self.error_count = 0
        
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        异步执行工具。
        
        Args:
            **kwargs: 工具执行参数
            
        Returns:
            工具执行结果
            
        Raises:
            ToolError: 工具执行失败
        """
        raise NotImplementedError
    
    def execute_sync(self, **kwargs) -> Any:
        """
        同步执行工具（默认实现抛出异常）。
        
        子类可以重写此方法提供同步版本。
        """
        raise NotImplementedError(f"Tool {self.name} does not support synchronous execution")
    
    async def validate_permissions(self, available_permissions: List[str]) -> bool:
        """
        验证工具所需权限。
        
        Args:
            available_permissions: 当前可用权限列表
            
        Returns:
            是否有足够权限执行工具
        """
        for permission in self.required_permissions:
            if permission not in available_permissions:
                logger.warning(f"Tool {self.name} requires permission {permission}")
                return False
        return True
    
    async def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """
        安全执行工具，包含错误处理和统计。
        
        Returns:
            包含执行结果和状态的字典
        """
        self.call_count += 1
        start_ts = time.perf_counter()
        arg_summary = str(kwargs)
        if len(arg_summary) > 256:
            arg_summary = arg_summary[:256] + "…"
        logger.info(f"{TOOL_LOG_PREFIX} start tool={self.name} args={arg_summary}")

        try:
            result = await self.execute(**kwargs)
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0

            # 结果预览
            preview: str
            if isinstance(result, (bytes, bytearray)):
                preview = f"<bytes:{len(result)}>"
            elif isinstance(result, str):
                preview = result[:256] + ("…" if len(result) > 256 else "")
            elif isinstance(result, dict):
                preview = f"dict(keys={list(result.keys())[:10]})"
            else:
                try:
                    _text = str(result)
                except Exception:
                    _text = "<unrepr>"
                preview = _text[:256] + ("…" if len(_text) > 256 else "")

            logger.info(
                f"{TOOL_LOG_PREFIX} success tool={self.name} cost_ms={elapsed_ms:.1f} result={preview}"
            )

            return {
                "success": True,
                "result": result,
                "tool_name": self.name,
                "error": None,
            }

        except Exception as e:
            self.error_count += 1
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.error(
                f"{TOOL_LOG_PREFIX} fail tool={self.name} cost_ms={elapsed_ms:.1f} error={str(e)}"
            )
            return {
                "success": False,
                "result": None,
                "tool_name": self.name,
                "error": str(e),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工具统计信息。"""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.call_count, 1),
            "required_permissions": self.required_permissions,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description}"


class ToolRegistry:
    """
    工具注册和管理系统。
    
    支持动态注册、查找和管理工具实例。
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        
    def register_tool(self, tool: BaseTool) -> None:
        """
        注册工具实例。
        
        Args:
            tool: 工具实例
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, got {type(tool)}")
            
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
            
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_tool_class(self, tool_class: Type[BaseTool], name: str) -> None:
        """
        注册工具类。
        
        Args:
            tool_class: 工具类
            name: 工具名称
        """
        if not issubclass(tool_class, BaseTool):
            raise TypeError(f"Expected BaseTool subclass, got {tool_class}")
            
        self._tool_classes[name] = tool_class
        logger.info(f"Registered tool class: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例。"""
        return self._tools.get(name)
    
    def create_tool(self, name: str, **kwargs) -> Optional[BaseTool]:
        """
        从注册的工具类创建工具实例。
        
        Args:
            name: 工具名称
            **kwargs: 工具初始化参数
            
        Returns:
            工具实例或None
        """
        tool_class = self._tool_classes.get(name)
        if not tool_class:
            return None
            
        try:
            return tool_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create tool {name}: {str(e)}")
            return None
    
    def list_tools(self) -> List[str]:
        """获取所有已注册工具名称。"""
        return list(self._tools.keys())
    
    def list_tool_classes(self) -> List[str]:
        """获取所有已注册工具类名称。"""
        return list(self._tool_classes.keys())
    
    def get_tools_by_permission(self, permissions: List[str]) -> List[BaseTool]:
        """
        根据权限筛选工具。
        
        Args:
            permissions: 可用权限列表
            
        Returns:
            有权限使用的工具列表
        """
        available_tools = []
        
        for tool in self._tools.values():
            # 检查是否有足够权限
            if all(perm in permissions for perm in tool.required_permissions):
                available_tools.append(tool)
                
        return available_tools
    
    def unregister_tool(self, name: str) -> bool:
        """
        注销工具。
        
        Args:
            name: 工具名称
            
        Returns:
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息。"""
        total_calls = sum(tool.call_count for tool in self._tools.values())
        total_errors = sum(tool.error_count for tool in self._tools.values())
        
        return {
            "total_tools": len(self._tools),
            "total_tool_classes": len(self._tool_classes),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "global_error_rate": total_errors / max(total_calls, 1),
            "tools": [tool.get_stats() for tool in self._tools.values()]
        }


# 全局工具注册表
default_registry = ToolRegistry()
