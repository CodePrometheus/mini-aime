"""代码执行工具，提供安全的代码执行沙箱环境。"""

import ast
import io
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolError


class CodeExecutorTool(BaseTool):
    """
    安全的代码执行工具。
    
    支持 Python 代码执行，提供沙箱环境和安全限制。
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_length: int = 10000,
        allowed_modules: Optional[List[str]] = None,
        restricted_functions: Optional[List[str]] = None
    ):
        super().__init__(
            name="execute_code",
            description="在安全沙箱环境中执行 Python 代码",
            required_permissions=["code_execution"],
            timeout=timeout,
            max_output_length=max_output_length,
            allowed_modules=allowed_modules or [],
            restricted_functions=restricted_functions or []
        )
        
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.allowed_modules = set(allowed_modules or self._get_default_allowed_modules())
        self.restricted_functions = set(restricted_functions or self._get_default_restricted_functions())
    
    def _get_default_allowed_modules(self) -> List[str]:
        """获取默认允许的模块列表。"""
        return [
            'math', 'statistics', 'random', 'datetime', 'time',
            'json', 'csv', 'base64', 'hashlib', 'uuid',
            'collections', 'itertools', 'functools', 'operator',
            're', 'string', 'textwrap',
            'numpy', 'pandas', 'matplotlib', 'seaborn',
            'requests'  # 网络请求（受限）
        ]
    
    def _get_default_restricted_functions(self) -> List[str]:
        """获取默认限制的函数列表。"""
        return [
            'open', 'file', 'input', 'raw_input',
            'exec', 'eval', 'compile', '__import__',
            'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'exit', 'quit', 'help'
        ]
    
    async def execute(
        self, 
        code: str, 
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行代码。
        
        Args:
            code: 要执行的代码
            language: 编程语言，目前支持 'python'
            timeout: 执行超时时间（秒）
            
        Returns:
            包含执行结果的字典：
            - success: 是否成功执行
            - output: 标准输出
            - error: 错误信息
            - execution_time: 执行时间
            - language: 编程语言
            
        Raises:
            ToolError: 代码执行失败
        """
        if not code or not code.strip():
            raise ToolError("Code cannot be empty")
        
        if language.lower() != "python":
            raise ToolError(f"Unsupported language: {language}")
        
        execution_timeout = timeout or self.timeout
        
        try:
            # 安全检查
            self._validate_code_safety(code)
            
            # 执行代码
            start_time = time.time()
            result = await self._execute_python_code(code, execution_timeout)
            execution_time = time.time() - start_time
            
            # 限制输出长度
            if result['output'] and len(result['output']) > self.max_output_length:
                result['output'] = result['output'][:self.max_output_length] + "\n[输出已截断]"
            
            if result['error'] and len(result['error']) > self.max_output_length:
                result['error'] = result['error'][:self.max_output_length] + "\n[错误信息已截断]"
            
            result['execution_time'] = execution_time
            result['language'] = language
            
            return result
            
        except Exception as e:
            raise ToolError(f"Code execution failed: {str(e)}")
    
    def _validate_code_safety(self, code: str) -> None:
        """
        验证代码安全性。
        
        Args:
            code: 要验证的代码
            
        Raises:
            ToolError: 代码包含不安全内容
        """
        try:
            # 解析 AST
            tree = ast.parse(code)
            
            # 检查危险操作
            for node in ast.walk(tree):
                # 检查导入语句
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_modules:
                            raise ToolError(f"Import of module '{alias.name}' is not allowed")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.allowed_modules:
                        raise ToolError(f"Import from module '{node.module}' is not allowed")
                
                # 检查函数调用
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.restricted_functions:
                            raise ToolError(f"Function '{node.func.id}' is not allowed")
                
                # 检查属性访问
                elif isinstance(node, ast.Attribute):
                    # 限制访问某些危险属性
                    dangerous_attrs = ['__import__', '__builtins__', '__globals__']
                    if node.attr in dangerous_attrs:
                        raise ToolError(f"Access to attribute '{node.attr}' is not allowed")
                
        except SyntaxError as e:
            raise ToolError(f"Syntax error in code: {str(e)}")
    
    async def _execute_python_code(self, code: str, timeout: int) -> Dict[str, Any]:
        """
        执行 Python 代码。
        
        Args:
            code: Python 代码
            timeout: 超时时间
            
        Returns:
            执行结果字典
        """
        # 创建受限的全局环境
        restricted_globals = self._create_restricted_globals()
        
        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                with self._timeout_context(timeout):
                    # 编译并执行代码
                    compiled_code = compile(code, '<string>', 'exec')
                    exec(compiled_code, restricted_globals)
            
            return {
                'success': True,
                'output': stdout_capture.getvalue(),
                'error': stderr_capture.getvalue() if stderr_capture.getvalue() else None
            }
            
        except TimeoutError:
            return {
                'success': False,
                'output': stdout_capture.getvalue(),
                'error': f"Code execution timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                'success': False,
                'output': stdout_capture.getvalue(),
                'error': f"{type(e).__name__}: {str(e)}"
            }
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """创建受限的全局变量环境。"""
        # 基础内置函数
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'format', 'frozenset', 'hex', 'int', 'len',
            'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'print',
            'range', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'zip'
        }
        
        # 创建受限的 builtins
        restricted_builtins = {}
        for name in safe_builtins:
            if hasattr(__builtins__, name):
                restricted_builtins[name] = getattr(__builtins__, name)
        
        # 添加安全的模块
        safe_modules = {}
        for module_name in self.allowed_modules:
            try:
                safe_modules[module_name] = __import__(module_name)
            except ImportError:
                pass  # 模块不存在，跳过
        
        return {
            '__builtins__': restricted_builtins,
            **safe_modules
        }
    
    @contextmanager
    def _timeout_context(self, timeout: int):
        """创建超时上下文管理器。"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError()
        
        # 设置信号处理器
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            yield
        finally:
            # 恢复原来的信号处理器
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def execute_sync(
        self, 
        code: str, 
        language: str = "python",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """同步版本的代码执行。"""
        import asyncio
        return asyncio.run(self.execute(code, language, timeout))


class ShellCommandTool(BaseTool):
    """
    Shell 命令执行工具。
    
    提供受限的系统命令执行能力。
    """
    
    def __init__(
        self,
        timeout: int = 30,
        allowed_commands: Optional[List[str]] = None,
        working_directory: Optional[str] = None
    ):
        super().__init__(
            name="execute_shell",
            description="执行系统 Shell 命令",
            required_permissions=["system_access"],
            timeout=timeout,
            allowed_commands=allowed_commands or [],
            working_directory=working_directory
        )
        
        self.timeout = timeout
        self.allowed_commands = set(allowed_commands or self._get_default_allowed_commands())
        self.working_directory = working_directory or tempfile.gettempdir()
    
    def _get_default_allowed_commands(self) -> List[str]:
        """获取默认允许的命令列表。"""
        return [
            'ls', 'dir', 'pwd', 'echo', 'cat', 'head', 'tail',
            'grep', 'find', 'wc', 'sort', 'uniq',
            'python', 'python3', 'pip', 'pip3',
            'git', 'curl', 'wget'
        ]
    
    async def execute(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        执行 Shell 命令。
        
        Args:
            command: 要执行的命令
            timeout: 超时时间
            
        Returns:
            执行结果字典
            
        Raises:
            ToolError: 命令执行失败
        """
        if not command or not command.strip():
            raise ToolError("Command cannot be empty")
        
        # 解析命令
        command_parts = command.strip().split()
        base_command = command_parts[0]
        
        # 检查命令是否被允许
        if base_command not in self.allowed_commands:
            raise ToolError(f"Command '{base_command}' is not allowed")
        
        execution_timeout = timeout or self.timeout
        
        try:
            start_time = time.time()
            
            # 执行命令
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_directory,
                timeout=execution_timeout,
                capture_output=True,
                text=True
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.stderr else None,
                'return_code': result.returncode,
                'execution_time': execution_time,
                'command': command
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f"Command timed out after {execution_timeout} seconds",
                'return_code': -1,
                'execution_time': execution_timeout,
                'command': command
            }
        except Exception as e:
            raise ToolError(f"Failed to execute command '{command}': {str(e)}")
    
    def execute_sync(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """同步版本的命令执行。"""
        import asyncio
        return asyncio.run(self.execute(command, timeout))
