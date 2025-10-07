"""File system operation tools.

This module provides safe file read/write utilities with optional path
restrictions. When no explicit allowed paths are provided for the write tool,
new files will be written under the project's ``docs`` directory by default.
"""

import os

from .base import BaseTool, ToolError


def _find_project_root(start_path: str | None = None) -> str:
    """Best-effort project root detector.

    The function walks up from the given path (or current file) to locate a
    directory that looks like the project root by checking for common markers
    such as ``pyproject.toml`` or ``.git``. Falls back to the current working
    directory if no marker is found.
    """
    path = start_path or os.path.dirname(os.path.abspath(__file__))
    last = None
    while path and path != last:
        if (
            os.path.exists(os.path.join(path, "pyproject.toml"))
            or os.path.exists(os.path.join(path, ".git"))
            or os.path.exists(os.path.join(path, "README.md"))
        ):
            return path
        last = path
        path = os.path.dirname(path)

    # Fallback
    return os.getcwd()


class FileReadTool(BaseTool):
    """文件读取工具。"""

    def __init__(self, allowed_paths: list[str] | None = None):
        super().__init__(
            name="read_file",
            description="读取指定路径的文件内容",
            required_permissions=["file_system_access"],
            allowed_paths=allowed_paths,
        )
        self.allowed_paths = allowed_paths

    async def execute(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        读取文件内容。

        Args:
            file_path: 文件路径
            encoding: 文件编码，默认utf-8

        Returns:
            文件内容

        Raises:
            ToolError: 文件读取失败
        """
        try:
            # 路径安全检查
            if self.allowed_paths:
                abs_path = os.path.abspath(file_path)
                if not any(
                    abs_path.startswith(os.path.abspath(allowed)) for allowed in self.allowed_paths
                ):
                    raise ToolError(f"Access denied: {file_path} not in allowed paths")

            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise ToolError(f"File not found: {file_path}")

            if not os.path.isfile(file_path):
                raise ToolError(f"Path is not a file: {file_path}")

            # 读取文件
            with open(file_path, encoding=encoding) as f:
                content = f.read()

            return content

        except UnicodeDecodeError as e:
            raise ToolError(f"Failed to decode file {file_path}: {e!s}")
        except OSError as e:
            raise ToolError(f"Failed to read file {file_path}: {e!s}")

    def execute_sync(self, file_path: str, encoding: str = "utf-8") -> str:
        """同步版本的文件读取。"""
        import asyncio

        return asyncio.run(self.execute(file_path, encoding))


class FileWriteTool(BaseTool):
    """文件写入工具。"""

    def __init__(self, allowed_paths: list[str] | None = None):
        super().__init__(
            name="write_file",
            description="将内容写入指定路径的文件",
            required_permissions=["file_system_access"],
            allowed_paths=allowed_paths,
        )
        # Default to project docs directory when no explicit allowed paths
        if allowed_paths is None:
            project_root = _find_project_root()
            docs_dir = os.path.join(project_root, "docs")
            # Ensure docs directory exists up-front for clearer behavior
            os.makedirs(docs_dir, exist_ok=True)
            self.allowed_paths = [docs_dir]
            self._default_base_dir = docs_dir
        else:
            self.allowed_paths = allowed_paths
            self._default_base_dir = None

    async def execute(
        self, file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
    ) -> str:
        """
        写入文件内容。

        Args:
            file_path: 文件路径
            content: 要写入的内容
            encoding: 文件编码，默认utf-8
            create_dirs: 是否自动创建目录

        Returns:
            成功消息

        Raises:
            ToolError: 文件写入失败
        """
        try:
            # If running in default mode (no custom allowed_paths), resolve
            # relative paths under the project's docs directory.
            if self._default_base_dir and not os.path.isabs(file_path):
                file_path = os.path.join(self._default_base_dir, file_path)

            # 路径安全检查
            if self.allowed_paths:
                abs_path = os.path.abspath(file_path)
                if not any(
                    abs_path.startswith(os.path.abspath(allowed)) for allowed in self.allowed_paths
                ):
                    raise ToolError(f"Access denied: {file_path} not in allowed paths")

            # 创建目录（如果需要）
            if create_dirs:
                dir_path = os.path.dirname(file_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)

            # 写入文件
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)

            file_size = os.path.getsize(file_path)
            return f"成功写入文件 {file_path}，大小: {file_size} 字节"

        except OSError as e:
            raise ToolError(f"Failed to write file {file_path}: {e!s}")

    def execute_sync(
        self, file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
    ) -> str:
        """同步版本的文件写入。"""
        import asyncio

        return asyncio.run(self.execute(file_path, content, encoding, create_dirs))


class DirectoryListTool(BaseTool):
    """目录列表工具。"""

    def __init__(self, allowed_paths: list[str] | None = None):
        super().__init__(
            name="list_directory",
            description="列出指定目录的所有文件和子目录",
            required_permissions=["file_system_access"],
            allowed_paths=allowed_paths,
        )
        self.allowed_paths = allowed_paths

    async def execute(
        self,
        directory_path: str,
        include_hidden: bool = False,
        include_size: bool = True,
        recursive: bool = False,
    ) -> str:
        """
        列出目录内容。

        Args:
            directory_path: 目录路径
            include_hidden: 是否包含隐藏文件
            include_size: 是否包含文件大小信息
            recursive: 是否递归列出子目录

        Returns:
            目录内容的格式化字符串

        Raises:
            ToolError: 目录列表失败
        """
        try:
            # 路径安全检查
            if self.allowed_paths:
                abs_path = os.path.abspath(directory_path)
                if not any(
                    abs_path.startswith(os.path.abspath(allowed)) for allowed in self.allowed_paths
                ):
                    raise ToolError(f"Access denied: {directory_path} not in allowed paths")

            # 检查目录是否存在
            if not os.path.exists(directory_path):
                raise ToolError(f"Directory not found: {directory_path}")

            if not os.path.isdir(directory_path):
                raise ToolError(f"Path is not a directory: {directory_path}")

            # 列出目录内容
            items = []

            if recursive:
                for root, _dirs, files in os.walk(directory_path):
                    level = root.replace(directory_path, "").count(os.sep)
                    indent = " " * 2 * level

                    items.append(f"{indent}{os.path.basename(root)}/")

                    sub_indent = " " * 2 * (level + 1)
                    for file in files:
                        if not include_hidden and file.startswith("."):
                            continue

                        file_path = os.path.join(root, file)
                        if include_size:
                            try:
                                size = os.path.getsize(file_path)
                                items.append(f"{sub_indent}{file} ({size} bytes)")
                            except OSError:
                                items.append(f"{sub_indent}{file} (size unknown)")
                        else:
                            items.append(f"{sub_indent}{file}")
            else:
                entries = os.listdir(directory_path)
                if not include_hidden:
                    entries = [e for e in entries if not e.startswith(".")]

                entries.sort()

                for entry in entries:
                    entry_path = os.path.join(directory_path, entry)

                    if os.path.isdir(entry_path):
                        items.append(f"{entry}/")
                    else:
                        if include_size:
                            try:
                                size = os.path.getsize(entry_path)
                                items.append(f"{entry} ({size} bytes)")
                            except OSError:
                                items.append(f"{entry} (size unknown)")
                        else:
                            items.append(entry)

            if not items:
                return f"目录 {directory_path} 为空"

            return f"目录 {directory_path} 内容:\n" + "\n".join(items)

        except OSError as e:
            raise ToolError(f"Failed to list directory {directory_path}: {e!s}")

    def execute_sync(
        self,
        directory_path: str,
        include_hidden: bool = False,
        include_size: bool = True,
        recursive: bool = False,
    ) -> str:
        """同步版本的目录列表。"""
        import asyncio

        return asyncio.run(self.execute(directory_path, include_hidden, include_size, recursive))


class CreateDirectoryTool(BaseTool):
    """目录创建工具。"""

    def __init__(self, allowed_paths: list[str] | None = None):
        super().__init__(
            name="create_directory",
            description="创建指定路径的目录",
            required_permissions=["file_system_access"],
            allowed_paths=allowed_paths,
        )
        self.allowed_paths = allowed_paths

    async def execute(self, directory_path: str, exist_ok: bool = True) -> str:
        """
        创建目录。

        Args:
            directory_path: 目录路径
            exist_ok: 如果目录已存在是否报错

        Returns:
            成功消息

        Raises:
            ToolError: 目录创建失败
        """
        try:
            # 路径安全检查
            if self.allowed_paths:
                abs_path = os.path.abspath(directory_path)
                if not any(
                    abs_path.startswith(os.path.abspath(allowed)) for allowed in self.allowed_paths
                ):
                    raise ToolError(f"Access denied: {directory_path} not in allowed paths")

            # 创建目录
            os.makedirs(directory_path, exist_ok=exist_ok)

            return f"成功创建目录: {directory_path}"

        except FileExistsError:
            raise ToolError(f"Directory already exists: {directory_path}")
        except OSError as e:
            raise ToolError(f"Failed to create directory {directory_path}: {e!s}")

    def execute_sync(self, directory_path: str, exist_ok: bool = True) -> str:
        """同步版本的目录创建。"""
        import asyncio

        return asyncio.run(self.execute(directory_path, exist_ok))
