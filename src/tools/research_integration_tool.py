"""研究整合工具 - 智能发现和整合子任务的研究成果。"""

import json
import logging
import os
import re
from typing import Any

from .base import BaseTool, ToolError

logger = logging.getLogger(__name__)


class ResearchIntegrationTool(BaseTool):
    """研究整合工具，用于智能发现和整合子任务的研究成果。"""

    def __init__(self, allowed_paths: list[str] | None = None):
        super().__init__(
            name="integrate_research",
            description="智能发现和整合任务目录下的所有研究文件，生成综合报告",
            required_permissions=["file_system_access"],
            allowed_paths=allowed_paths,
        )
        self.allowed_paths = allowed_paths
        self._docs_base_dir = self._find_docs_dir()

    def _find_docs_dir(self) -> str:
        path = os.path.dirname(os.path.abspath(__file__))
        while path and path != os.path.dirname(path):
            docs_path = os.path.join(path, "docs")
            if os.path.exists(docs_path):
                return docs_path
            if os.path.exists(os.path.join(path, "pyproject.toml")):
                docs_path = os.path.join(path, "docs")
                os.makedirs(docs_path, exist_ok=True)
                return docs_path
            path = os.path.dirname(path)
        return os.path.join(os.getcwd(), "docs")

    def _resolve_task_directory(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self._docs_base_dir, path)

    async def execute(
        self, task_directory: str, output_file: str, original_goal: str = "", llm_client=None
    ) -> str:
        """
        整合研究文件并生成综合报告。

        Args:
            task_directory: 任务目录路径
            output_file: 输出文件路径
            original_goal: 原始目标描述
            llm_client: LLM客户端（用于内容分析）

        Returns:
            整合结果的描述
        """
        try:
            # 1. 发现研究文件
            research_files = await self._discover_research_files(task_directory)

            if not research_files:
                raise ToolError(f"在目录 {task_directory} 中未找到任何研究文件")

            # 2. 读取并分析研究内容
            research_content = await self._read_and_analyze_research(research_files, llm_client)

            # 3. 生成综合报告
            final_report = await self._generate_comprehensive_report(
                research_content, original_goal, llm_client
            )

            # 4. 保存报告
            await self._save_report(final_report, output_file)

            return f"成功整合 {len(research_files)} 个研究文件，生成综合报告: {output_file}"

        except Exception as e:
            raise ToolError(f"研究整合失败: {e!s}")

    async def _discover_research_files(self, task_directory: str) -> list[str]:
        """发现任务目录下的所有研究文件。"""
        try:
            resolved_dir = self._resolve_task_directory(task_directory)
            logger.info(f"ResearchIntegrationTool: 解析后的目录路径: {resolved_dir}")
            
            if not os.path.exists(resolved_dir):
                raise ToolError(f"目录不存在: {task_directory} (解析后: {resolved_dir})")

            if not os.path.isdir(resolved_dir):
                raise ToolError(f"路径不是目录: {task_directory}")

            # 获取所有文件
            all_files = []
            for root, _dirs, files in os.walk(resolved_dir):
                for file in files:
                    if file.endswith((".md", ".json")):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)

            logger.info(f"ResearchIntegrationTool: 发现的所有文件: {all_files}")

            # 过滤掉最终报告文件
            research_files = []
            for file_path in all_files:
                filename = os.path.basename(file_path)
                # 排除 final_report*.md 文件
                if not filename.startswith("final_report"):
                    research_files.append(file_path)

            logger.info(f"ResearchIntegrationTool: 过滤后的研究文件: {research_files}")
            
            # 如果发现的文件数量异常少，尝试等待并重新扫描
            if len(research_files) < 2:
                logger.warning(f"ResearchIntegrationTool: 只发现 {len(research_files)} 个研究文件，可能存在文件生成延迟")
                import time
                time.sleep(2)  # 等待2秒
                
                # 重新扫描
                all_files = []
                for root, _dirs, files in os.walk(resolved_dir):
                    for file in files:
                        if file.endswith((".md", ".json")):
                            file_path = os.path.join(root, file)
                            all_files.append(file_path)
                
                research_files = []
                for file_path in all_files:
                    filename = os.path.basename(file_path)
                    if not filename.startswith("final_report"):
                        research_files.append(file_path)
                
                logger.info(f"ResearchIntegrationTool: 重新扫描后的研究文件: {research_files}")

            return research_files

        except OSError as e:
            raise ToolError(f"文件发现失败: {e!s}")

    async def _read_and_analyze_research(
        self, file_paths: list[str], llm_client=None
    ) -> dict[str, Any]:
        """读取并分析研究文件内容。"""
        research_data = {
            "files": [],
            "content_summary": {},
            "key_findings": [],
            "structured_data": {},
        }

        for file_path in file_paths:
            try:
                # 读取文件内容
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                filename = os.path.basename(file_path)

                # 分析文件内容
                analysis = await self._analyze_file_content(filename, content, llm_client)

                research_data["files"].append(
                    {
                        "path": file_path,
                        "filename": filename,
                        "content": content,
                        "analysis": analysis,
                    }
                )

                # 提取关键信息
                if analysis.get("key_findings"):
                    research_data["key_findings"].extend(analysis["key_findings"])

                if analysis.get("structured_data"):
                    research_data["structured_data"][filename] = analysis["structured_data"]

            except Exception as e:
                print(f"警告: 无法读取文件 {file_path}: {e}")
                continue

        return research_data

    async def _analyze_file_content(
        self, filename: str, content: str, llm_client=None
    ) -> dict[str, Any]:
        """使用LLM分析文件内容。"""
        if not llm_client:
            # 简单的关键词提取作为后备
            return self._simple_content_analysis(filename, content)

        try:
            # 使用专门的JSON模式调用LLM
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的研究内容分析助手，必须返回有效的JSON格式响应。",
                },
                {
                    "role": "user",
                    "content": f"""
分析以下研究文件的内容，提取关键信息：

文件名: {filename}
内容: {content[:3000]}...

请提取：
1. 关键发现和结论
2. 具体数据（价格、时间、地点等）
3. 重要建议和注意事项
4. 文件类型和主题

特别注意：
- 如果是JSON文件，深度提取所有结构化数据（价格、时间、地点、政策等）
- 如果是Markdown文件，提取所有文本信息和格式
- 确保不遗漏任何重要的价格、费用、门票信息

返回JSON格式：
{{
    "file_type": "文件类型",
    "theme": "主题",
    "key_findings": ["发现1", "发现2"],
    "specific_data": {{"价格": "xxx", "时间": "xxx"}},
    "recommendations": ["建议1", "建议2"],
    "summary": "内容摘要"
}}
""",
                },
            ]

            response = await llm_client.complete_chat_json(messages)

            # 验证响应结构
            if not isinstance(response, dict):
                raise ValueError("LLM response is not a dictionary")

            return response

        except Exception as e:
            print(f"LLM分析失败，使用简单分析: {e}")
            return self._simple_content_analysis(filename, content)

    def _simple_content_analysis(self, filename: str, content: str) -> dict[str, Any]:
        """简单的关键词分析作为后备方案。"""
        analysis = {
            "file_type": "research",
            "theme": filename.replace(".md", "").replace(".json", ""),
            "key_findings": [],
            "specific_data": {},
            "recommendations": [],
            "summary": content[:200] + "..." if len(content) > 200 else content,
        }

        # 提取价格信息
        price_patterns = [
            r"(\d+)\s*澳元",
            r"(\d+)\s*元",
            r"价格[：:]\s*(\d+)",
            r"费用[：:]\s*(\d+)",
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis["specific_data"]["价格"] = matches[0]
                break

        # 提取时间信息
        time_patterns = [r"(\d+)\s*小时", r"(\d+)\s*天", r"(\d+)\s*周", r"(\d+)\s*月"]

        for pattern in time_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis["specific_data"]["时间"] = matches[0]
                break

        return analysis

    async def _generate_comprehensive_report(
        self, research_data: dict[str, Any], original_goal: str, llm_client=None
    ) -> str:
        """生成综合报告。"""
        if not llm_client:
            return await self._generate_simple_report(research_data, original_goal, llm_client)

        try:
            research_content = self._extract_research_content(research_data)
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的研究报告整合专家，擅长深度分析研究资料，根据用户目标生成详尽、准确、实用的综合报告。你必须充分利用所有研究数据，不遗漏任何重要信息。",
                },
                {
                    "role": "user",
                    "content": f"""
请基于以下完整的研究资料，为用户生成一份详尽的综合报告。

**用户目标**：{original_goal}

===== 完整研究资料 =====
{research_content}
===== 研究资料结束 =====

## 你的任务

深度分析以上所有研究资料，理解用户的真实需求，生成一份完整、详细、准确的Markdown格式综合报告。

## 报告生成要求

### 1. 智能理解用户意图
- 仔细分析用户目标，理解任务类型（旅行规划、技术研究、学习计划、商业分析、产品对比等）
- 根据任务类型，自动调整报告结构和内容重点
- 确保报告内容与用户目标高度相关

### 2. 充分利用研究数据
- **完整性**：提取研究资料中的所有关键信息、数据、发现、建议
- **准确性**：所有数据、时间、价格、名称等必须来自研究资料，不要编造
- **系统性**：将分散的信息进行逻辑性整合，形成连贯的知识体系
- **深度性**：不仅罗列信息，还要提供分析、对比、建议
- **🔥 JSON数据重点**：特别关注JSON文件中的结构化数据，如价格、时间、地点等，必须详细提取并整合到报告中
- **🚨 强制要求**：每个子任务的研究结果都必须在报告中体现，不能遗漏任何重要信息
- **📊 工具调用结果**：所有工具调用（如brave_search、write_file等）产生的数据都必须充分利用

### 3. 生成高质量报告结构

根据用户目标和研究内容，智能组织报告结构。参考但不限于以下框架：

**通用结构框架**：
- 概述和背景
- 核心发现/主要内容
- 详细分析（分主题、分阶段、分类别等）
- 数据对比和评估
- 实施建议/行动方案
- 注意事项和风险
- 总结和展望

**根据具体任务类型调整**：
- 旅行计划 → 行程安排、交通住宿、预算、实用信息
- 技术研究 → 技术方案、实现细节、性能对比、最佳实践
- 学习计划 → 学习路线、时间安排、资源推荐、评估方式
- 商业分析 → 市场分析、竞争格局、SWOT、战略建议
- 产品对比 → 功能对比、价格分析、适用场景、选购建议

### 4. 格式和呈现

- **Markdown格式**：使用清晰的标题层级（# ## ###）、列表（- *）、表格等
- **结构化**：信息分类明确，层次清晰，易于阅读和查找
- **专业性**：语言准确、逻辑严密、表达清晰
- **实用性**：提供可操作的建议，而不是泛泛而谈
- **完整性**：报告应该是自包含的，读者无需查阅原始资料即可理解全部内容

### 5. 质量标准

- ✅ 报告标题准确反映用户目标
- ✅ 内容完整覆盖所有研究发现
- ✅ 数据准确，来源于研究资料
- ✅ 结构合理，逻辑清晰
- ✅ 建议具体，可操作性强
- ✅ 无冗余信息，无遗漏要点
- ✅ **🔥 必须包含所有子任务结果**：每个子任务的研究成果都要在报告中体现
- ✅ **📊 必须利用所有工具调用数据**：JSON文件、搜索结果、分析数据等都要充分利用
- ✅ **💰 价格信息必须详细**：所有价格、费用、门票信息都要具体列出
- ✅ **📍 地点信息必须完整**：所有景点、城市、地址信息都要包含

## 输出格式

直接输出完整的Markdown格式报告，以 `# [标题]` 开头。
不要包含任何解释性的前言、后记或元信息。

## 🚨 生成前强制检查清单

在生成报告前，请确认：
1. ✅ 是否包含了所有子任务的研究结果？
2. ✅ 是否充分利用了所有JSON文件中的价格、时间、地点数据？
3. ✅ 是否提取了所有工具调用产生的具体信息？
4. ✅ 是否遗漏了任何重要的价格、费用、门票信息？
5. ✅ 是否遗漏了任何景点、城市、地址信息？
6. ✅ 报告是否真正实用，用户能否直接使用？

**记住：用户需要的是完整、详细、实用的报告，不是简单的信息罗列！**

现在请开始生成这份详尽的综合报告。
""",
                },
            ]

            response = await llm_client.complete_with_context(messages)
            
            if isinstance(response, dict):
                return response.get("content", str(response))
            
            return str(response)

        except Exception as e:
            print(f"LLM报告生成失败，使用简单报告: {e}")
            return await self._generate_simple_report(research_data, original_goal, llm_client)
    
    def _extract_research_content(self, research_data: dict[str, Any]) -> str:
        """提取研究文件的原始内容。"""
        content_parts = []
        
        for file_info in research_data.get("files", []):
            filename = file_info.get("filename", "未知文件")
            content = file_info.get("content", "")
            
            if content.strip():
                content_parts.append(f"## 文件：{filename}\n\n{content}\n\n---\n")
        
        return "\n".join(content_parts) if content_parts else "无研究数据"

    async def _generate_simple_report(
        self, research_data: dict[str, Any], original_goal: str, llm_client=None
    ) -> str:
        """生成简单的综合报告。"""
        # 使用LLM智能提取标题
        title = await self._extract_title_with_llm(original_goal, llm_client)

        report_lines = [
            f"# {title}",
            "",
            f"**目标**: {original_goal}",
            "",
            "## 计划概述",
            "",
        ]

        # 提取关键信息
        key_findings = []
        specific_data = {}

        for file_info in research_data["files"]:
            analysis = file_info.get("analysis", {})
            if analysis.get("key_findings"):
                key_findings.extend(analysis["key_findings"])
            if analysis.get("specific_data"):
                specific_data.update(analysis["specific_data"])

        # 添加关键发现
        if key_findings:
            report_lines.extend(["## 主要发现", ""])
            for finding in key_findings[:10]:  # 限制数量
                report_lines.append(f"- {finding}")
            report_lines.append("")

        # 添加具体数据
        if specific_data:
            report_lines.extend(["## 重要信息", ""])
            for key, value in specific_data.items():
                report_lines.append(f"- **{key}**: {value}")
            report_lines.append("")

        # 添加文件内容摘要
        report_lines.extend(["## 详细计划", ""])
        for file_info in research_data["files"]:
            filename = file_info["filename"]
            content = file_info["content"]

            # 提取文件标题
            lines = content.split("\n")
            file_title = lines[0] if lines and lines[0].startswith("#") else filename
            if file_title.startswith("#"):
                file_title = file_title[1:].strip()

            report_lines.append(f"### {file_title}")
            report_lines.append("")

            # 添加内容摘要（前几段）
            paragraphs = content.split("\n\n")
            for para in paragraphs[:3]:  # 只取前3段
                if para.strip() and not para.startswith("#"):
                    report_lines.append(para.strip())
                    report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        return "\n".join(report_lines)

    def _extract_title_from_goal(self, goal: str) -> str:
        """从目标中提取合适的标题。"""
        # 如果目标很短，直接使用
        if len(goal) <= 20:
            return goal

        # 简单的关键词提取作为后备
        goal_clean = (
            goal.replace("生成", "")
            .replace("制定", "")
            .replace("创建", "")
            .replace("的", "")
            .strip()
        )

        # 如果目标包含"报告"，提取报告前的部分
        if "报告" in goal_clean:
            goal_clean = goal_clean.split("报告")[0].strip()

        # 如果目标包含"计划"，提取计划前的部分
        if "计划" in goal_clean:
            goal_clean = goal_clean.split("计划")[0].strip()

        # 如果目标太长，截取前30个字符
        if len(goal_clean) > 30:
            goal_clean = goal_clean[:30] + "..."

        # 如果提取失败，使用默认标题
        if not goal_clean:
            goal_clean = "综合计划报告"

        return goal_clean

    async def _extract_title_with_llm(self, goal: str, llm_client=None) -> str:
        """使用LLM智能提取标题。"""
        if not llm_client:
            return self._extract_title_from_goal(goal)

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的标题生成助手，能够从用户目标中提取简洁、准确的标题。",
                },
                {
                    "role": "user",
                    "content": f"""
请从以下用户目标中提取一个简洁、准确的标题（不超过20个字符）：

用户目标: {goal}

要求：
1. 标题应该简洁明了，不超过20个字符
2. 去除"生成"、"制定"、"创建"等动词
3. 保留核心内容
4. 如果目标是关于报告或计划，提取主要内容部分
5. 只返回标题，不要其他解释

示例：
- "生成三个月学习计划报告" → "三个月学习计划"
- "制定公司年度营销策略" → "公司年度营销策略"
- "创建用户使用手册" → "用户使用手册"
""",
                },
            ]

            response = await llm_client.complete_chat_json(messages)

            if isinstance(response, dict) and "content" in response:
                title = response["content"].strip()
                # 确保标题不会太长
                if len(title) > 30:
                    title = title[:30] + "..."
                return title
            else:
                return self._extract_title_from_goal(goal)

        except Exception as e:
            print(f"LLM标题提取失败，使用简单方法: {e}")
            return self._extract_title_from_goal(goal)

    async def _save_report(self, content: str, output_file: str) -> None:
        """保存报告到文件。"""
        try:
            resolved_output = self._resolve_task_directory(output_file)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(resolved_output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 写入文件
            with open(resolved_output, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            raise ToolError(f"保存报告失败: {e!s}")

    def execute_sync(
        self, task_directory: str, output_file: str, original_goal: str = "", llm_client=None
    ) -> str:
        """同步版本的研究整合。"""
        import asyncio

        return asyncio.run(self.execute(task_directory, output_file, original_goal, llm_client))
