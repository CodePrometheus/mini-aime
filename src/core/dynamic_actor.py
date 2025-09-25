"""支持 ReAct 范式与 Function Calling 的动态智能体实现。"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.tools import Tool

from ..llm.base import BaseLLMClient
from .models import ExecutionStep, TaskStatus


logger = logging.getLogger(__name__)
ACTOR_LOG_PREFIX = "MiniAime|Actor|"
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

class DynamicActor:
    """用于执行特定任务的自主智能体（采用 ReAct 范式）。"""
    
    def __init__(
        self,
        actor_id: str,
        task_id: str,
        task_description: str,
        llm_client: BaseLLMClient,
        tools: List[Tool],
        functions: List[Dict],
        system_prompt: str,
        config: Any,  # ActorConfiguration
    ):
        self.actor_id = actor_id
        self.task_id = task_id
        self.task_description = task_description
        self.llm = llm_client
        self.tools = tools
        self.functions = functions
        self.system_prompt = system_prompt
        self.config = config
        
        # 执行状态
        self.memory: List[ExecutionStep] = []
        self.status = TaskStatus.PENDING
        self.progress_manager = None
        self.start_time = None
        self.end_time = None
        
        # 工具映射
        self.tool_map = {tool.name: tool for tool in tools}
    
    async def execute(self, progress_manager) -> Dict[str, Any]:
        """
        执行任务的主循环。

        Args:
            progress_manager: 进度管理器

        Returns:
            执行结果字典
        """
        self.progress_manager = progress_manager
        self.start_time = datetime.now()
        self.status = TaskStatus.IN_PROGRESS
        logger.info(f"{ACTOR_LOG_PREFIX} actor_start actor={self.actor_id} task={self.task_id}")
        
        try:
            # 报告开始执行
            await self._report_progress("开始执行任务", "info")
            
            # 执行配置
            max_iterations = self.config.execution_config.get("max_iterations", 10)
            
            for step in range(max_iterations):
                logger.info(f"{ACTOR_LOG_PREFIX} react_step_begin actor={self.actor_id} step={step + 1}")
                # ReAct cycle: thought -> action -> observation
                try:
                    thought, action_result = await self._react_step()
                    
                    # 记录执行步骤
                    execution_step = ExecutionStep(
                        thought=thought,
                        action=action_result.get("action", ""),
                        observation=action_result.get("observation", ""),
                        step_id=f"{self.actor_id}_step_{step + 1}"
                    )
                    self.memory.append(execution_step)
                    
                    # 判断是否需要报告进度
                    if await self._should_report_progress(thought, action_result.get("observation", "")):
                        await self._report_progress(
                            f"步骤 {step + 1}: {action_result.get('observation', '')[:100]}",
                            "progress"
                        )
                    
                    # 判断任务是否完成
                    if await self._is_task_complete(action_result.get("observation", "")):
                        self.status = TaskStatus.COMPLETED
                        logger.info(f"{ACTOR_LOG_PREFIX} task_completed actor={self.actor_id} steps={len(self.memory)}")
                        break
                        
                except Exception as e:
                    # 错误处理
                    error_handled = await self._handle_error(e)
                    if not error_handled:
                        self.status = TaskStatus.FAILED
                        logger.error(f"{ACTOR_LOG_PREFIX} task_failed actor={self.actor_id} error={str(e)}")
                        break
            
            # 生成最终报告
            final_result = await self._generate_final_report()
            
            # 报告完成
            await self._report_progress("任务执行完成", "completed")
            logger.info(f"{ACTOR_LOG_PREFIX} actor_end actor={self.actor_id} status={self.status.value}")
            
            return final_result
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            await self._report_progress(f"任务执行失败: {str(e)}", "error")
            logger.error(f"{ACTOR_LOG_PREFIX} actor_exception actor={self.actor_id} error={str(e)}")
            return {"status": "failed", "error": str(e)}
        finally:
            self.end_time = datetime.now()
    
    async def _react_step(self) -> tuple[str, Dict[str, Any]]:
        """
        执行一个 ReAct 步骤：思考 -> 行动 -> 观察。

        Returns:
            (thought, action_result) 元组
        """
        # 构建对话消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"任务：{self.task_description}"}
        ]
        
        # 添加执行历史（最近3步）
        for step in self.memory[-3:]:
            messages.extend([
                {"role": "user", "content": f"之前的思考：{step.thought}\n之前的行动：{step.action}"},
                {"role": "assistant", "content": f"观察结果：{step.observation}"}
            ])
        
        # 当前推理请求
        messages.append({
            "role": "user",
            "content": "基于以上上下文，请进行下一步的思考和行动。如果任务已完成，请说明完成情况。"
        })
        
        try:
            # 调用 LLM（支持 Function Calling）
            if hasattr(self.llm, 'complete_with_functions'):
                # 如果 LLM 客户端支持 Function Calling
                logger.info(f"{ACTOR_LOG_PREFIX} llm_request actor={self.actor_id} mode=function_call")
                response = await self.llm.complete_with_functions(messages, self.functions)
            else:
                # 降级到普通对话
                logger.info(f"{ACTOR_LOG_PREFIX} llm_request actor={self.actor_id} mode=text")
                response = await self.llm.complete_with_context(messages)
            
            # 解析响应
            if isinstance(response, dict) and "function_call" in response:
                # 处理函数调用
                logger.info(f"{ACTOR_LOG_PREFIX} llm_response actor={self.actor_id} type=function_call")
                return await self._handle_function_call(response)
            else:
                # 处理普通响应
                logger.info(f"{ACTOR_LOG_PREFIX} llm_response actor={self.actor_id} type=text")
                return await self._handle_text_response(response)
                
        except Exception as e:
            # LLM 调用失败，返回错误
            thought = f"LLM调用失败: {str(e)}"
            action_result = {
                "action": "error",
                "observation": f"无法获取LLM响应: {str(e)}",
                "success": False
            }
            return thought, action_result
    
    async def _handle_function_call(self, response: Dict) -> tuple[str, Dict[str, Any]]:
        """处理函数调用响应。"""
        
        function_call = response["function_call"]
        function_name = function_call["name"]
        function_args = json.loads(function_call.get("arguments", "{}"))
        
        thought = f"决定调用工具 {function_name}"
        logger.info(f"{ACTOR_LOG_PREFIX} tool_call actor={self.actor_id} name={function_name}")
        
        # 执行工具
        if function_name in self.tool_map:
            try:
                tool = self.tool_map[function_name]
                
                # 调用工具（处理不同的参数格式）
                if len(function_args) == 1 and "input" in function_args:
                    # 单一输入参数
                    result = await tool.arun(function_args["input"])
                else:
                    # 多个命名参数
                    result = await tool.arun(**function_args)
                
                action_result = {
                    "action": f"调用工具 {function_name}({json.dumps(function_args, ensure_ascii=False)})",
                    "observation": str(result),
                    "success": True
                }
                
            except Exception as e:
                action_result = {
                    "action": f"调用工具 {function_name}",
                    "observation": f"工具执行失败: {str(e)}",
                    "success": False
                }
                logger.error(f"{ACTOR_LOG_PREFIX} tool_fail actor={self.actor_id} name={function_name} error={str(e)}")
        else:
            action_result = {
                "action": f"尝试调用工具 {function_name}",
                "observation": f"工具 {function_name} 不存在",
                "success": False
            }
        
        return thought, action_result
    
    async def _handle_text_response(self, response: str) -> tuple[str, Dict[str, Any]]:
        """处理文本响应（非函数调用）。"""
        
        # 尝试解析思考和行动
        lines = response.split('\n')
        thought = ""
        action = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("思考：") or line.startswith("Thought:"):
                thought = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("行动：") or line.startswith("Action:"):
                action = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        
        if not thought:
            thought = response[:100]  # 取前100字符作为思考
        
        if not action:
            action = "继续分析任务"
        
        # 如果没有明确的工具调用，作为思考记录
        action_result = {
            "action": action,
            "observation": "继续思考中，尚未调用具体工具",
            "success": True
        }
        
        logger.info(f"{ACTOR_LOG_PREFIX} text_step actor={self.actor_id} action={action}")
        return thought, action_result
    
    async def _should_report_progress(self, thought: str, observation: str) -> bool:
        """使用LLM判断是否应该报告进度。"""
        
        progress_prompt = f"""
        判断这个执行步骤是否值得向用户报告进度：
        
        思考过程：{thought}
        执行观察：{observation}
        
        考虑因素：
        - 是否取得了重要进展？
        - 是否遇到了问题或错误？
        - 用户是否会关心这个结果？
        - 是否是关键里程碑？
        
        返回JSON：
        {{
            "should_report": true/false,
            "reason": "报告/不报告的原因"
        }}
        """
        
        try:
            response = await self.llm.complete(progress_prompt)
            result = json.loads(response)
            return result.get("should_report", False)
        except Exception:
            # 降级策略：重要步骤总是报告
            return len(observation) > 20 and ("成功" in observation or "失败" in observation or "完成" in observation)
    
    async def _is_task_complete(self, observation: str) -> bool:
        """使用LLM判断任务是否完成。"""
        
        completion_prompt = f"""
        基于执行观察结果，判断任务是否已经完成：
        
        原始任务：{self.task_description}
        最新观察：{observation}
        执行历史：{len(self.memory)} 步已完成
        
        判断标准：
        - 任务目标是否达成？
        - 是否产生了预期的结果？
        - 还有重要步骤未完成吗？
        
        返回JSON：
        {{
            "is_complete": true/false,
            "completion_reason": "完成/未完成的具体原因",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.llm.complete(completion_prompt)
            result = json.loads(response)
            return result.get("is_complete", False)
        except Exception:
            # 降级策略：基于执行步数和观察内容的简单判断
            return len(self.memory) >= 3 and len(observation) > 50
    
    async def _handle_error(self, error: Exception) -> bool:
        """
        处理执行错误。
        
        Returns:
            是否成功处理错误（True 表示可以继续，False 表示需要停止）
        """
        error_msg = str(error)
        
        # 记录错误步骤
        error_step = ExecutionStep(
            thought=f"遇到错误: {error_msg}",
            action="error_handling",
            observation=f"错误详情: {error_msg}",
            step_id=f"{self.actor_id}_error_{len(self.memory)}"
        )
        self.memory.append(error_step)
        
        # 报告错误
        await self._report_progress(f"遇到错误: {error_msg[:100]}", "error")
        logger.error(f"{ACTOR_LOG_PREFIX} error actor={self.actor_id} msg={error_msg}")
        
        # 使用LLM分析错误并制定恢复策略
        recovery_prompt = f"""
        分析这个错误并制定恢复策略：
        
        错误信息：{error_msg}
        当前任务：{self.task_description}
        执行步数：{len(self.memory)}
        
        请分析：
        1. 这是什么类型的错误？
        2. 错误是否可以恢复？
        3. 应该采取什么恢复策略？
        
        返回JSON：
        {{
            "error_type": "network|permission|timeout|logic|unknown",
            "recoverable": true/false,
            "recovery_strategy": "retry|skip|abort|alternative_approach",
            "reasoning": "分析和建议的详细说明"
        }}
        """
        
        try:
            response = await self.llm.complete(recovery_prompt)
            recovery_plan = json.loads(response)
            
            if recovery_plan.get("recoverable", False):
                strategy = recovery_plan.get("recovery_strategy", "retry")
                reasoning = recovery_plan.get("reasoning", "")
                await self._report_progress(f"错误恢复策略: {strategy} - {reasoning}", "retry")
                logger.info(f"{ACTOR_LOG_PREFIX} recovery actor={self.actor_id} strategy={strategy}")
                return True
            else:
                reasoning = recovery_plan.get("reasoning", "无法恢复")
                await self._report_progress(f"无法恢复错误: {reasoning}", "error")
                logger.info(f"{ACTOR_LOG_PREFIX} unrecoverable actor={self.actor_id}")
                return False
                
        except Exception:
            # 降级策略：基于执行次数的简单判断
            if len(self.memory) < 10:
                await self._report_progress("尝试继续执行（降级恢复）", "retry")
                logger.info(f"{ACTOR_LOG_PREFIX} recovery_fallback actor={self.actor_id} mode=continue")
                return True
            else:
                await self._report_progress("执行步骤过多，停止尝试", "error")
                logger.info(f"{ACTOR_LOG_PREFIX} recovery_fallback actor={self.actor_id} mode=stop")
                return False
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """
        生成结构化最终报告。
        
        根据论文要求，最终报告包含三个基本部分：
        1. 状态更新 (Status Update)
        2. 结论摘要 (Conclusion Summary)
        3. 参考指针 (Reference Pointers)
        """
        
        execution_time = 0
        if self.start_time and self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()
        
        # 1. 状态更新 - 明确的任务状态
        status_update = {
            "task_id": self.task_id,
            "final_status": self.status.value,
            "completed": self.status == TaskStatus.COMPLETED,
            "execution_steps": len(self.memory),
            "execution_time_seconds": execution_time
        }
        
        # 2. 结论摘要 - 任务执行的叙述性总结
        conclusion_summary = await self._generate_conclusion_summary()
        
        # 3. 参考指针 - 关键工件的结构化指针集合
        reference_pointers = self._extract_reference_pointers()
        
        # 构建完整的结构化报告
        final_report = {
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            
            # 论文要求的三个核心部分
            "status_update": status_update,
            "conclusion_summary": conclusion_summary,
            "reference_pointers": reference_pointers,
            
            # 额外的有用信息
            "metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_steps": len(self.memory),
                "tools_used": list(set(
                    step.action.split("(")[0].replace("调用工具 ", "")
                    for step in self.memory
                    if "调用工具" in step.action
                ))
            }
        }
        
        # 提交最终报告给进度管理器
        if self.progress_manager:
            await self.progress_manager.submit_final_report(
                task_id=self.task_id,
                agent_id=self.actor_id,
                report=final_report
            )
        logger.info(f"{ACTOR_LOG_PREFIX} final_report actor={self.actor_id} steps={len(self.memory)}")
        
        return final_report
    
    async def _generate_conclusion_summary(self) -> Dict[str, str]:
        """生成任务执行的叙述性摘要。"""
        
        # 提取关键信息
        key_observations = []
        obstacles = []
        insights = []
        
        for step in self.memory:
            # 识别关键观察
            if any(kw in step.observation for kw in ["成功", "完成", "找到", "创建"]):
                key_observations.append(step.observation[:200])
            
            # 识别障碍
            if any(kw in step.observation for kw in ["失败", "错误", "无法", "问题"]):
                obstacles.append(step.observation[:200])
            
            # 识别洞察（重要发现）
            if any(kw in step.thought for kw in ["发现", "注意到", "意识到", "了解到"]):
                insights.append(step.thought[:200])
        
        # 构建摘要
        summary = {
            "final_outcome": key_observations[-1] if key_observations else "任务执行完成",
            "obstacles_encountered": obstacles[:3] if obstacles else [],
            "key_insights": insights[:3] if insights else [],
            "execution_summary": self._create_execution_narrative()
        }
        
        return summary
    
    def _create_execution_narrative(self) -> str:
        """创建执行过程的叙述性描述。"""
        
        if len(self.memory) == 0:
            return "任务未执行任何步骤"
        
        # 构建叙述
        narrative_parts = []
        
        # 开始
        narrative_parts.append(f"任务开始执行，目标是：{self.task_description}")
        
        # 中间过程（取关键步骤）
        if len(self.memory) > 2:
            mid_step = self.memory[len(self.memory) // 2]
            narrative_parts.append(f"执行过程中，{mid_step.thought}")
        
        # 结果
        final_step = self.memory[-1]
        narrative_parts.append(f"最终，{final_step.observation}")
        
        return " ".join(narrative_parts)
    
    def _extract_reference_pointers(self) -> Dict[str, Any]:
        """提取关键工件的结构化指针集合。"""
        
        pointers = {
            "files": [],
            "urls": [],
            "data_outputs": [],
            "tool_results": {}
        }
        
        # 从执行历史中提取工件
        for step in self.memory:
            observation = step.observation
            
            # 提取文件路径
            if any(kw in observation for kw in ["文件", "保存", "创建", ".txt", ".json", ".md"]):
                # 简单的文件路径提取（实际应该更智能）
                if "/" in observation or "\\" in observation:
                    pointers["files"].append({
                        "step_id": step.step_id,
                        "description": observation[:100]
                    })
            
            # 提取URL
            if "http" in observation or "www." in observation:
                pointers["urls"].append({
                    "step_id": step.step_id,
                    "description": observation[:100]
                })
            
            # 提取数据输出
            if "{" in observation and "}" in observation:
                try:
                    # 尝试解析JSON数据
                    json_start = observation.index("{")
                    json_end = observation.rindex("}") + 1
                    json_str = observation[json_start:json_end]
                    data = json.loads(json_str)
                    pointers["data_outputs"].append({
                        "step_id": step.step_id,
                        "data_preview": str(data)[:200]
                    })
                except (ValueError, json.JSONDecodeError):
                    pass
            
            # 记录工具调用结果
            if "调用工具" in step.action:
                tool_name = step.action.split("调用工具 ")[1].split("(")[0]
                if tool_name not in pointers["tool_results"]:
                    pointers["tool_results"][tool_name] = []
                pointers["tool_results"][tool_name].append({
                    "step_id": step.step_id,
                    "result_preview": step.observation[:100]
                })
        
        return pointers
    
    async def _report_progress(self, message: str, level: str = "info"):
        """报告进度给进度管理器。"""
        
        if self.progress_manager:
            await self.progress_manager.update_progress(
                task_id=self.task_id,
                agent_id=self.actor_id,
                status=self.status.value,
                message=message,
                details={
                    "level": level,
                    "step_count": len(self.memory),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态。"""
        
        return {
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "steps_completed": len(self.memory),
            "tools_available": len(self.tools),
            "execution_time": (
                (datetime.now() - self.start_time).total_seconds() 
                if self.start_time else 0
            )
        }
