"""
流程管道模块
实现多智能体之间的协作和任务调度
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import logging
import json
from pathlib import Path

from .agent_base import (
    BaseAgent, AgentStatus, AgentType, Message, TaskResult
)


class PipelineStatus(Enum):
    """管道状态"""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()
    CANCELLED = auto()


@dataclass
class PipelineStep:
    """管道步骤"""
    name: str
    agent: BaseAgent
    input_data: Optional[Dict[str, Any]] = None
    depends_on: Optional[List[str]] = None
    enabled: bool = True
    result: Optional[TaskResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class PipelineResult:
    """管道执行结果"""
    success: bool
    step_results: Dict[str, TaskResult] = field(default_factory=dict)
    total_execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_data: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class Pipeline:
    """多智能体协作管道"""

    def __init__(self, name: str = "QuantResearchPipeline"):
        self.name = name
        self.steps: Dict[str, PipelineStep] = {}
        self.step_order: List[str] = []
        self.status = PipelineStatus.IDLE
        self.logger = self._setup_logger()
        self.start_time: Optional[datetime] = None
        self.callbacks: Dict[str, List[Callable]] = {
            "on_step_start": [],
            "on_step_complete": [],
            "on_pipeline_complete": [],
            "on_error": []
        }
        self._message_router: Dict[str, List[BaseAgent]] = {}

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"pipeline.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def add_step(
        self,
        name: str,
        agent: BaseAgent,
        input_data: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        position: Optional[int] = None
    ) -> 'Pipeline':
        """添加步骤到管道"""
        step = PipelineStep(
            name=name,
            agent=agent,
            input_data=input_data,
            depends_on=depends_on or []
        )
        self.steps[name] = step

        if position is not None and 0 <= position <= len(self.step_order):
            self.step_order.insert(position, name)
        else:
            self.step_order.append(name)

        # 设置消息路由
        self._setup_message_routing(agent)

        self.logger.info(f"Added step: {name} with agent {agent.name}")
        return self

    def _setup_message_routing(self, agent: BaseAgent) -> None:
        """设置智能体之间的消息路由"""
        # 订阅智能体的消息
        def callback(message: Message):
            self._route_message(message)

        agent.subscribe(callback)

    def _route_message(self, message: Message) -> None:
        """路由消息到目标智能体"""
        if message.receiver == "*":
            # 广播消息给所有智能体
            for step in self.steps.values():
                if step.agent.name != message.sender:
                    step.agent.receive_message(message)
        elif message.receiver in self.steps:
            # 发送给特定智能体
            self.steps[message.receiver].agent.receive_message(message)
        else:
            self.logger.warning(f"Message recipient not found: {message.receiver}")

    def on(self, event: str, callback: Callable) -> 'Pipeline':
        """注册事件回调"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        return self

    def _trigger_event(self, event: str, *args, **kwargs) -> None:
        """触发事件"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")

    def _get_execution_order(self) -> List[str]:
        """获取执行顺序（基于依赖关系拓扑排序）"""
        # 简单实现：按照添加顺序，检查依赖
        ordered = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            step = self.steps[name]
            for dep in step.depends_on:
                visit(dep)

            ordered.append(name)

        for name in self.step_order:
            visit(name)

        return ordered

    def run(
        self,
        initial_input: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = True
    ) -> PipelineResult:
        """运行管道"""
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()
        result = PipelineResult(success=True)

        self.logger.info("Pipeline started")

        execution_order = self._get_execution_order()
        self.logger.info(f"Execution order: {execution_order}")

        step_outputs: Dict[str, Dict[str, Any]] = {}

        for step_name in execution_order:
            step = self.steps[step_name]

            if not step.enabled:
                self.logger.info(f"Skipping disabled step: {step_name}")
                continue

            # 检查依赖
            for dep_name in step.depends_on:
                dep_step = self.steps.get(dep_name)
                if dep_step and dep_step.result and not dep_step.result.success:
                    error_msg = f"Dependency {dep_name} failed for step {step_name}"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                    if stop_on_error:
                        result.success = False
                        self.status = PipelineStatus.ERROR
                        return result
                    continue

            # 准备输入数据
            step_input = {}
            if initial_input:
                step_input.update(initial_input)
            if step.input_data:
                step_input.update(step.input_data)

            # 从依赖步骤获取输出
            for dep_name in step.depends_on:
                if dep_name in step_outputs:
                    step_input[f"from_{dep_name}"] = step_outputs[dep_name]

            self._trigger_event("on_step_start", step_name, step_input)

            # 执行步骤
            step.start_time = datetime.now()
            self.logger.info(f"Executing step: {step_name}")

            try:
                step_result = step.agent.run(step_input)
                step.result = step_result
                step.end_time = datetime.now()

                result.step_results[step_name] = step_result

                if step_result.success:
                    step_outputs[step_name] = step_result.data
                    result.output_data[step_name] = step_result.data

                    if step_result.warnings:
                        result.warnings.extend(step_result.warnings)

                    self.logger.info(f"Step {step_name} completed successfully")
                else:
                    result.errors.extend(step_result.errors)
                    self.logger.error(f"Step {step_name} failed: {step_result.errors}")
                    if stop_on_error:
                        result.success = False
                        self.status = PipelineStatus.ERROR
                        return result

            except Exception as e:
                error_msg = f"Step {step_name} exception: {str(e)}"
                result.errors.append(error_msg)
                self.logger.error(error_msg, exc_info=True)
                if stop_on_error:
                    result.success = False
                    self.status = PipelineStatus.ERROR
                    return result

            self._trigger_event("on_step_complete", step_name, step_result)

        # 完成
        result.total_execution_time = (datetime.now() - self.start_time).total_seconds()
        result.success = all(
            r.success for r in result.step_results.values()
        ) if result.step_results else False

        self.status = PipelineStatus.COMPLETED if result.success else PipelineStatus.ERROR
        self.logger.info(f"Pipeline completed. Success: {result.success}")

        self._trigger_event("on_pipeline_complete", result)

        return result

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """获取步骤"""
        return self.steps.get(name)

    def enable_step(self, name: str) -> 'Pipeline':
        """启用步骤"""
        if name in self.steps:
            self.steps[name].enabled = True
            self.logger.info(f"Enabled step: {name}")
        return self

    def disable_step(self, name: str) -> 'Pipeline':
        """禁用步骤"""
        if name in self.steps:
            self.steps[name].enabled = False
            self.logger.info(f"Disabled step: {name}")
        return self

    def reset(self) -> 'Pipeline':
        """重置管道"""
        for step in self.steps.values():
            step.result = None
            step.start_time = None
            step.end_time = None
            step.agent.status = AgentStatus.IDLE
        self.status = PipelineStatus.IDLE
        self.start_time = None
        self.logger.info("Pipeline reset")
        return self

    def save_summary(self, filepath: str) -> None:
        """保存管道执行摘要"""
        summary = {
            "pipeline_name": self.name,
            "status": self.status.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "steps": {
                name: {
                    "agent": step.agent.name,
                    "agent_type": step.agent.agent_type.name,
                    "enabled": step.enabled,
                    "depends_on": step.depends_on,
                    "success": step.result.success if step.result else None,
                    "execution_time": (step.end_time - step.start_time).total_seconds()
                    if step.start_time and step.end_time else None
                }
                for name, step in self.steps.items()
            }
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Pipeline summary saved to {filepath}")
