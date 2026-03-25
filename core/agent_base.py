"""
智能体基类模块
定义所有智能体的通用接口和行为
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import os


class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


class AgentType(Enum):
    """智能体类型枚举"""
    DATA = auto()
    FACTOR = auto()
    STRATEGY = auto()
    BACKTEST = auto()
    RISK = auto()
    REPORT = auto()
    COORDINATOR = auto()


@dataclass
class Message:
    """智能体间消息传递结构"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default="")

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"{self.sender}_{self.receiver}_{int(self.timestamp.timestamp())}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"]
        )


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    def __bool__(self) -> bool:
        return self.success


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(self, name: str, agent_type: AgentType, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.subscribers: List[Callable[[Message], None]] = []
        self.logger = self._setup_logger()
        self.start_time: Optional[datetime] = None
        self.task_results: List[TaskResult] = []

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"agent.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def initialize(self) -> TaskResult:
        """初始化智能体"""
        pass

    @abstractmethod
    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行智能体主任务"""
        pass

    @abstractmethod
    def cleanup(self) -> TaskResult:
        """清理资源"""
        pass

    def run(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """运行智能体完整流程"""
        self.status = AgentStatus.INITIALIZING
        self.start_time = datetime.now()

        try:
            # 初始化
            init_result = self.initialize()
            if not init_result.success:
                self.status = AgentStatus.ERROR
                return init_result

            # 执行主任务
            self.status = AgentStatus.RUNNING
            exec_result = self.execute(input_data)

            if exec_result.success:
                self.status = AgentStatus.COMPLETED
            else:
                self.status = AgentStatus.ERROR

            self.task_results.append(exec_result)
            return exec_result

        except Exception as e:
            self.status = AgentStatus.ERROR
            error_result = TaskResult(
                success=False,
                errors=[f"Agent execution failed: {str(e)}"]
            )
            self.task_results.append(error_result)
            self.logger.error(f"Agent error: {str(e)}", exc_info=True)
            return error_result
        finally:
            # 清理资源
            self.cleanup()

    def send_message(self, message: Message) -> None:
        """发送消息到输出队列"""
        self.outbox.append(message)
        self.logger.debug(f"Sent message to {message.receiver}")
        # 通知订阅者
        for subscriber in self.subscribers:
            subscriber(message)

    def receive_message(self, message: Message) -> None:
        """接收消息到输入队列"""
        self.inbox.append(message)
        self.logger.debug(f"Received message from {message.sender}")

    def subscribe(self, callback: Callable[[Message], None]) -> None:
        """订阅消息"""
        self.subscribers.append(callback)

    def process_inbox(self) -> List[Message]:
        """处理所有输入消息并清空队列"""
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages

    def get_outbox(self) -> List[Message]:
        """获取输出消息并清空队列"""
        messages = self.outbox.copy()
        self.outbox.clear()
        return messages

    def save_state(self, filepath: str) -> None:
        """保存智能体状态"""
        state = {
            "name": self.name,
            "agent_type": self.agent_type.name,
            "status": self.status.name,
            "config": self.config,
            "task_results": [
                {
                    "success": r.success,
                    "data": r.data,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "execution_time": r.execution_time
                }
                for r in self.task_results
            ]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self, filepath: str) -> None:
        """加载智能体状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.config = state.get("config", {})
        self.task_results = [
            TaskResult(
                success=r["success"],
                data=r["data"],
                errors=r["errors"],
                warnings=r["warnings"],
                execution_time=r["execution_time"]
            )
            for r in state.get("task_results", [])
        ]

    def get_execution_time(self) -> float:
        """获取执行时间（秒）"""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status})"
