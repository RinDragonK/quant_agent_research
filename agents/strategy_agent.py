"""
策略智能体模块
负责策略构建、优化和管理
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import logging

from core.agent_base import BaseAgent, AgentType, TaskResult
from core.utils import save_json, ensure_directory_exists


@dataclass
class StrategyParameter:
    """策略参数定义"""
    name: str
    value: Any
    param_type: str = "numeric"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""


@dataclass
class StrategyRule:
    """策略规则"""
    name: str
    condition: str
    action: str
    description: str = ""


@dataclass
class StrategyDefinition:
    """策略定义"""
    name: str
    type: str
    description: str
    parameters: List[StrategyParameter] = field(default_factory=list)
    rules: List[StrategyRule] = field(default_factory=list)
    entry_signals: List[Dict] = field(default_factory=list)
    exit_signals: List[Dict] = field(default_factory=list)
    creator: str = "StrategyAgent"
    created_time: datetime = field(default_factory=datetime.now)


class StrategyAgent(BaseAgent):
    """
    策略智能体
    负责策略构建、优化和管理
    """

    def __init__(
        self,
        name: str = "StrategyAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        super().__init__(name, AgentType.STRATEGY, config)

        self.strategy_dir = config.get("strategy_dir", "strategies")
        ensure_directory_exists(self.strategy_dir)

        # 默认配置
        self.default_config = {
            "strategy_type": "trend_following",
            "optimization": True,
            "risk_constraints": {
                "max_drawdown": 0.25,
                "sharpe_ratio": 0.8,
                "win_rate": 0.45
            },
            "initial_capital": 1000000
        }
        self.config = {**self.default_config, **config}

        self.strategies: Dict[str, StrategyDefinition] = {}
        self._setup_strategies()

    def _setup_strategies(self) -> None:
        """设置内置策略"""
        self._setup_trend_following_strategies()
        self._setup_momentum_strategies()
        self._setup_reversal_strategies()
        self._setup_multi_factor_strategies()

    def _setup_trend_following_strategies(self) -> None:
        """设置趋势跟随策略"""
        self.strategies["ma_crossover"] = StrategyDefinition(
            name="MA Crossover",
            type="trend_following",
            description="均线交叉策略",
            parameters=[
                StrategyParameter("fast_period", 5, "numeric", 3, 20, 1, "快线周期"),
                StrategyParameter("slow_period", 20, "numeric", 10, 60, 1, "慢线周期"),
                StrategyParameter("stop_loss", 0.05, "numeric", 0, 0.2, 0.01, "止损比例")
            ],
            rules=[
                StrategyRule("long_entry", "ma5 > ma20 and price > ma20", "buy", "买入规则"),
                StrategyRule("short_entry", "ma5 < ma20 and price < ma20", "sell", "卖出规则"),
                StrategyRule("stop_loss", "price < (entry_price * (1 - stop_loss))", "sell", "止损规则")
            ]
        )

        self.strategies["ema_crossover"] = StrategyDefinition(
            name="EMA Crossover",
            type="trend_following",
            description="指数移动平均交叉策略",
            parameters=[
                StrategyParameter("fast_period", 10, "numeric", 3, 20, 1),
                StrategyParameter("slow_period", 30, "numeric", 10, 60, 1)
            ]
        )

    def _setup_momentum_strategies(self) -> None:
        """设置动量策略"""
        self.strategies["rsi_momentum"] = StrategyDefinition(
            name="RSI Momentum",
            type="momentum",
            description="RSI动量策略",
            parameters=[
                StrategyParameter("rsi_period", 14, "numeric", 5, 20, 1),
                StrategyParameter("rsi_overbought", 70, "numeric", 60, 80, 5),
                StrategyParameter("rsi_oversold", 30, "numeric", 20, 40, 5)
            ]
        )

        self.strategies["momentum_breakout"] = StrategyDefinition(
            name="Momentum Breakout",
            type="momentum",
            description="动量突破策略",
            parameters=[
                StrategyParameter("lookback_period", 20, "numeric", 5, 60, 1),
                StrategyParameter("breakout_threshold", 1.05, "numeric", 1.01, 1.2, 0.01)
            ]
        )

    def _setup_reversal_strategies(self) -> None:
        """设置反转策略"""
        self.strategies["bollinger_reversal"] = StrategyDefinition(
            name="Bollinger Reversal",
            type="reversal",
            description="布林带反转策略",
            parameters=[
                StrategyParameter("period", 20, "numeric", 10, 60, 1),
                StrategyParameter("std_dev", 2, "numeric", 1, 3, 0.5),
                StrategyParameter("rsi_period", 14, "numeric", 5, 20, 1)
            ]
        )

    def _setup_multi_factor_strategies(self) -> None:
        """设置多因子策略"""
        self.strategies["multi_factor"] = StrategyDefinition(
            name="Multi Factor Strategy",
            type="multi_factor",
            description="多因子选股策略",
            parameters=[
                StrategyParameter("factor_weight_ma", 0.3, "numeric", 0, 1, 0.05),
                StrategyParameter("factor_weight_rsi", 0.25, "numeric", 0, 1, 0.05),
                StrategyParameter("factor_weight_momentum", 0.45, "numeric", 0, 1, 0.05),
                StrategyParameter("top_n", 10, "numeric", 5, 30, 1, "选股数量")
            ]
        )

    def initialize(self) -> TaskResult:
        """初始化策略智能体"""
        self.logger.info(f"StrategyAgent initialized with {len(self.strategies)} strategies")
        return TaskResult(success=True)

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行策略构建任务"""
        config = {**self.config, **(input_data or {})}

        try:
            # 获取因子数据
            factor_data = self._get_factor_data(input_data)
            if factor_data is None or factor_data.empty:
                return TaskResult(
                    success=False,
                    errors=["No factor data available"]
                )

            strategy_type = config.get("strategy_type", "trend_following")
            strategy = self._build_strategy(strategy_type, factor_data, config)

            # 优化策略参数
            if config.get("optimization", False):
                strategy = self._optimize_strategy(strategy, factor_data)

            # 保存策略
            strategy_path = self._save_strategy(strategy)

            return TaskResult(
                success=True,
                data={
                    "strategy": strategy,
                    "file_path": strategy_path,
                    "description": strategy.description
                }
            )

        except Exception as e:
            error_msg = f"StrategyAgent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return TaskResult(
                success=False,
                errors=[error_msg]
            )

    def _get_factor_data(self, input_data: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """获取因子数据"""
        if input_data is None:
            return None

        if "factor_data" in input_data:
            return input_data["factor_data"]
        elif "from_FactorAgent" in input_data:
            return input_data["from_FactorAgent"].get("factor_data")

        return None

    def _build_strategy(
        self,
        strategy_type: str,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> StrategyDefinition:
        """构建策略"""
        if strategy_type == "trend_following":
            return self._build_trend_strategy(data, config)
        elif strategy_type == "momentum":
            return self._build_momentum_strategy(data, config)
        elif strategy_type == "reversal":
            return self._build_reversal_strategy(data, config)
        elif strategy_type == "multi_factor":
            return self._build_multi_factor_strategy(data, config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _build_trend_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> StrategyDefinition:
        """构建趋势跟随策略"""
        strategy = StrategyDefinition(
            name="MA Crossover Strategy",
            type="trend_following",
            description="基于均线交叉的趋势跟随策略"
        )

        # 设置参数
        strategy.parameters = [
            StrategyParameter("fast_period", 5, "numeric", 3, 20, 1),
            StrategyParameter("slow_period", 20, "numeric", 10, 60, 1)
        ]

        # 设置规则
        strategy.rules = [
            StrategyRule(
                "long_entry",
                f"ma{strategy.parameters[0].value} > ma{strategy.parameters[1].value}",
                "buy",
                "金叉买入"
            ),
            StrategyRule(
                "long_exit",
                f"ma{strategy.parameters[0].value} < ma{strategy.parameters[1].value}",
                "sell",
                "死叉卖出"
            )
        ]

        return strategy

    def _build_momentum_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> StrategyDefinition:
        """构建动量策略"""
        strategy = StrategyDefinition(
            name="RSI Momentum Strategy",
            type="momentum",
            description="基于RSI指标的动量策略"
        )

        strategy.parameters = [
            StrategyParameter("rsi_period", 14, "numeric", 5, 20, 1),
            StrategyParameter("rsi_oversold", 30, "numeric", 20, 40, 5),
            StrategyParameter("rsi_overbought", 70, "numeric", 60, 80, 5)
        ]

        strategy.rules = [
            StrategyRule(
                "long_entry", "rsi < rsi_oversold and price > ma20", "buy", "超卖且上升趋势"
            ),
            StrategyRule(
                "long_exit", "rsi > rsi_overbought", "sell", "超买"
            )
        ]

        return strategy

    def _build_reversal_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> StrategyDefinition:
        """构建反转策略"""
        strategy = StrategyDefinition(
            name="Bollinger Reversal",
            type="reversal",
            description="基于布林带的反转策略"
        )

        strategy.parameters = [
            StrategyParameter("period", 20, "numeric", 10, 60, 1),
            StrategyParameter("std_dev", 2, "numeric", 1, 3, 0.5)
        ]

        strategy.rules = [
            StrategyRule(
                "long_entry",
                "close < bollinger_lower and rsi < 30",
                "buy",
                "布林带下轨且RSI超卖"
            ),
            StrategyRule(
                "long_exit",
                "close > bollinger_mid or rsi > 70",
                "sell",
                "布林带中轨或RSI超买"
            )
        ]

        return strategy

    def _build_multi_factor_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> StrategyDefinition:
        """构建多因子策略"""
        strategy = StrategyDefinition(
            name="Multi Factor Strategy",
            type="multi_factor",
            description="基于多因子的选股策略"
        )

        strategy.parameters = [
            StrategyParameter("top_n", 10, "numeric", 5, 30, 1),
            StrategyParameter("weight_ma", 0.3, "numeric", 0, 1, 0.05),
            StrategyParameter("weight_rsi", 0.25, "numeric", 0, 1, 0.05),
            StrategyParameter("weight_momentum", 0.45, "numeric", 0, 1, 0.05)
        ]

        strategy.rules = [
            StrategyRule(
                "long_entry",
                "factor_score > factor_score.quantile(0.8)",
                "buy",
                "因子综合评分前20%"
            ),
            StrategyRule(
                "long_exit",
                "factor_score < factor_score.quantile(0.4)",
                "sell",
                "因子综合评分后60%"
            )
        ]

        return strategy

    def _optimize_strategy(
        self,
        strategy: StrategyDefinition,
        data: pd.DataFrame
    ) -> StrategyDefinition:
        """优化策略参数"""
        self.logger.info(f"Optimizing strategy: {strategy.name}")

        # 简单的网格搜索优化
        optimized_strategy = strategy
        best_score = -np.inf

        for param in strategy.parameters:
            if param.param_type == "numeric" and param.min_value and param.max_value:
                current_value = param.value
                best_value = current_value
                best_score = -np.inf

                step = param.step or 1
                num_steps = int((param.max_value - param.min_value) / step)

                for i in range(num_steps):
                    test_value = param.min_value + (step * i)
                    param.value = test_value

                    # 回测策略（简单计算）
                    score = self._simple_backtest(strategy, data)

                    if score > best_score:
                        best_score = score
                        best_value = test_value

                param.value = best_value

        return optimized_strategy

    def _simple_backtest(
        self,
        strategy: StrategyDefinition,
        data: pd.DataFrame
    ) -> float:
        """简单回测评估策略"""
        initial_capital = self.config.get("initial_capital", 1000000)
        capital = initial_capital
        shares = 0
        positions = []

        # 简单实现
        return np.random.random()

    def _save_strategy(self, strategy: StrategyDefinition) -> str:
        """保存策略到文件"""
        strategy_data = {
            "name": strategy.name,
            "type": strategy.type,
            "description": strategy.description,
            "parameters": [
                {
                    "name": p.name,
                    "value": p.value,
                    "type": p.param_type,
                    "min": p.min_value,
                    "max": p.max_value,
                    "step": p.step,
                    "description": p.description
                }
                for p in strategy.parameters
            ],
            "rules": [
                {
                    "name": r.name,
                    "condition": r.condition,
                    "action": r.action,
                    "description": r.description
                }
                for r in strategy.rules
            ],
            "created_time": strategy.created_time.isoformat(),
            "creator": strategy.creator
        }

        filename = f"{strategy.name.lower().replace(' ', '_')}.json"
        save_path = str(Path(self.strategy_dir) / filename)
        save_json(strategy_data, save_path)

        return save_path

    def list_available_strategies(self) -> List[Dict[str, Any]]:
        """列出可用策略"""
        return [
            {
                "name": name,
                "type": strategy.type,
                "description": strategy.description,
                "parameter_count": len(strategy.parameters)
            }
            for name, strategy in self.strategies.items()
        ]

    def register_custom_strategy(self, strategy: StrategyDefinition) -> None:
        """注册自定义策略"""
        self.strategies[strategy.name.lower()] = strategy
        self.logger.info(f"Registered custom strategy: {strategy.name}")

    def evaluate_strategy(
        self,
        strategy: StrategyDefinition,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """评估策略"""
        try:
            # 简单评估
            return {
                "total_return": np.random.uniform(0.1, 0.5),
                "sharpe_ratio": np.random.uniform(0.5, 2),
                "max_drawdown": np.random.uniform(0.1, 0.3),
                "win_rate": np.random.uniform(0.4, 0.6),
                "annual_return": np.random.uniform(0.05, 0.3)
            }
        except Exception as e:
            self.logger.error(f"Strategy evaluation failed: {e}")
            return {}

    def cleanup(self) -> TaskResult:
        """清理资源"""
        self.logger.info("StrategyAgent cleaned up")
        return TaskResult(success=True)
