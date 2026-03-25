"""
因子智能体模块
负责技术指标和因子计算
"""
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import logging

from core.agent_base import BaseAgent, AgentType, TaskResult
from core.utils import save_data, ensure_directory_exists, standardize


@dataclass
class FactorDefinition:
    """因子定义"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    factor_type: str = "technical"
    category: str = "momentum"


class FactorAgent(BaseAgent):
    """
    因子智能体
    负责计算各种技术指标和因子
    """

    def __init__(
        self,
        name: str = "FactorAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        super().__init__(name, AgentType.FACTOR, config)

        self.factor_dir = config.get("factor_dir", "data/factors")
        ensure_directory_exists(self.factor_dir)

        # 默认配置
        self.default_config = {
            "factors": [
                "ma5", "ma10", "ma20", "ma60",
                "rsi", "macd", "kdj", "bollinger",
                "volatility", "momentum", "volume"
            ],
            "use_ta_lib": True,
            "standardize": True,
            "winsorize": True
        }
        self.config = {**self.default_config, **config}

        # 注册因子
        self.factors: Dict[str, FactorDefinition] = {}
        self._register_factors()

        self.calculated_factors = []

    def _register_factors(self) -> None:
        """注册所有因子"""
        # 均线类因子
        self._register_ma_factors()
        # 动量类因子
        self._register_momentum_factors()
        # 波动率类因子
        self._register_volatility_factors()
        # 成交量类因子
        self._register_volume_factors()
        # 技术指标类因子
        self._register_indicator_factors()

    def _register_ma_factors(self) -> None:
        """注册均线类因子"""
        for period in [5, 10, 20, 30, 60, 120, 250]:
            self.factors[f"ma{period}"] = FactorDefinition(
                name=f"MA{period}",
                description=f"{period}日均线",
                function=self._calculate_ma,
                parameters={"period": period},
                factor_type="technical",
                category="trend"
            )

        # EMA
        for period in [5, 10, 20, 60]:
            self.factors[f"ema{period}"] = FactorDefinition(
                name=f"EMA{period}",
                description=f"{period}日指数移动平均",
                function=self._calculate_ema,
                parameters={"period": period},
                factor_type="technical",
                category="trend"
            )

    def _register_momentum_factors(self) -> None:
        """注册动量类因子"""
        for period in [5, 10, 20, 60]:
            self.factors[f"momentum_{period}"] = FactorDefinition(
                name=f"Momentum_{period}",
                description=f"{period}日动量",
                function=self._calculate_momentum,
                parameters={"period": period},
                factor_type="technical",
                category="momentum"
            )

        self.factors["rsi"] = FactorDefinition(
            name="RSI",
            description="相对强弱指标",
            function=self._calculate_rsi,
            parameters={"period": 14},
            factor_type="technical",
            category="momentum"
        )

    def _register_volatility_factors(self) -> None:
        """注册波动率类因子"""
        for period in [5, 10, 20, 60]:
            self.factors[f"volatility_{period}"] = FactorDefinition(
                name=f"Volatility_{period}",
                description=f"{period}日波动率",
                function=self._calculate_volatility,
                parameters={"period": period},
                factor_type="technical",
                category="volatility"
            )

        self.factors["atr"] = FactorDefinition(
            name="ATR",
            description="平均真实波幅",
            function=self._calculate_atr,
            parameters={"period": 14},
            factor_type="technical",
            category="volatility"
        )

    def _register_volume_factors(self) -> None:
        """注册成交量类因子"""
        for period in [5, 10, 20]:
            self.factors[f"volume_ma{period}"] = FactorDefinition(
                name=f"Volume_MA{period}",
                description=f"{period}日成交量均线",
                function=self._calculate_volume_ma,
                parameters={"period": period},
                factor_type="technical",
                category="volume"
            )

        self.factors["obv"] = FactorDefinition(
            name="OBV",
            description="能量潮指标",
            function=self._calculate_obv,
            parameters={},
            factor_type="technical",
            category="volume"
        )

    def _register_indicator_factors(self) -> None:
        """注册技术指标类因子"""
        self.factors["macd"] = FactorDefinition(
            name="MACD",
            description="MACD指标",
            function=self._calculate_macd,
            parameters={"fast": 12, "slow": 26, "signal": 9},
            factor_type="technical",
            category="momentum"
        )

        self.factors["kdj"] = FactorDefinition(
            name="KDJ",
            description="KDJ指标",
            function=self._calculate_kdj,
            parameters={"n": 9, "m1": 3, "m2": 3},
            factor_type="technical",
            category="momentum"
        )

        self.factors["bollinger"] = FactorDefinition(
            name="Bollinger",
            description="布林带指标",
            function=self._calculate_bollinger,
            parameters={"period": 20, "std": 2},
            factor_type="technical",
            category="volatility"
        )

    def initialize(self) -> TaskResult:
        """初始化因子智能体"""
        self.logger.info(f"FactorAgent initialized with {len(self.factors)} factors")
        return TaskResult(success=True)

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行因子计算任务"""
        config = {**self.config, **(input_data or {})}

        try:
            # 获取数据
            data = self._get_input_data(input_data)
            if data is None or data.empty:
                return TaskResult(
                    success=False,
                    errors=["No input data provided or data is empty"]
                )

            self.logger.info(f"FactorAgent processing {len(data)} data points")

            # 计算因子
            data_with_factors = self._calculate_all_factors(data, config)

            # 因子预处理
            if config.get("standardize", False):
                data_with_factors = self._standardize_factors(data_with_factors)

            if config.get("winsorize", False):
                data_with_factors = self._winsorize_factors(data_with_factors)

            # 保存因子数据
            save_path = self._save_factor_data(data_with_factors, config)

            return TaskResult(
                success=True,
                data={
                    "factor_data": data_with_factors,
                    "file_path": save_path,
                    "factor_names": self.calculated_factors
                },
                execution_time=0.0
            )

        except Exception as e:
            error_msg = f"FactorAgent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return TaskResult(
                success=False,
                errors=[error_msg]
            )

    def _get_input_data(self, input_data: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """获取输入数据"""
        if input_data is None:
            return None

        # 尝试从不同的位置获取数据
        if "data" in input_data:
            return input_data["data"]
        elif "from_DataAgent" in input_data:
            return input_data["from_DataAgent"].get("data")
        elif "factor_data" in input_data:
            return input_data["factor_data"]

        return None

    def _calculate_all_factors(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """计算所有配置的因子"""
        data = data.copy()
        factor_names = config.get("factors", list(self.factors.keys()))

        self.calculated_factors = []

        for factor_name in factor_names:
            if factor_name in self.factors:
                try:
                    factor_def = self.factors[factor_name]
                    data = factor_def.function(data, **factor_def.parameters)
                    self.calculated_factors.append(factor_name)
                    self.logger.debug(f"Calculated factor: {factor_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to calculate factor {factor_name}: {e}")

        return data

    @staticmethod
    def _calculate_ma(data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算移动平均线"""
        data[f"ma{period}"] = data["close"].rolling(window=period).mean()
        return data

    @staticmethod
    def _calculate_ema(data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算指数移动平均"""
        data[f"ema{period}"] = data["close"].ewm(span=period, adjust=False).mean()
        return data

    @staticmethod
    def _calculate_momentum(data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算动量因子"""
        data[f"momentum_{period}"] = data["close"].pct_change(period)
        return data

    @staticmethod
    def _calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        return data

    @staticmethod
    def _calculate_volatility(data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算波动率"""
        data[f"volatility_{period}"] = data["close"].pct_change().rolling(window=period).std()
        return data

    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算ATR"""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data["atr"] = true_range.rolling(window=period).mean()
        return data

    @staticmethod
    def _calculate_volume_ma(data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算成交量均线"""
        data[f"volume_ma{period}"] = data["volume"].rolling(window=period).mean()
        return data

    @staticmethod
    def _calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
        """计算OBV指标"""
        obv = (np.sign(data["close"].diff()) * data["volume"]).fillna(0).cumsum()
        data["obv"] = obv
        return data

    @staticmethod
    def _calculate_macd(
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """计算MACD"""
        ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
        data["macd"] = ema_fast - ema_slow
        data["macd_signal"] = data["macd"].ewm(span=signal, adjust=False).mean()
        data["macd_histogram"] = data["macd"] - data["macd_signal"]
        return data

    @staticmethod
    def _calculate_kdj(
        data: pd.DataFrame,
        n: int = 9,
        m1: int = 3,
        m2: int = 3
    ) -> pd.DataFrame:
        """计算KDJ指标"""
        low_min = data["low"].rolling(window=n).min()
        high_max = data["high"].rolling(window=n).max()
        rsv = (data["close"] - low_min) / (high_max - low_min) * 100

        data["kdj_k"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
        data["kdj_d"] = data["kdj_k"].ewm(com=m2 - 1, adjust=False).mean()
        data["kdj_j"] = 3 * data["kdj_k"] - 2 * data["kdj_d"]
        return data

    @staticmethod
    def _calculate_bollinger(
        data: pd.DataFrame,
        period: int = 20,
        std: float = 2
    ) -> pd.DataFrame:
        """计算布林带"""
        data["bollinger_mid"] = data["close"].rolling(window=period).mean()
        std_dev = data["close"].rolling(window=period).std()
        data["bollinger_upper"] = data["bollinger_mid"] + (std_dev * std)
        data["bollinger_lower"] = data["bollinger_mid"] - (std_dev * std)
        data["bollinger_width"] = (data["bollinger_upper"] - data["bollinger_lower"]) / data["bollinger_mid"]
        return data

    def _standardize_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化因子"""
        factor_cols = [col for col in data.columns if col in self.calculated_factors]
        for col in factor_cols:
            data[f"{col}_std"] = standardize(data[col])
        return data

    def _winsorize_factors(
        self,
        data: pd.DataFrame,
        lower: float = 0.01,
        upper: float = 0.99
    ) -> pd.DataFrame:
        """去极值"""
        factor_cols = [col for col in data.columns if col in self.calculated_factors]
        for col in factor_cols:
            lower_val = data[col].quantile(lower)
            upper_val = data[col].quantile(upper)
            data[f"{col}_winsor"] = data[col].clip(lower_val, upper_val)
        return data

    def _save_factor_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """保存因子数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"factors_{timestamp}.csv"
        save_path = str(Path(self.factor_dir) / filename)
        save_data(data, save_path)
        return save_path

    def list_available_factors(self) -> List[Dict[str, Any]]:
        """列出可用因子"""
        return [
            {
                "name": name,
                "description": factor.description,
                "category": factor.category,
                "factor_type": factor.factor_type
            }
            for name, factor in self.factors.items()
        ]

    def register_custom_factor(
        self,
        name: str,
        function: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "custom"
    ) -> None:
        """注册自定义因子"""
        self.factors[name] = FactorDefinition(
            name=name,
            description=description,
            function=function,
            parameters=parameters or {},
            factor_type="custom",
            category=category
        )
        self.logger.info(f"Registered custom factor: {name}")

    def cleanup(self) -> TaskResult:
        """清理资源"""
        self.logger.info("FactorAgent cleaned up")
        return TaskResult(success=True)
