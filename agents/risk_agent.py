"""
风险管理智能体模块
负责风险分析和管理
"""
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import logging

from core.agent_base import BaseAgent, AgentType, TaskResult
from core.utils import save_json, ensure_directory_exists


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    volatility: float = 0.0
    annual_volatility: float = 0.0
    downside_risk: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    beta: float = 0.0
    correlation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0


@dataclass
class RiskAlert:
    """风险警报"""
    alert_type: str
    severity: str
    message: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class RiskAgent(BaseAgent):
    """
    风险智能体
    负责风险分析、计算和管理
    """

    def __init__(
        self,
        name: str = "RiskAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        super().__init__(name, AgentType.RISK, config)

        self.report_dir = config.get("report_dir", "reports")
        ensure_directory_exists(self.report_dir)

        # 默认风险阈值配置
        self.default_config = {
            "var_threshold": 0.02,
            "max_drawdown_threshold": 0.15,
            "volatility_threshold": 0.25,
            "beta_threshold": 1.2,
            "correlation_threshold": 0.7,
            "risk_free_rate": 0.03,
            "target_return": 0.0,
            "confidence_level": 0.95
        }
        self.config = {**self.default_config, **config}

        self.risk_alerts: List[RiskAlert] = []
        self.risk_history: List[RiskMetrics] = []

    def initialize(self) -> TaskResult:
        """初始化风险智能体"""
        self.logger.info("RiskAgent initialized")
        return TaskResult(success=True)

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行风险分析任务"""
        config = {**self.config, **(input_data or {})}

        try:
            # 获取数据
            returns, positions, backtest_result = self._get_input_data(input_data)

            if returns is None or len(returns) == 0:
                return TaskResult(
                    success=False,
                    errors=["No return data available"]
                )

            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(returns, positions, backtest_result)

            # 风险警报检查
            self.risk_alerts = self._check_risk_alerts(risk_metrics)

            # 保存风险报告
            report_path = self._save_risk_report(risk_metrics, self.risk_alerts)

            # 更新风险历史
            self.risk_history.append(risk_metrics)

            return TaskResult(
                success=True,
                data={
                    "risk_metrics": risk_metrics,
                    "risk_alerts": self.risk_alerts,
                    "report_path": report_path
                }
            )

        except Exception as e:
            error_msg = f"RiskAgent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return TaskResult(
                success=False,
                errors=[error_msg]
            )

    def _get_input_data(
        self,
        input_data: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], Optional[Dict]]:
        """获取输入数据"""
        if input_data is None:
            return None, None, None

        returns = None
        positions = None
        backtest_result = None

        # 尝试从不同来源获取数据
        if "returns" in input_data:
            returns = input_data["returns"]
        elif "from_BacktestAgent" in input_data:
            bt_data = input_data["from_BacktestAgent"]
            if "backtest_result" in bt_data:
                backtest_result = bt_data["backtest_result"]
                # 从回测结果中获取收益率
                if hasattr(backtest_result, 'equity_curve'):
                    returns = backtest_result.equity_curve.pct_change().dropna()

        elif "data" in input_data:
            data = input_data["data"]
            if "return" in data.columns:
                returns = data["return"]
            elif "close" in data.columns:
                returns = data["close"].pct_change().dropna()

        # 获取持仓信息
        if "positions" in input_data:
            positions = input_data["positions"]

        return returns, positions, backtest_result

    def _calculate_risk_metrics(
        self,
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
        backtest_result: Optional[Dict] = None
    ) -> RiskMetrics:
        """计算风险指标"""
        metrics = RiskMetrics()

        # 基本统计
        n = len(returns)
        if n == 0:
            return metrics

        # 波动率
        metrics.volatility = returns.std()
        metrics.annual_volatility = returns.std() * np.sqrt(252)

        # 下行风险
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics.downside_risk = downside_returns.std() * np.sqrt(252)

        # VaR 和 CVaR
        metrics.var_95 = -np.percentile(returns, 5)
        metrics.var_99 = -np.percentile(returns, 1)

        # 条件VaR (CVaR)
        returns_arr = returns.values
        mask_95 = returns_arr <= -metrics.var_95
        mask_99 = returns_arr <= -metrics.var_99
        if np.any(mask_95):
            metrics.cvar_95 = -np.mean(returns_arr[mask_95])
        if np.any(mask_99):
            metrics.cvar_99 = -np.mean(returns_arr[mask_99])

        # 偏度和峰度
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / running_max) - 1
        metrics.max_drawdown = drawdown.min()

        # 回撤持续时间
        metrics.drawdown_duration = self._calculate_drawdown_duration(drawdown)

        # 风险调整收益指标
        risk_free_rate = self.config.get("risk_free_rate", 0.03)
        target_return = self.config.get("target_return", 0.0)

        if metrics.annual_volatility > 0:
            metrics.sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / metrics.annual_volatility

        if metrics.downside_risk > 0:
            metrics.sortino_ratio = (returns.mean() * 252 - risk_free_rate) / metrics.downside_risk

        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = (returns.mean() * 252) / abs(metrics.max_drawdown)

        # Omega比率
        if target_return != 0:
            excess_returns = returns - target_return / 252
            positive_excess = excess_returns[excess_returns > 0].sum()
            negative_excess = -excess_returns[excess_returns < 0].sum()
            if negative_excess > 0:
                metrics.omega_ratio = positive_excess / negative_excess

        # 如果有回测结果，补充计算
        if backtest_result is not None:
            if hasattr(backtest_result, 'total_return'):
                if hasattr(backtest_result, 'max_drawdown'):
                    metrics.max_drawdown = min(metrics.max_drawdown, backtest_result.max_drawdown)
            if hasattr(backtest_result, 'sharpe_ratio'):
                metrics.sharpe_ratio = max(metrics.sharpe_ratio, backtest_result.sharpe_ratio)

        return metrics

    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最大回撤持续期（天数）"""
        is_in_drawdown = drawdown < 0
        if not is_in_drawdown.any():
            return 0

        # 标记回撤区间
        drawdown_intervals = (is_in_drawdown != is_in_drawdown.shift()).cumsum()
        drawdown_periods = drawdown_intervals[is_in_drawdown].value_counts()

        if not drawdown_periods.empty:
            return drawdown_periods.max()

        return 0

    def _check_risk_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """检查风险警报"""
        alerts = []

        thresholds = {
            "var_threshold": ("Value at Risk", metrics.var_95, "high"),
            "max_drawdown_threshold": ("Max Drawdown", abs(metrics.max_drawdown), "critical"),
            "volatility_threshold": ("Annual Volatility", metrics.annual_volatility, "medium"),
            "beta_threshold": ("Beta", metrics.beta, "medium"),
        }

        for key, (name, value, severity) in thresholds.items():
            threshold = self.config.get(key, 0.5)
            if value > threshold:
                alert = RiskAlert(
                    alert_type=name,
                    severity=severity,
                    message=f"{name} exceeded threshold: {value:.2%} > {threshold:.2%}",
                    threshold=threshold,
                    current_value=value
                )
                alerts.append(alert)
                self.logger.warning(f"Risk alert: {alert.message}")

        return alerts

    def _save_risk_report(
        self,
        metrics: RiskMetrics,
        alerts: List[RiskAlert]
    ) -> str:
        """保存风险报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risk_report_{timestamp}.json"
        save_path = str(Path(self.report_dir) / filename)

        report = {
            "metrics": {
                "var_95": metrics.var_95,
                "var_99": metrics.var_99,
                "cvar_95": metrics.cvar_95,
                "cvar_99": metrics.cvar_99,
                "volatility": metrics.volatility,
                "annual_volatility": metrics.annual_volatility,
                "downside_risk": metrics.downside_risk,
                "max_drawdown": metrics.max_drawdown,
                "drawdown_duration": metrics.drawdown_duration,
                "beta": metrics.beta,
                "skewness": metrics.skewness,
                "kurtosis": metrics.kurtosis,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "omega_ratio": metrics.omega_ratio
            },
            "alerts": [
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "threshold": a.threshold,
                    "current_value": a.current_value,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in alerts
            ],
            "timestamp": datetime.now().isoformat()
        }

        save_json(report, save_path)
        return save_path

    def get_risk_summary(self, metrics: Optional[RiskMetrics] = None) -> Dict[str, str]:
        """获取风险摘要"""
        if metrics is None:
            if not self.risk_history:
                return {}
            metrics = self.risk_history[-1]

        return {
            "VaR (95%)": f"{metrics.var_95:.2%}",
            "VaR (99%)": f"{metrics.var_99:.2%}",
            "CVaR (95%)": f"{metrics.cvar_95:.2%}",
            "Annual Volatility": f"{metrics.annual_volatility:.2%}",
            "Max Drawdown": f"{metrics.max_drawdown:.2%}",
            "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
            "Calmar Ratio": f"{metrics.calmar_ratio:.2f}"
        }

    def assess_risk_level(self, metrics: Optional[RiskMetrics] = None) -> str:
        """评估风险等级"""
        if metrics is None:
            if not self.risk_history:
                return "unknown"
            metrics = self.risk_history[-1]

        # 综合判断风险等级
        risk_score = 0
        risk_score += min(abs(metrics.max_drawdown) / 0.2, 2)
        risk_score += min(metrics.annual_volatility / 0.3, 2)
        risk_score += min(metrics.var_95 / 0.03, 1)
        risk_score += min(1 - metrics.sharpe_ratio, 1)

        if risk_score < 2:
            return "low"
        elif risk_score < 4:
            return "medium"
        else:
            return "high"

    def calculate_position_risk(
        self,
        positions: pd.DataFrame,
        returns: pd.Series
    ) -> Dict[str, float]:
        """计算持仓风险"""
        if positions.empty:
            return {}

        total_value = positions["value"].sum()
        weights = positions["value"] / total_value

        # 组合VaR（简化计算）
        portfolio_var = (weights * positions["value"].pct_change().dropna().std() * 1.645).sum()

        # 集中度风险
        herfindahl = (weights ** 2).sum()

        return {
            "portfolio_var": portfolio_var,
            "concentration": herfindahl,
            "position_count": len(positions),
            "total_value": total_value
        }

    def stress_test(
        self,
        returns: pd.Series,
        scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """压力测试"""
        scenarios = scenarios or self._default_stress_scenarios()

        results = {}
        for scenario in scenarios:
            name = scenario.get("name", "Unknown")
            shock = scenario.get("shock", 0)
            shocked_returns = returns + shock

            results[name] = {
                "shock": shock,
                "worst_return": shocked_returns.min(),
                "total_return": shocked_returns.sum(),
                "new_var_95": -np.percentile(shocked_returns, 5)
            }

        return results

    def _default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """默认压力测试场景"""
        return [
            {"name": "Normal", "shock": 0},
            {"name": "Mild Shock", "shock": -0.01},
            {"name": "Moderate Shock", "shock": -0.03},
            {"name": "Severe Shock", "shock": -0.05},
            {"name": "Extreme Shock", "shock": -0.1}
        ]

    def cleanup(self) -> TaskResult:
        """清理资源"""
        self.logger.info("RiskAgent cleaned up")
        return TaskResult(success=True)
