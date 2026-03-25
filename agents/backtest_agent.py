"""
回测智能体模块
负责回测执行和性能评估
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import backtrader as bt
import vectorbt as vbt
import logging

from core.agent_base import BaseAgent, AgentType, TaskResult
from core.utils import save_data, ensure_directory_exists


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000
    commission: float = 0.0003
    slippage: float = 0.001
    margin: float = 0.1
    frequency: str = "1d"
    start_date: str = (datetime.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    strategy_type: str = "trend_following"


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: int = 0
    holding_period: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    drawdown_details: Dict = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestAgent(BaseAgent):
    """
    回测智能体
    负责回测执行和性能评估
    """

    def __init__(
        self,
        name: str = "BacktestAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        super().__init__(name, AgentType.BACKTEST, config)

        self.backtest_dir = config.get("backtest_dir", "backtests")
        ensure_directory_exists(self.backtest_dir)

        self.default_config = {
            "engine": "backtrader",
            "optimization": False,
            "benchmark": "000300.SH",
            "report_format": ["csv", "html"]
        }
        self.config = {**self.default_config, **config}

        self.backtest_results: Dict[str, BacktestResult] = {}
        self.current_result: Optional[BacktestResult] = None

    def initialize(self) -> TaskResult:
        """初始化回测智能体"""
        self.logger.info("BacktestAgent initialized")
        return TaskResult(success=True)

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行回测任务"""
        config = {**self.config, **(input_data or {})}

        try:
            # 获取输入数据
            data, strategy = self._get_input_data(input_data)

            if data is None or data.empty:
                return TaskResult(
                    success=False,
                    errors=["No data available for backtest"]
                )

            # 创建回测配置
            bt_config = BacktestConfig(**config)

            # 执行回测
            if self.config.get("engine") == "vectorbt":
                result = self._run_vectorbt_backtest(data, strategy, bt_config)
            else:
                result = self._run_backtrader_backtest(data, strategy, bt_config)

            # 保存结果
            save_path = self._save_backtest_result(result, config)

            self.current_result = result
            self.backtest_results[save_path] = result

            return TaskResult(
                success=True,
                data={
                    "backtest_result": result,
                    "file_path": save_path,
                    "config": bt_config
                }
            )

        except Exception as e:
            error_msg = f"BacktestAgent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return TaskResult(
                success=False,
                errors=[error_msg]
            )

    def _get_input_data(
        self,
        input_data: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """获取输入数据"""
        if input_data is None:
            return None, None

        # 从不同来源获取数据
        if "data" in input_data:
            data = input_data["data"]
        elif "from_DataAgent" in input_data:
            data = input_data["from_DataAgent"].get("data")
        elif "factor_data" in input_data:
            data = input_data["factor_data"]
        elif "from_FactorAgent" in input_data:
            data = input_data["from_FactorAgent"].get("factor_data")
        else:
            data = None

        # 获取策略
        if "strategy" in input_data:
            strategy = input_data["strategy"]
        elif "from_StrategyAgent" in input_data:
            strategy = input_data["from_StrategyAgent"].get("strategy")
        else:
            strategy = None

        return data, strategy

    def _run_backtrader_backtest(
        self,
        data: pd.DataFrame,
        strategy: Dict,
        config: BacktestConfig
    ) -> BacktestResult:
        """使用 Backtrader 运行回测"""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(config.initial_capital)
        cerebro.broker.setcommission(commission=config.commission)
        cerebro.broker.set_slippage_fixed(config.slippage)

        # 加载数据
        data_feed = self._prepare_backtrader_data(data)
        cerebro.adddata(data_feed)

        # 添加策略
        cerebro.addstrategy(self._create_backtrader_strategy(strategy))

        # 运行回测
        starting_value = cerebro.broker.getvalue()
        cerebro.run()
        ending_value = cerebro.broker.getvalue()

        # 计算结果
        result = BacktestResult(
            total_return=(ending_value - starting_value) / starting_value,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            trades=0
        )

        return result

    def _run_vectorbt_backtest(
        self,
        data: pd.DataFrame,
        strategy: Dict,
        config: BacktestConfig
    ) -> BacktestResult:
        """使用 VectorBT 运行回测"""
        close = data["close"]

        # 简单策略实现
        entries = data["close"] > data["ma5"]
        exits = data["close"] < data["ma5"]

        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            direction='longonly',
            init_cash=config.initial_capital,
            fees=config.commission,
            slippage=config.slippage
        )

        # 计算回测结果
        total_return = portfolio.total_return()
        annual_return = portfolio.annual_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        win_rate = portfolio.win_rate()
        profit_factor = portfolio.profit_factor()

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=int(portfolio.trades().count())
        )

    def _prepare_backtrader_data(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """准备 Backtrader 数据格式"""
        # 确保日期列是 datetime 类型
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        elif 'trade_date' in data.columns:
            data['datetime'] = pd.to_datetime(data['trade_date'])
        else:
            # 如果没有日期列，使用索引
            data = data.copy()
            data['datetime'] = data.index

        data = data.sort_values('datetime')
        data.set_index('datetime', inplace=True)

        return bt.feeds.PandasData(dataname=data)

    def _create_backtrader_strategy(self, strategy: Dict) -> type:
        """创建 Backtrader 策略类"""
        class CustomStrategy(bt.Strategy):
            params = dict(
                fast_period=5,
                slow_period=20,
                stop_loss=0.05
            )

            def log(self, txt, dt=None):
                ''' Logging function for this strategy'''
                dt = dt or self.datas[0].datetime.date(0)
                self.log_message(f'{dt.isoformat()}, {txt}')

            def __init__(self):
                self.dataclose = self.datas[0].close

                # 初始化订单变量
                self.order = None
                self.buyprice = None
                self.buycomm = None

                # 添加指标
                self.sma5 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.fast_period)
                self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.slow_period)

            def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    return

                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                        self.buyprice = order.executed.price
                        self.buycomm = order.executed.comm
                    elif order.issell():
                        self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

                elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                    self.log('Order Canceled/Margin/Rejected')

                self.order = None

            def notify_trade(self, trade):
                if not trade.isclosed:
                    return

                self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')

            def next(self):
                if self.order:
                    return

                # 入场逻辑
                if self.position.size == 0:
                    if self.sma5[0] > self.sma20[0]:
                        self.log('BUY CREATE, Price: {:.2f}'.format(self.dataclose[0]))
                        self.order = self.buy()

                # 出场逻辑
                elif self.position.size > 0:
                    if self.sma5[0] < self.sma20[0]:
                        self.log('SELL CREATE, Price: {:.2f}'.format(self.dataclose[0]))
                        self.order = self.sell()

        return CustomStrategy

    def _save_backtest_result(self, result: BacktestResult, config: Dict[str, Any]) -> str:
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_result_{timestamp}.csv"
        save_path = str(Path(self.backtest_dir) / filename)

        # 保存结果数据
        result_dict = {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "profit_factor": result.profit_factor,
            "trades": result.trades,
            "holding_period": result.holding_period
        }

        save_data(pd.Series(result_dict), save_path)

        return save_path

    def _save_detailed_result(self, result: BacktestResult, save_path: str) -> None:
        """保存详细结果（如权益曲线）"""
        if not result.equity_curve.empty:
            equity_path = save_path.replace('.csv', '_equity.csv')
            save_data(result.equity_curve, equity_path)

        if not result.positions.empty:
            positions_path = save_path.replace('.csv', '_positions.csv')
            save_data(result.positions, positions_path)

    def get_result_summary(self, result: Optional[BacktestResult] = None) -> Dict[str, str]:
        """获取结果摘要"""
        if result is None:
            result = self.current_result

        if result is None:
            return {}

        return {
            "Total Return": f"{result.total_return:.2%}",
            "Annual Return": f"{result.annual_return:.2%}",
            "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
            "Max Drawdown": f"{result.max_drawdown:.2%}",
            "Win Rate": f"{result.win_rate:.2%}",
            "Trades": str(result.trades),
            "Profit Factor": f"{result.profit_factor:.2f}"
        }

    def analyze_risk_reward(self, result: Optional[BacktestResult] = None) -> Dict:
        """分析风险收益比"""
        if result is None:
            result = self.current_result

        if result is None:
            return {}

        return {
            "risk_adjusted_return": result.total_return / abs(result.max_drawdown) if result.max_drawdown else 0,
            "return_per_trade": result.total_return / result.trades if result.trades else 0,
            "win_loss_ratio": result.avg_win / abs(result.avg_loss) if result.avg_loss else 0
        }

    def cleanup(self) -> TaskResult:
        """清理资源"""
        self.logger.info("BacktestAgent cleaned up")
        return TaskResult(success=True)
