"""
报告生成智能体模块
负责报告生成和可视化
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import logging

from core.agent_base import BaseAgent, AgentType, TaskResult
from core.utils import ensure_directory_exists


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: Any
    section_type: str = "text"
    priority: int = 5
    requires_data: bool = False


@dataclass
class ReportConfig:
    """报告配置"""
    title: str = "Quantitative Research Report"
    author: str = "Quant Agent System"
    date: str = datetime.now().strftime("%Y-%m-%d")
    format: str = "html"
    sections: List[str] = field(default_factory=lambda: [
        "executive_summary", "data_overview", "factor_analysis",
        "strategy_details", "backtest_results", "risk_analysis", "conclusions"
    ])
    include_charts: bool = True
    include_tables: bool = True
    include_appendix: bool = False


class ReportAgent(BaseAgent):
    """
    报告智能体
    负责生成量化投研报告
    """

    def __init__(
        self,
        name: str = "ReportAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        super().__init__(name, AgentType.REPORT, config)

        self.report_dir = config.get("report_dir", "reports")
        ensure_directory_exists(self.report_dir)

        # 默认配置
        self.default_config = {
            "template_dir": "templates",
            "output_formats": ["html", "pdf"],
            "charts": True,
            "tables": True,
            "interactive": False
        }
        self.config = {**self.default_config, **config}

        # 设置图表风格
        self._setup_plot_style()

        # 报告模板加载器
        self.template_env = self._setup_template_env()

        self.generated_reports: List[str] = []

    def _setup_plot_style(self) -> None:
        """设置图表风格"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    def _setup_template_env(self):
        """设置模板环境"""
        template_dir = Path(self.config.get("template_dir", "templates"))
        if not template_dir.exists():
            template_dir = Path(__file__).parent / "templates"
            if not template_dir.exists():
                template_dir.mkdir(exist_ok=True)
                self._create_default_templates(template_dir)

        return Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

    def _create_default_templates(self, template_dir: Path):
        """创建默认报告模板"""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        .section-title { color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        .chart-container { margin: 20px 0; text-align: center; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .highlight { background-color: #ffffcc; font-weight: bold; }
        .risk-alert { background-color: #ffcccc; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Author: {{ author }}</p>
        <p>Date: {{ date }}</p>
    </div>

    {% for section in sections %}
    <div class="section">
        <h2 class="section-title">{{ section.title }}</h2>
        {{ section.content }}
    </div>
    {% endfor %}
</body>
</html>
"""

        with open(template_dir / "report_template.html", 'w', encoding='utf-8') as f:
            f.write(html_template)

    def initialize(self) -> TaskResult:
        """初始化报告智能体"""
        self.logger.info("ReportAgent initialized")
        return TaskResult(success=True)

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行报告生成任务"""
        config = {**self.config, **(input_data or {})}

        try:
            # 获取输入数据
            agent_results = self._collect_agent_results(input_data)

            # 创建报告配置
            report_config = ReportConfig(
                title=config.get("title", "Quantitative Research Report"),
                author=config.get("author", "Quant Agent System"),
                sections=config.get("sections", report_config.sections)
            )

            # 生成报告内容
            report_content = self._generate_report_content(report_config, agent_results)

            # 保存报告
            report_path = self._save_report(report_content, report_config)

            self.generated_reports.append(report_path)

            return TaskResult(
                success=True,
                data={
                    "report_path": report_path,
                    "config": report_config,
                    "sections": [s.title for s in report_content]
                }
            )

        except Exception as e:
            error_msg = f"ReportAgent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return TaskResult(
                success=False,
                errors=[error_msg]
            )

    def _collect_agent_results(self, input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """收集各个智能体的结果"""
        results = {}

        if input_data:
            for key, value in input_data.items():
                if key.startswith("from_") and isinstance(value, dict):
                    agent_name = key.split("_", 1)[1]
                    results[agent_name.lower()] = value

        return results

    def _generate_report_content(
        self,
        config: ReportConfig,
        agent_results: Dict[str, Any]
    ) -> List[ReportSection]:
        """生成报告内容"""
        sections = []

        # 添加执行摘要
        if "executive_summary" in config.sections:
            sections.append(self._create_executive_summary(agent_results))

        # 添加数据概述
        if "data_overview" in config.sections:
            sections.append(self._create_data_overview(agent_results))

        # 添加因子分析
        if "factor_analysis" in config.sections:
            sections.append(self._create_factor_analysis(agent_results))

        # 添加策略详情
        if "strategy_details" in config.sections:
            sections.append(self._create_strategy_details(agent_results))

        # 添加回测结果
        if "backtest_results" in config.sections:
            sections.append(self._create_backtest_results(agent_results))

        # 添加风险分析
        if "risk_analysis" in config.sections:
            sections.append(self._create_risk_analysis(agent_results))

        # 添加结论
        if "conclusions" in config.sections:
            sections.append(self._create_conclusions(agent_results))

        return sections

    def _create_executive_summary(self, results: Dict[str, Any]) -> ReportSection:
        """创建执行摘要"""
        summary = ["<h3>执行摘要</h3>"]

        if "backtestagent" in results:
            bt_result = results["backtestagent"]
            summary.append("<p>本报告包含量化策略的完整回测结果，包括：</p>")
            if "backtest_result" in bt_result:
                result = bt_result["backtest_result"]
                summary.append(f"<ul>")
                summary.append(f"<li>总收益率: {result.total_return:.2%}</li>")
                summary.append(f"<li>年化收益率: {result.annual_return:.2%}</li>")
                summary.append(f"<li>夏普比率: {result.sharpe_ratio:.2f}</li>")
                summary.append(f"<li>最大回撤: {result.max_drawdown:.2%}</li>")
                summary.append(f"<li>胜率: {result.win_rate:.2%}</li>")
                summary.append(f"<li>交易次数: {result.trades}</li>")
                summary.append(f"</ul>")

        if "riskagent" in results and "risk_alerts" in results["riskagent"]:
            alerts = results["riskagent"]["risk_alerts"]
            if alerts:
                summary.append("<p>风险警报:</p>")
                for alert in alerts:
                    summary.append(f"<div class='risk-alert'>{alert.message}</div>")

        return ReportSection(
            title="执行摘要",
            content="".join(summary),
            section_type="html",
            priority=1
        )

    def _create_data_overview(self, results: Dict[str, Any]) -> ReportSection:
        """创建数据概述"""
        content = ["<h3>数据概述</h3>"]

        if "dataagent" in results:
            content.append("<p>数据来源: Yahoo Finance</p>")
            content.append("<p>数据周期: 2025-03-19 至 2026-03-19</p>")
            content.append("<p>包含股票: 600036.SH (招商银行), 000001.SZ (平安银行)</p>")

        return ReportSection(
            title="数据概述",
            content="".join(content),
            section_type="html",
            priority=2
        )

    def _create_factor_analysis(self, results: Dict[str, Any]) -> ReportSection:
        """创建因子分析"""
        content = ["<h3>因子分析</h3>"]

        if "factoragent" in results:
            content.append("<p>计算的技术指标因子:</p>")
            content.append("<ul>")
            content.append("<li>移动平均线: MA5, MA10, MA20, MA60</li>")
            content.append("<li>动量指标: RSI(14), Momentum(10)</li>")
            content.append("<li>波动率指标: Bollinger Bands, ATR</li>")
            content.append("<li>成交量指标: OBV, Volume MA</li>")
            content.append("<li>其他指标: MACD, KDJ</li>")
            content.append("</ul>")

        return ReportSection(
            title="因子分析",
            content="".join(content),
            section_type="html",
            priority=3
        )

    def _create_strategy_details(self, results: Dict[str, Any]) -> ReportSection:
        """创建策略详情"""
        content = ["<h3>策略详情</h3>"]

        if "strategyagent" in results:
            strategy = results["strategyagent"].get("strategy")
            if strategy:
                content.append(f"<p>策略名称: {strategy.get('name', 'MA Crossover')}</p>")
                content.append(f"<p>策略类型: Trend Following</p>")
                content.append(f"<p>参数优化: {results['strategyagent'].get('optimization', False)}</p>")
                content.append("<h4>入场条件</h4>")
                content.append("<ul>")
                content.append("<li>MA5 > MA20</li>")
                content.append("<li>价格 > MA20</li>")
                content.append("</ul>")
                content.append("<h4>出场条件</h4>")
                content.append("<ul>")
                content.append("<li>MA5 < MA20</li>")
                content.append("<li>价格 < MA20</li>")
                content.append("</ul>")

        return ReportSection(
            title="策略详情",
            content="".join(content),
            section_type="html",
            priority=4
        )

    def _create_backtest_results(self, results: Dict[str, Any]) -> ReportSection:
        """创建回测结果"""
        content = ["<h3>回测结果</h3>"]

        if "backtestagent" in results:
            bt_result = results["backtestagent"]
            if "backtest_result" in bt_result:
                result = bt_result["backtest_result"]
                content.append("<h4>回测统计</h4>")
                content.append("<table>")
                content.append("<tr><th>指标</th><th>数值</th></tr>")
                content.append(f"<tr><td>总收益率</td><td>{result.total_return:.2%}</td></tr>")
                content.append(f"<tr><td>年化收益率</td><td>{result.annual_return:.2%}</td></tr>")
                content.append(f"<tr><td>夏普比率</td><td>{result.sharpe_ratio:.2f}</td></tr>")
                content.append(f"<tr><td>最大回撤</td><td>{result.max_drawdown:.2%}</td></tr>")
                content.append(f"<tr><td>胜率</td><td>{result.win_rate:.2%}</td></tr>")
                content.append(f"<tr><td>盈亏比</td><td>{result.profit_factor:.2f}</td></tr>")
                content.append(f"<tr><td>交易次数</td><td>{result.trades}</td></tr>")
                content.append("</table>")

        return ReportSection(
            title="回测结果",
            content="".join(content),
            section_type="html",
            priority=4
        )

    def _create_risk_analysis(self, results: Dict[str, Any]) -> ReportSection:
        """创建风险分析"""
        content = ["<h3>风险分析</h3>"]

        if "riskagent" in results:
            risk_result = results["riskagent"]
            if "risk_metrics" in risk_result:
                metrics = risk_result["risk_metrics"]
                content.append("<h4>风险指标</h4>")
                content.append("<table>")
                content.append("<tr><th>指标</th><th>数值</th></tr>")
                content.append(f"<tr><td>VaR(95%)</td><td>{metrics.var_95:.2%}</td></tr>")
                content.append(f"<tr><td>CVaR(95%)</td><td>{metrics.cvar_95:.2%}</td></tr>")
                content.append(f"<tr><td>年化波动率</td><td>{metrics.annual_volatility:.2%}</td></tr>")
                content.append(f"<tr><td>最大回撤</td><td>{metrics.max_drawdown:.2%}</td></tr>")
                content.append(f"<tr><td>偏度</td><td>{metrics.skewness:.2f}</td></tr>")
                content.append(f"<tr><td>峰度</td><td>{metrics.kurtosis:.2f}</td></tr>")
                content.append("</table>")

        return ReportSection(
            title="风险分析",
            content="".join(content),
            section_type="html",
            priority=5
        )

    def _create_conclusions(self, results: Dict[str, Any]) -> ReportSection:
        """创建结论"""
        content = ["<h3>结论</h3>"]
        content.append("<p>本策略表现出一定的盈利能力，但仍需注意以下几点：</p>")
        content.append("<ul>")
        content.append("<li>策略在单边上涨行情中表现较好</li>")
        content.append("<li>震荡市场中可能会产生较多的无效信号</li>")
        content.append("<li>建议结合其他指标进行优化</li>")
        content.append("<li>风险控制措施需要进一步完善</li>")
        content.append("</ul>")

        return ReportSection(
            title="结论",
            content="".join(content),
            section_type="html",
            priority=6
        )

    def _save_report(
        self,
        sections: List[ReportSection],
        config: ReportConfig
    ) -> str:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.html"
        save_path = str(Path(self.report_dir) / filename)

        # 渲染模板
        template = self.template_env.get_template("report_template.html")
        html_content = template.render(
            title=config.title,
            author=config.author,
            date=config.date,
            sections=sections
        )

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return save_path

    def list_generated_reports(self) -> List[str]:
        """列出已生成的报告"""
        return self.generated_reports

    def _generate_chart(self, data: pd.DataFrame, title: str, chart_type: str = "line"):
        """生成图表（简化实现）"""
        if not self.config.get("charts", True):
            return ""

        try:
            fig, ax = plt.subplots()

            if chart_type == "line":
                sns.lineplot(data=data, ax=ax)
            elif chart_type == "scatter":
                sns.scatterplot(data=data, ax=ax)
            elif chart_type == "hist":
                sns.histplot(data=data, ax=ax)
            elif chart_type == "bar":
                sns.barplot(data=data, ax=ax)
            else:
                sns.lineplot(data=data, ax=ax)

            ax.set_title(title)
            ax.set_xlabel('时间')
            ax.set_ylabel('数值')
            ax.legend(loc='best')

            # 保存图表
            chart_dir = Path(self.report_dir) / "charts"
            chart_dir.mkdir(exist_ok=True)
            chart_filename = f"{title.lower().replace(' ', '_')}.png"
            chart_path = str(chart_dir / chart_filename)
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            return f"<img src='charts/{chart_filename}' alt='{title}'>"

        except Exception as e:
            self.logger.error(f"Chart generation failed: {str(e)}")
            return ""

    def cleanup(self) -> TaskResult:
        """清理资源"""
        plt.close('all')
        self.logger.info("ReportAgent cleaned up")
        return TaskResult(success=True)
