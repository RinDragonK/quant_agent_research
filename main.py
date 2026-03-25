"""
Multi-Agent 量化投研自动化系统
主程序入口
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

from core import (
    setup_logging,
    create_directories,
    load_config,
    save_config,
    Pipeline,
    format_number,
    format_time,
    timeit
)
from agents import (
    DataAgent,
    FactorAgent,
    StrategyAgent,
    BacktestAgent,
    RiskAgent,
    ReportAgent
)


class QuantResearchSystem:
    """
    量化投研系统主类
    负责协调所有智能体的工作流程
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path) if Path(config_path).exists() else {}

        # 设置日志
        log_file = self.config.get("log_file", "logs/quant_agent.log")
        Path(log_file).parent.mkdir(exist_ok=True)
        setup_logging(log_file)
        self.logger = logging.getLogger("QuantResearchSystem")

        # 创建目录
        create_directories()

        # 初始化管道
        self.pipeline = Pipeline("QuantResearchPipeline")

        # 初始化智能体
        self.agents = {}
        self._initialize_agents()

        # 添加步骤到管道
        self._build_pipeline()

    def _initialize_agents(self):
        """初始化所有智能体"""
        agent_configs = self.config.get("agents", {})

        # 数据智能体
        self.agents["data"] = DataAgent(
            name="DataAgent",
            config=agent_configs.get("data", {})
        )

        # 因子智能体
        self.agents["factor"] = FactorAgent(
            name="FactorAgent",
            config=agent_configs.get("factor", {})
        )

        # 策略智能体
        self.agents["strategy"] = StrategyAgent(
            name="StrategyAgent",
            config=agent_configs.get("strategy", {})
        )

        # 回测智能体
        self.agents["backtest"] = BacktestAgent(
            name="BacktestAgent",
            config=agent_configs.get("backtest", {})
        )

        # 风险智能体
        self.agents["risk"] = RiskAgent(
            name="RiskAgent",
            config=agent_configs.get("risk", {})
        )

        # 报告智能体
        self.agents["report"] = ReportAgent(
            name="ReportAgent",
            config=agent_configs.get("report", {})
        )

        self.logger.info("All agents initialized")

    def _build_pipeline(self):
        """构建管道"""
        # 添加数据步骤
        self.pipeline.add_step(
            name="data",
            agent=self.agents["data"],
            depends_on=[]
        )

        # 添加因子步骤
        self.pipeline.add_step(
            name="factor",
            agent=self.agents["factor"],
            depends_on=["data"]
        )

        # 添加策略步骤
        self.pipeline.add_step(
            name="strategy",
            agent=self.agents["strategy"],
            depends_on=["factor"]
        )

        # 添加回测步骤
        self.pipeline.add_step(
            name="backtest",
            agent=self.agents["backtest"],
            depends_on=["strategy", "factor"]
        )

        # 添加风险步骤
        self.pipeline.add_step(
            name="risk",
            agent=self.agents["risk"],
            depends_on=["backtest"]
        )

        # 添加报告步骤
        self.pipeline.add_step(
            name="report",
            agent=self.agents["report"],
            depends_on=["data", "factor", "strategy", "backtest", "risk"]
        )

        self.logger.info("Pipeline built successfully")

    @timeit
    def run_full_pipeline(self, custom_config: dict = None) -> dict:
        """运行完整管道"""
        self.logger.info("=" * 60)
        self.logger.info("Starting full quant research pipeline")
        self.logger.info("=" * 60)

        # 合并配置
        input_data = {**self.config, **(custom_config or {})}

        # 运行管道
        result = self.pipeline.run(initial_input=input_data)

        # 输出结果摘要
        self._print_summary(result)

        # 保存执行摘要
        summary_path = Path("backtests") / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.pipeline.save_summary(str(summary_path))

        return {
            "success": result.success,
            "result": result,
            "summary_path": str(summary_path)
        }

    def _print_summary(self, result):
        """打印执行摘要"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 60)

        for step_name, step_result in result.step_results.items():
            status = "OK" if step_result.success else "ERROR"
            self.logger.info(f"{status} {step_name}: {'Success' if step_result.success else 'Failed'}")

            if step_result.warnings:
                for warning in step_result.warnings:
                    self.logger.warning(f"  - Warning: {warning}")

            if step_result.errors:
                for error in step_result.errors:
                    self.logger.error(f"  - Error: {error}")

        self.logger.info(f"\nTotal execution time: {format_time(result.total_execution_time)}")
        self.logger.info(f"Overall status: {'SUCCESS' if result.success else 'FAILED'}")
        self.logger.info("=" * 60 + "\n")

    def run_single_agent(self, agent_name: str, input_data: dict = None) -> dict:
        """单独运行某个智能体"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        agent = self.agents[agent_name]
        self.logger.info(f"Running single agent: {agent_name}")

        result = agent.run(input_data)

        return {
            "agent": agent_name,
            "success": result.success,
            "result": result
        }

    def list_available_agents(self) -> list:
        """列出可用智能体"""
        return list(self.agents.keys())

    def get_agent_info(self, agent_name: str) -> dict:
        """Get agent information"""
        if agent_name == "data":
            return {
                "name": "Data Agent",
                "sources": self.agents["data"].list_available_sources(),
                "description": "Responsible for data fetching, cleaning, and storage"
            }
        elif agent_name == "factor":
            return {
                "name": "Factor Agent",
                "factors": self.agents["factor"].list_available_factors(),
                "description": "Responsible for technical indicators and factor calculation"
            }
        elif agent_name == "strategy":
            return {
                "name": "Strategy Agent",
                "strategies": self.agents["strategy"].list_available_strategies(),
                "description": "Responsible for strategy construction and optimization"
            }
        else:
            return {
                "name": agent_name,
                "description": "Quantitative Research Agent"
            }

    def save_config(self):
        """保存配置"""
        save_config(self.config, self.config_path)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Quantitative Research System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 运行完整管道
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")
    run_parser.add_argument("--assets", "-a", nargs="+", help="Assets to analyze")
    run_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    run_parser.add_argument("--strategy", help="Strategy type")

    # 单独运行智能体
    agent_parser = subparsers.add_parser("agent", help="Run single agent")
    agent_parser.add_argument("agent_name", help="Agent name")
    agent_parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")

    # 列出可用智能体
    list_parser = subparsers.add_parser("list", help="List available agents")
    list_parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")

    # 显示帮助
    subparsers.add_parser("help", help="Show help")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    if args.command == "help":
        print("""
Multi-Agent Quantitative Research System

Available commands:
  run      - Run full quant research pipeline
  agent    - Run single agent
  list     - List available agents
  help     - Show help

Examples:
  # Run full pipeline
  python main.py run

  # Use custom config
  python main.py run --config my_config.yaml

  # Specify assets and date range
  python main.py run --assets 600036.SH 000001.SZ --start-date 2024-01-01 --end-date 2024-12-31

  # Run single agent
  python main.py agent data

  # List agents
  python main.py list
        """)
        return

    # 初始化系统
    config_path = getattr(args, "config", "config/config.yaml")
    system = QuantResearchSystem(config_path=config_path)

    if args.command == "run":
        # Build custom config
        custom_config = {}
        if args.assets:
            custom_config["assets"] = args.assets
        if args.start_date:
            custom_config["start_date"] = args.start_date
        if args.end_date:
            custom_config["end_date"] = args.end_date
        if args.strategy:
            custom_config["strategy_type"] = args.strategy

        # Run full pipeline
        result = system.run_full_pipeline(custom_config)

        if result["success"]:
            print("\n[OK] Quant research pipeline executed successfully!")
            print(f"[INFO] Summary saved to: {result['summary_path']}")
        else:
            print("\n[ERROR] Quant research pipeline failed, check logs")
            sys.exit(1)

    elif args.command == "agent":
        # Run single agent
        result = system.run_single_agent(args.agent_name)
        status = "succeeded" if result["success"] else "failed"
        print(f"\nAgent {result['agent']} {status}")

    elif args.command == "list":
        # List available agents
        agents = system.list_available_agents()
        print("\nAvailable agents:")
        print("-" * 40)
        for agent in agents:
            info = system.get_agent_info(agent)
            print(f"  - {agent} - {info['description']}")
        print()


if __name__ == "__main__":
    main()
