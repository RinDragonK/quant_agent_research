"""
快速入门示例
展示如何使用 Multi-Agent 量化投研自动化系统
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import QuantResearchSystem


def example_full_pipeline():
    """示例 1: 运行完整管道"""
    print("=" * 60)
    print("示例 1: 运行完整量化投研管道")
    print("=" * 60)

    # 初始化系统
    system = QuantResearchSystem()

    # 自定义配置
    custom_config = {
        "assets": ["NVDA", "AAPL"],
        "start_date": "2025-01-01",
        "end_date": "2026-03-19",
        "source": "yfinance"
    }

    # 运行完整管道
    result = system.run_full_pipeline(custom_config)

    if result["success"]:
        print("\n✅ 完整管道运行成功!")
    else:
        print("\n❌ 管道运行失败")

    return result


def example_single_agent():
    """示例 2: 单独运行智能体"""
    print("\n" + "=" * 60)
    print("示例 2: 单独运行智能体")
    print("=" * 60)

    system = QuantResearchSystem()

    # 列出可用智能体
    agents = system.list_available_agents()
    print(f"\n可用智能体: {agents}")

    # 单独运行数据智能体
    print("\n1. 运行数据智能体...")
    data_result = system.run_single_agent("data", {
        "assets": ["600036.SH"],
        "source": "yfinance"
    })
    print(f"   数据智能体: {'成功' if data_result['success'] else '失败'}")

    # 单独运行因子智能体
    if data_result["success"]:
        print("\n2. 运行因子智能体...")
        factor_result = system.run_single_agent("factor", data_result["result"].data)
        print(f"   因子智能体: {'成功' if factor_result['success'] else '失败'}")


def example_strategy_info():
    """示例 3: 查看可用策略和因子"""
    print("\n" + "=" * 60)
    print("示例 3: 查看可用策略和因子")
    print("=" * 60)

    system = QuantResearchSystem()

    # 获取策略信息
    strategy_info = system.get_agent_info("strategy")
    print(f"\n策略: {strategy_info['name']}")
    print(f"描述: {strategy_info['description']}")
    print(f"可用策略:")
    for strategy in strategy_info["strategies"]:
        print(f"  - {strategy['name']}: {strategy['description']}")

    # 获取因子信息
    factor_info = system.get_agent_info("factor")
    print(f"\n因子: {factor_info['name']}")
    print(f"描述: {factor_info['description']}")
    print(f"可用因子 ({len(factor_info['factors'])}个):")
    for i, factor in enumerate(factor_info['factors'][:10], 1):
        print(f"  {i}. {factor['name']}: {factor['description']}")
    if len(factor_info['factors']) > 10:
        print(f"  ... 还有 {len(factor_info['factors']) - 10} 个因子")


def example_custom_workflow():
    """示例 4: 自定义工作流"""
    print("\n" + "=" * 60)
    print("示例 4: 自定义工作流")
    print("=" * 60)

    system = QuantResearchSystem()

    # 自定义工作流配置
    custom_config = {
        "assets": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2025-01-01",
        "end_date": "2026-03-19",
        "strategy_type": "multi_factor",
        "optimization": True,
        "initial_capital": 2000000
    }

    print(f"\n自定义配置:")
    print(f"  - 股票池: {custom_config['assets']}")
    print(f"  - 日期范围: {custom_config['start_date']} ~ {custom_config['end_date']}")
    print(f"  - 策略类型: {custom_config['strategy_type']}")
    print(f"  - 初始资金: {custom_config['initial_capital']:,.2f}")
    print(f"  - 参数优化: {'开启' if custom_config['optimization'] else '关闭'}")

    # 运行自定义工作流
    print("\n开始运行自定义工作流...")
    result = system.run_full_pipeline(custom_config)

    return result


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Multi-Agent 量化投研自动化系统 - 快速入门                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 运行示例
    try:
        # 示例 1: 完整管道
        example_full_pipeline()

        # 示例 2: 单独运行智能体
        example_single_agent()

        # 示例 3: 查看信息
        example_strategy_info()

        # 示例 4: 自定义工作流
        example_custom_workflow()

        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
