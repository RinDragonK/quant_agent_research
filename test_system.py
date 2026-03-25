"""
系统测试脚本
验证 Multi-Agent 量化投研系统的核心功能
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from core import setup_logging
from agents import DataAgent, FactorAgent, StrategyAgent


def generate_test_data(start_date: str = None, end_date: str = None):
    """生成测试数据"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

    # 生成模拟价格数据
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * (1 + returns).cumprod()

    high = prices * (1 + np.random.uniform(0, 0.02, n))
    low = prices * (1 - np.random.uniform(0, 0.02, n))
    close = prices
    open_ = close * (1 + np.random.uniform(-0.01, 0.01, n))
    volume = np.random.randint(100000, 1000000, n)

    data = pd.DataFrame({
        'datetime': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'symbol': 'TEST'
    })

    return data


def test_data_agent():
    """测试数据智能体"""
    print("\n" + "=" * 50)
    print("测试 1: 数据智能体")
    print("=" * 50)

    try:
        agent = DataAgent()

        # 测试初始化
        init_result = agent.initialize()
        print(f"✓ 初始化: {'成功' if init_result.success else '失败'}")

        # 测试可用数据源
        sources = agent.list_available_sources()
        print(f"✓ 可用数据源: {sources}")

        # 使用模拟数据测试
        test_data = generate_test_data()
        print(f"✓ 生成测试数据: {len(test_data)} 条记录")

        # 保存测试数据
        agent._save_data(test_data, {"source": "test", "assets": ["TEST"]})
        print(f"✓ 保存测试数据")

        # 检查数据质量
        quality_check = agent._check_data_quality(test_data)
        print(f"✓ 数据完整性: {quality_check['completeness']:.2%}")

        # 清理
        agent.cleanup()
        print("✓ 数据智能体测试通过!")
        return True

    except Exception as e:
        print(f"✗ 数据智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factor_agent():
    """测试因子智能体"""
    print("\n" + "=" * 50)
    print("测试 2: 因子智能体")
    print("=" * 50)

    try:
        agent = FactorAgent()

        # 测试初始化
        init_result = agent.initialize()
        print(f"✓ 初始化: {'成功' if init_result.success else '失败'}")

        # 列出可用因子
        factors = agent.list_available_factors()
        print(f"✓ 可用因子数量: {len(factors)}")
        for i, factor in enumerate(factors[:5], 1):
            print(f"  {i}. {factor['name']}: {factor['description']}")
        if len(factors) > 5:
            print(f"  ... 还有 {len(factors) - 5} 个因子")

        # 生成测试数据
        test_data = generate_test_data()

        # 测试因子计算
        config = {"factors": ["ma5", "ma10", "rsi", "macd", "bollinger"]}
        data_with_factors = agent._calculate_all_factors(test_data, config)

        # 检查因子是否计算成功
        factor_columns = [col for col in data_with_factors.columns
                         if col in ['ma5', 'ma10', 'rsi', 'macd', 'bollinger_mid']]
        print(f"✓ 计算的因子: {factor_columns}")

        # 验证因子值
        if 'ma5' in data_with_factors.columns:
            non_null = data_with_factors['ma5'].notnull().sum()
            print(f"✓ MA5 有效数据: {non_null}/{len(data_with_factors)}")

        # 清理
        agent.cleanup()
        print("✓ 因子智能体测试通过!")
        return True

    except Exception as e:
        print(f"✗ 因子智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_agent():
    """测试策略智能体"""
    print("\n" + "=" * 50)
    print("测试 3: 策略智能体")
    print("=" * 50)

    try:
        agent = StrategyAgent()

        # 测试初始化
        init_result = agent.initialize()
        print(f"✓ 初始化: {'成功' if init_result.success else '失败'}")

        # 列出可用策略
        strategies = agent.list_available_strategies()
        print(f"✓ 可用策略数量: {len(strategies)}")
        for strategy in strategies:
            print(f"  - {strategy['name']}: {strategy['description']}")

        # 生成测试数据
        test_data = generate_test_data()

        # 测试策略构建
        config = {"strategy_type": "trend_following"}
        strategy = agent._build_trend_strategy(test_data, config)

        print(f"✓ 策略名称: {strategy.name}")
        print(f"✓ 策略类型: {strategy.type}")
        print(f"✓ 策略参数数量: {len(strategy.parameters)}")
        for param in strategy.parameters:
            print(f"  - {param.name}: {param.value}")

        print(f"✓ 策略规则数量: {len(strategy.rules)}")
        for rule in strategy.rules:
            print(f"  - {rule.name}: {rule.condition}")

        # 清理
        agent.cleanup()
        print("✓ 策略智能体测试通过!")
        return True

    except Exception as e:
        print(f"✗ 策略智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_modules():
    """测试核心模块"""
    print("\n" + "=" * 50)
    print("测试 4: 核心模块")
    print("=" * 50)

    try:
        # 测试 utils 模块
        from core import (
            format_number,
            format_time,
            safe_divide,
            standardize,
            normalize,
            calculate_metrics
        )

        # 测试格式化函数
        print(f"✓ format_number: {format_number(0.156)}")
        print(f"✓ format_number: {format_number(1.234)}")
        print(f"✓ format_time: {format_time(65)}")
        print(f"✓ format_time: {format_time(3665)}")

        # 测试数学函数
        print(f"✓ safe_divide: {safe_divide(10, 2)}")
        print(f"✓ safe_divide: {safe_divide(10, 0)}")

        # 测试数据处理
        data = pd.Series([1, 2, 3, 4, 5])
        std_data = standardize(data)
        norm_data = normalize(data)
        print(f"✓ standardize: mean={std_data.mean():.2f}, std={std_data.std():.2f}")
        print(f"✓ normalize: min={norm_data.min():.2f}, max={norm_data.max():.2f}")

        # 测试指标计算
        df = pd.DataFrame({
            'return': [0.01, -0.02, 0.03, -0.01, 0.02],
            'close': [100, 98, 101, 100, 102]
        })
        metrics = calculate_metrics(df)
        print(f"✓ calculate_metrics: {list(metrics.keys())}")

        print("✓ 核心模块测试通过!")
        return True

    except Exception as e:
        print(f"✗ 核心模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Multi-Agent 量化投研系统 - 测试套件                           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    setup_logging(log_file="test.log")

    results = {
        "数据智能体": test_data_agent(),
        "因子智能体": test_factor_agent(),
        "策略智能体": test_strategy_agent(),
        "核心模块": test_core_modules()
    }

    # 输出总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    passed = sum(results.values())
    total = len(results)

    print("\n" + "=" * 50)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 50)

    if passed == total:
        print("\n🎉 所有测试通过! 系统运行正常。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查日志。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
