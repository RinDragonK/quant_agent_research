"""
Simple system test script (English)
Verify core functionality without encoding issues
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


def generate_test_data(start_date: str = None, end_date: str = None):
    """Generate test data"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

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


def test_core_imports():
    """Test core imports"""
    print("\n" + "=" * 50)
    print("Test 1: Core imports")
    print("=" * 50)

    try:
        from core import (
            BaseAgent, AgentStatus, AgentType, Message, TaskResult,
            Pipeline, PipelineStep, PipelineStatus, PipelineResult,
            setup_logging, create_directories, load_config, save_config,
            load_data, save_data, load_json, save_json,
            save_pickle, load_pickle, calculate_metrics,
            calculate_max_drawdown, format_number, format_time,
            timeit, validate_date, get_date_range, generate_filename,
            ensure_directory_exists, safe_divide, standardize, normalize
        )
        print("[OK] Core module imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Core imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_imports():
    """Test agent imports"""
    print("\n" + "=" * 50)
    print("Test 2: Agent imports")
    print("=" * 50)

    try:
        from agents import (
            DataAgent, FactorAgent, StrategyAgent,
            BacktestAgent, RiskAgent, ReportAgent
        )
        print("[OK] Agent module imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Agent imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_agent():
    """Test Data Agent"""
    print("\n" + "=" * 50)
    print("Test 3: Data Agent")
    print("=" * 50)

    try:
        from agents import DataAgent

        agent = DataAgent()
        init_result = agent.initialize()
        print(f"[OK] DataAgent initialized: {init_result.success}")

        sources = agent.list_available_sources()
        print(f"[OK] Available sources: {sources}")

        test_data = generate_test_data()
        print(f"[OK] Generated test data: {len(test_data)} records")

        quality_check = agent._check_data_quality(test_data)
        print(f"[OK] Data completeness: {quality_check['completeness']:.2%}")

        agent.cleanup()
        print("[OK] DataAgent test passed!")
        return True

    except Exception as e:
        print(f"[FAIL] DataAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factor_agent():
    """Test Factor Agent"""
    print("\n" + "=" * 50)
    print("Test 4: Factor Agent")
    print("=" * 50)

    try:
        from agents import FactorAgent

        agent = FactorAgent()
        init_result = agent.initialize()
        print(f"[OK] FactorAgent initialized: {init_result.success}")

        factors = agent.list_available_factors()
        print(f"[OK] Available factors: {len(factors)}")
        for i, factor in enumerate(factors[:5], 1):
            print(f"  {i}. {factor['name']}: {factor['description']}")
        if len(factors) > 5:
            print(f"  ... and {len(factors) - 5} more factors")

        test_data = generate_test_data()
        config = {"factors": ["ma5", "ma10", "rsi"]}
        data_with_factors = agent._calculate_all_factors(test_data, config)

        factor_columns = [col for col in data_with_factors.columns
                         if col in ['ma5', 'ma10', 'rsi']]
        print(f"[OK] Calculated factors: {factor_columns}")

        if 'ma5' in data_with_factors.columns:
            non_null = data_with_factors['ma5'].notnull().sum()
            print(f"[OK] MA5 valid data: {non_null}/{len(data_with_factors)}")

        agent.cleanup()
        print("[OK] FactorAgent test passed!")
        return True

    except Exception as e:
        print(f"[FAIL] FactorAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_agent():
    """Test Strategy Agent"""
    print("\n" + "=" * 50)
    print("Test 5: Strategy Agent")
    print("=" * 50)

    try:
        from agents import StrategyAgent

        agent = StrategyAgent()
        init_result = agent.initialize()
        print(f"[OK] StrategyAgent initialized: {init_result.success}")

        strategies = agent.list_available_strategies()
        print(f"[OK] Available strategies: {len(strategies)}")
        for strategy in strategies:
            print(f"  - {strategy['name']}: {strategy['description']}")

        test_data = generate_test_data()
        config = {"strategy_type": "trend_following"}
        strategy = agent._build_trend_strategy(test_data, config)

        print(f"[OK] Strategy name: {strategy.name}")
        print(f"[OK] Strategy type: {strategy.type}")
        print(f"[OK] Strategy parameters: {len(strategy.parameters)}")

        agent.cleanup()
        print("[OK] StrategyAgent test passed!")
        return True

    except Exception as e:
        print(f"[FAIL] StrategyAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions"""
    print("\n" + "=" * 50)
    print("Test 6: Utility functions")
    print("=" * 50)

    try:
        from core import (
            format_number, format_time, safe_divide,
            standardize, normalize, calculate_metrics
        )

        print(f"[OK] format_number(0.156): {format_number(0.156)}")
        print(f"[OK] format_number(1.234): {format_number(1.234)}")
        print(f"[OK] format_time(65): {format_time(65)}")
        print(f"[OK] format_time(3665): {format_time(3665)}")
        print(f"[OK] safe_divide(10, 2): {safe_divide(10, 2)}")
        print(f"[OK] safe_divide(10, 0): {safe_divide(10, 0)}")

        data = pd.Series([1, 2, 3, 4, 5])
        std_data = standardize(data)
        norm_data = normalize(data)
        print(f"[OK] standardize: mean={std_data.mean():.2f}, std={std_data.std():.2f}")
        print(f"[OK] normalize: min={norm_data.min():.2f}, max={norm_data.max():.2f}")

        df = pd.DataFrame({
            'return': [0.01, -0.02, 0.03, -0.01, 0.02],
            'close': [100, 98, 101, 100, 102]
        })
        metrics = calculate_metrics(df)
        print(f"[OK] calculate_metrics: {list(metrics.keys())}")

        print("[OK] Utility functions test passed!")
        return True

    except Exception as e:
        print(f"[FAIL] Utility functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("""
============================================================
  Multi-Agent Quantitative Research System - Test Suite
============================================================
    """)

    results = {
        "Core imports": test_core_imports(),
        "Agent imports": test_agent_imports(),
        "Data Agent": test_data_agent(),
        "Factor Agent": test_factor_agent(),
        "Strategy Agent": test_strategy_agent(),
        "Utilities": test_utils()
    }

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {name}: {status}")

    passed = sum(results.values())
    total = len(results)

    print("\n" + "=" * 50)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("\nAll tests passed! System is working correctly.")
        return 0
    else:
        print(f"\n{total - passed} tests failed. Please check logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
