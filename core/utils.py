"""
通用工具函数模块
包含各种辅助函数
"""
from typing import Dict, Any, List, Optional, Union
import os
import sys
import json
import yaml
import pickle
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


def setup_logging(log_file: str = "quant_agent.log", level: int = logging.INFO) -> None:
    """设置日志系统"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 文件日志
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 配置根日志记录器
    logging.root.setLevel(logging.DEBUG)
    logging.root.handlers.clear()
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)

    # 禁用一些库的冗余日志
    for logger_name in ['urllib3', 'requests', 'yahoo_finance']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def create_directories(base_dir: str = "quant_agent") -> None:
    """创建必要的目录"""
    directories = [
        Path(base_dir) / 'data/raw',
        Path(base_dir) / 'data/processed',
        Path(base_dir) / 'data/factors',
        Path(base_dir) / 'strategies',
        Path(base_dir) / 'backtests',
        Path(base_dir) / 'reports'
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using default config")
        return {}
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """保存配置文件"""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logging.info(f"Config saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save config: {e}")


def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件（支持 CSV, JSON, Excel）"""
    file_path = Path(file_path)

    try:
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise


def save_data(data: Union[pd.DataFrame, pd.Series], file_path: str) -> None:
    """保存数据到文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if file_path.suffix == '.csv':
            data.to_csv(file_path, index=False)
        elif file_path.suffix == '.json':
            data.to_json(file_path, orient='records', force_ascii=False, indent=2)
        elif file_path.suffix == '.parquet':
            data.to_parquet(file_path)
        elif file_path.suffix == '.xlsx':
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logging.debug(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
        raise


def load_json(file_path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """保存到 JSON 文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.debug(f"JSON saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file: {e}")
        raise


def save_pickle(obj: Any, file_path: str) -> None:
    """保存对象到 pickle 文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.debug(f"Object pickled to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save pickle: {e}")
        raise


def load_pickle(file_path: str) -> Any:
    """从 pickle 文件加载对象"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load pickle: {e}")
        raise


def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """计算基本统计指标"""
    metrics = {}

    if 'return' in data.columns:
        returns = data['return']
        metrics['total_return'] = (returns + 1).prod() - 1
        metrics['avg_return'] = returns.mean()
        metrics['std_return'] = returns.std()
        metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['std_return'] if metrics['std_return'] != 0 else 0
        metrics['max_drawdown'] = calculate_max_drawdown(returns)
        metrics['win_rate'] = (returns > 0).mean()

    if 'close' in data.columns:
        metrics['price_std'] = data['close'].std()
        metrics['price_change'] = data['close'].iloc[-1] / data['close'].iloc[0] - 1

    return metrics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    cumulative = (returns + 1).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    return drawdown.min()


def format_number(num: float, precision: int = 2) -> str:
    """格式化数字为字符串（自动处理百分比）"""
    if abs(num) < 0.0001:
        return '0'
    elif abs(num) < 1:
        return f"{num * 100:.{precision}f}%"
    else:
        return f"{num:.{precision}f}"


def format_time(seconds: float) -> str:
    """格式化时间（秒转时分秒）"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"


def timeit(func):
    """装饰器：测量函数执行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logging.debug(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper


def validate_date(date_str: str) -> bool:
    """验证日期格式"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """获取日期范围列表"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=1)

    dates = []
    while start <= end:
        dates.append(start.strftime('%Y-%m-%d'))
        start += delta

    return dates


def generate_filename(prefix: str, extension: str, include_date: bool = True) -> str:
    """生成文件名"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if include_date:
        return f"{prefix}_{timestamp}.{extension}"
    return f"{prefix}.{extension}"


def ensure_directory_exists(dir_path: str) -> str:
    """确保目录存在"""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_data_dir(base_dir: str = "data") -> str:
    """获取数据目录路径"""
    return str(Path(base_dir).resolve())


def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件信息"""
    path = Path(file_path)
    try:
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "suffix": path.suffix
        }
    except FileNotFoundError:
        return {}


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """安全除法"""
    if denominator == 0:
        return default
    return numerator / denominator


def remove_outliers(data: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """移除异常值"""
    data = data.copy()

    if method == 'iqr':
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        filtered = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    elif method == 'zscore':
        mean = data[column].mean()
        std = data[column].std()
        z_scores = (data[column] - mean) / std
        filtered = data[abs(z_scores) <= threshold]
    else:
        raise ValueError(f"Unknown method: {method}")

    removed_count = len(data) - len(filtered)
    logging.info(f"Removed {removed_count} outliers from {column}")
    return filtered


def standardize(data: pd.Series) -> pd.Series:
    """标准化数据"""
    mean = data.mean()
    std = data.std()
    if std == 0:
        return data - mean
    return (data - mean) / std


def normalize(data: pd.Series) -> pd.Series:
    """归一化数据到 [0, 1] 范围"""
    min_val = data.min()
    max_val = data.max()
    if min_val == max_val:
        return data - min_val
    return (data - min_val) / (max_val - min_val)
