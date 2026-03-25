"""
数据智能体模块
负责数据抓取、清洗和存储
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import logging

from core.agent_base import BaseAgent, AgentType, TaskResult
from core.utils import (
    save_data, load_data, ensure_directory_exists,
    generate_filename, get_file_info
)


@dataclass
class DataSource:
    """数据源配置"""
    name: str
    description: str
    required_params: List[str]
    supported_assets: List[str]


class DataAgent(BaseAgent):
    """
    数据智能体
    负责从各种数据源抓取金融数据
    """

    def __init__(
        self,
        name: str = "DataAgent",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}
        super().__init__(name, AgentType.DATA, config)

        # 数据源配置
        self.data_sources = self._setup_data_sources()
        self.data_dir = config.get("data_dir", "data/raw")
        ensure_directory_exists(self.data_dir)

        # 默认配置
        self.default_config = {
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "assets": ["600036.SH", "000001.SZ"],
            "data_types": ["daily", "minute"],
            "source": "yfinance",
            "frequency": "1d"
        }
        self.config = {**self.default_config, **config}

        self.available_symbols = []
        self.downloaded_files = []

    def _setup_data_sources(self) -> Dict[str, DataSource]:
        """设置支持的数据源"""
        return {
            "tushare": DataSource(
                name="Tushare",
                description="Tushare 金融数据接口",
                required_params=["token", "symbol", "start_date", "end_date"],
                supported_assets=["stock", "index", "futures"]
            ),
            "akshare": DataSource(
                name="AkShare",
                description="AkShare A股数据接口",
                required_params=["symbol", "start_date", "end_date"],
                supported_assets=["stock", "fund", "futures"]
            ),
            "yfinance": DataSource(
                name="Yahoo Finance",
                description="Yahoo Finance 数据接口",
                required_params=["symbol", "start_date", "end_date"],
                supported_assets=["stock", "index", "etf", "currency", "crypto"]
            ),
            "local": DataSource(
                name="Local Files",
                description="本地文件数据源",
                required_params=["file_path"],
                supported_assets=["stock", "index", "etf"]
            )
        }

    def initialize(self) -> TaskResult:
        """初始化数据智能体"""
        self.logger.info("DataAgent initialized")
        return TaskResult(success=True)

    def execute(self, input_data: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行数据抓取任务"""
        config = {**self.config, **(input_data or {})}
        self.logger.info(f"DataAgent executing with config: {config}")

        results = []

        try:
            source_name = config.get("source", "yfinance")

            if source_name == "local":
                data = self._load_local_data(config)
            else:
                data = self._fetch_data_from_source(source_name, config)

            # 数据质量检查
            quality_check = self._check_data_quality(data)

            if quality_check["completeness"] < 0.9:
                return TaskResult(
                    success=False,
                    errors=["Data completeness too low", str(data)]
                )

            # 保存数据
            save_path = self._save_data(data, config)

            self.downloaded_files.append(save_path)
            self.logger.info(f"Data saved to: {save_path}")

            return TaskResult(
                success=True,
                data={
                    "data": data,
                    "file_path": save_path,
                    "quality_check": quality_check,
                    "config": config
                },
                warnings=[],
                execution_time=0.0
            )

        except Exception as e:
            error_msg = f"DataAgent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return TaskResult(
                success=False,
                errors=[error_msg],
                warnings=[]
            )

    def _fetch_data_from_source(self, source_name: str, config: Dict[str, Any]) -> pd.DataFrame:
        """从指定数据源抓取数据"""
        if source_name == "yfinance":
            return self._fetch_yfinance_data(config)
        elif source_name == "akshare":
            return self._fetch_akshare_data(config)
        elif source_name == "tushare":
            return self._fetch_tushare_data(config)
        else:
            raise ValueError(f"Unknown data source: {source_name}")

    def _fetch_yfinance_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """从 Yahoo Finance 抓取数据"""
        import yfinance as yf

        symbols = config.get("assets", ["600036.SH"])
        start_date = config.get("start_date")
        end_date = config.get("end_date")

        all_data = []

        for symbol in symbols:
            try:
                self.logger.info(f"Fetching data for {symbol} from Yahoo Finance")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue

                data["symbol"] = symbol
                data.reset_index(inplace=True)

                # 规范化列名
                data = data.rename(columns={
                    "Date": "datetime",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })

                all_data.append(data)
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol} from Yahoo Finance: {str(e)}")

        if not all_data:
            raise Exception("No data fetched from Yahoo Finance")

        return pd.concat(all_data, ignore_index=True)

    def _fetch_akshare_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """从 AkShare 抓取 A股数据"""
        try:
            import akshare as ak

            symbols = config.get("assets", ["000001"])
            start_date = config.get("start_date")
            end_date = config.get("end_date")

            all_data = []

            for symbol in symbols:
                try:
                    self.logger.info(f"Fetching data for {symbol} from AkShare")

                    # 获取股票历史行情数据
                    data = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", "")
                    )

                    if data.empty:
                        self.logger.warning(f"No data found for {symbol}")
                        continue

                    data["symbol"] = symbol

                    all_data.append(data)
                except Exception as e:
                    self.logger.error(f"Failed to fetch {symbol} from AkShare: {str(e)}")

            if not all_data:
                raise Exception("No data fetched from AkShare")

            return pd.concat(all_data, ignore_index=True)

        except ImportError:
            self.logger.error("AkShare not installed")
            raise

    def _fetch_tushare_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """从 Tushare 抓取数据"""
        token = self.config.get("tushare_token", config.get("token"))

        if not token:
            raise Exception("Tushare token not configured")

        try:
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()

            symbols = config.get("assets", ["000001.SZ"])
            start_date = config.get("start_date")
            end_date = config.get("end_date")

            all_data = []

            for symbol in symbols:
                try:
                    self.logger.info(f"Fetching data for {symbol} from Tushare")

                    # 获取股票日线数据
                    data = pro.daily(
                        ts_code=symbol,
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", "")
                    )

                    if data.empty:
                        self.logger.warning(f"No data found for {symbol}")
                        continue

                    # 处理日期格式
                    data["trade_date"] = pd.to_datetime(data["trade_date"])
                    data = data.sort_values("trade_date")
                    data.reset_index(drop=True, inplace=True)

                    all_data.append(data)
                except Exception as e:
                    self.logger.error(f"Failed to fetch {symbol} from Tushare: {str(e)}")

            if not all_data:
                raise Exception("No data fetched from Tushare")

            return pd.concat(all_data, ignore_index=True)

        except ImportError:
            self.logger.error("Tushare not installed")
            raise

    def _load_local_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """加载本地数据文件"""
        file_path = config.get("file_path")
        if not file_path:
            raise Exception("File path not specified")

        try:
            data = load_data(file_path)
            self.logger.info(f"Loaded local data from {file_path}")
            return data
        except Exception as e:
            raise Exception(f"Failed to load local data: {str(e)}")

    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """数据质量检查"""
        if data.empty:
            return {
                "completeness": 0,
                "validity": 0,
                "consistency": 0,
                "missing_rate": 1
            }

        missing_rate = data.isnull().sum().sum() / data.size
        completeness = 1 - missing_rate
        validity = self._validate_data_types(data)

        return {
            "completeness": completeness,
            "validity": validity,
            "consistency": 1.0,
            "missing_rate": missing_rate
        }

    def _validate_data_types(self, data: pd.DataFrame) -> float:
        """验证数据类型"""
        required_columns = ["datetime", "open", "high", "low", "close", "volume"]
        valid_cols = 0

        for col in required_columns:
            if col in data.columns:
                valid_cols += 1

        return valid_cols / len(required_columns)

    def _save_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """保存数据到文件"""
        symbol = config.get("assets", ["unknown"])[0]
        source = config.get("source", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{source}_{symbol}_{timestamp}.csv"
        save_path = str(Path(self.data_dir) / filename)

        save_data(data, save_path)
        self.downloaded_files.append(save_path)

        return save_path

    def list_available_sources(self) -> List[str]:
        """列出可用数据源"""
        return list(self.data_sources.keys())

    def get_source_info(self, source_name: str) -> Optional[Dict[str, Any]]:
        """获取数据源信息"""
        source = self.data_sources.get(source_name)
        if source:
            return {
                "name": source.name,
                "description": source.description,
                "required_params": source.required_params,
                "supported_assets": source.supported_assets
            }
        return None

    def get_downloaded_files(self) -> List[str]:
        """获取下载文件列表"""
        return self.downloaded_files

    def cleanup(self) -> TaskResult:
        """清理资源"""
        self.logger.info("DataAgent cleaned up")
        return TaskResult(success=True)
