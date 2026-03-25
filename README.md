# Multi-Agent 量化投研自动化系统

一个基于多智能体协作的量化投研自动化系统，实现从数据抓取、因子生成、策略构建、回测评估到报告生成的全流程自动化。

## 系统架构

### 智能体类型

1. **Data Agent（数据智能体）** - 负责数据抓取、清洗和存储
2. **Factor Agent（因子智能体）** - 负责技术指标和因子计算
3. **Strategy Agent（策略智能体）** - 负责策略构建和优化
4. **Backtest Agent（回测智能体）** - 负责回测执行和性能评估
5. **Risk Agent（风险智能体）** - 负责风险分析和管理
6. **Report Agent（报告智能体）** - 负责报告生成和可视化

### 协作流程

```
Data Agent → Factor Agent → Strategy Agent → Backtest Agent → Risk Agent → Report Agent
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行系统

```bash
python main.py
```

## 项目结构

```
quant_agent_research/
├── agents/              # 智能体模块
│   ├── data_agent.py    # 数据智能体
│   ├── factor_agent.py  # 因子智能体
│   ├── strategy_agent.py # 策略智能体
│   ├── backtest_agent.py # 回测智能体
│   ├── risk_agent.py    # 风险智能体
│   └── report_agent.py  # 报告智能体
├── core/               # 核心组件
│   ├── agent_base.py   # 智能体基类
│   ├── pipeline.py     # 流程管道
│   └── utils.py        # 工具函数
├── data/               # 数据目录
│   ├── raw/            # 原始数据
│   ├── processed/      # 处理后数据
│   └── factors/        # 因子数据
├── strategies/         # 策略目录
├── backtests/          # 回测结果
├── reports/            # 报告目录
├── config/             # 配置文件
├── main.py             # 主入口
└── requirements.txt    # 依赖包
```

## 配置说明

### 数据源配置

支持以下数据源：
- Tushare（需要配置 Token）
- Yahoo Finance
- AkShare（A股数据）
- 本地 CSV 文件

### 因子库

预定义因子：
- 技术指标：MA、MACD、RSI、KDJ、OBV 等
- 波动率因子：ATR、Bollinger Bands 等
- 动量因子：ROC、WR 等

### 策略类型

支持策略：
- 均线策略
- 动量策略
- 反转策略
- 多因子策略

## 使用指南

### 自定义因子

在 `agents/factor_agent.py` 中添加新因子计算方法。

### 自定义策略

在 `strategies/` 目录下创建新策略文件。

### 回测配置

修改 `config/backtest_config.yaml` 配置回测参数。

## 输出结果

- 数据文件：`data/` 目录
- 因子计算结果：`data/factors/` 目录
- 策略参数：`strategies/` 目录
- 回测结果：`backtests/` 目录
- 报告文件：`reports/` 目录（包含 HTML、PDF 格式）

## 技术栈

- Python 3.9+
- 量化库：pandas, numpy, talib
- 回测框架：backtrader, vectorbt
- 可视化：matplotlib, seaborn, plotly
- 报告生成：jinja2, weasyprint
- 多智能体：autogen, langchain
