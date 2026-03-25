# VSCode Code Runner 配置指南

## 问题解决

如果您在 VSCode 中使用 Code Runner 运行代码时遇到 `ModuleNotFoundError: No module named 'pandas'` 错误，这是因为 Code Runner 没有使用项目的虚拟环境。

## 解决方案

### 方法 1: 使用 .vscode/settings.json (已创建)

我们已经在项目根目录创建了 `.vscode/settings.json` 文件，该文件会自动配置 Code Runner 使用项目的虚拟环境。

**请重启 VSCode**，然后重新尝试运行代码。

### 方法 2: 在 VSCode 中手动选择 Python 解释器

1. 打开 VSCode
2. 按 `Ctrl+Shift+P` (Windows) 或 `Cmd+Shift+P` (Mac)
3. 输入 "Python: Select Interpreter"
4. 选择项目的虚拟环境：`.venv\Scripts\python.exe`
5. 重新运行代码

### 方法 3: 在终端中手动运行

如果以上方法仍然有问题，您可以直接在 VSCode 终端中运行：

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 运行代码
python main.py list
```

### 方法 4: 使用 Code Runner 的终端模式

我们的配置文件已经设置了 `"code-runner.runInTerminal": true`，这会让 Code Runner 在 VSCode 终端中运行，这样应该可以正确使用虚拟环境。

## 验证配置

您可以运行以下命令来验证虚拟环境是否正确激活：

```bash
.venv\Scripts\python -c "import sys; print(sys.executable)"
```

这应该输出项目虚拟环境的 Python 路径：
`d:\Github\gitclone\quant_agent_research\.venv\Scripts\python.exe`

您还可以检查已安装的包：

```bash
.venv\Scripts\pip list
```

## 快速测试

配置完成后，运行以下测试：

```bash
# 列出可用智能体
.venv\Scripts\python main.py list

# 运行完整系统测试
.venv\Scripts\python test_system_simple.py
```
