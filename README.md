# 妙妙期末小工具

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[English](README_en.md) | 简体中文

---

## 项目简介

本项目是一个用 AI 辅助期末复习的工具，帮助从课程材料中自动生成复习笔记。

**注意：本工具仅用于学习辅助，不包含任何作弊功能。**

### 主要功能

| 功能 | 描述 |
|------|------|
| **PDF 处理** | 从教学 PPT 生成结构化复习笔记 |
| **音频处理** | 从复习课录音生成复习笔记 |
| **Ask AI** | 基于已处理内容回答问题，支持知识库隔离 |
| **知识库管理** | 按科目分组管理 PDF，独立索引和搜索 |
| **智能缓存** | 基于文件哈希的缓存，避免重复处理 |
| **分块总结** | 处理超长文本，突破模型上下文窗口限制 |
| **费用跟踪** | 自动跟踪 API 使用量和费用（可选配置） |

---

## 快速开始

### 前置要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (推荐的 Python 包管理器)
- poppler-utils (PDF 处理依赖)

### 一键安装 (推荐)

```bash
# 克隆项目
git clone https://github.com/yourusername/MagicExamUtilities.git
cd MagicExamUtilities

# 创建虚拟环境并安装依赖
uv venv .venv
source .venv/bin/activate.sh  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 安装 poppler

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
下载并安装 [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

---

## 配置指南

### 1. 获取 API Key

本工具需要使用支持 OpenAI 兼容 API 的 LLM 服务。推荐的服务商：

| 服务商 | 网址 | 推荐模型 |
|--------|------|----------|
| 硅基流动 | https://siliconflow.cn | Qwen3-Next-80B-A3B-Instruct |
| DeepSeek | https://platform.deepseek.com | DeepSeek-V3 |

### 2. 配置环境变量

复制示例配置文件并填入你的 API 信息：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下必要项：

```bash
# === OCR 配置 (PDF 处理) ===
OCR_API_URL=https://api.siliconflow.cn/v1
OCR_API_KEY=your_api_key_here
OCR_MODEL=deepseek-ai/DeepSeek-OCR

# === 总结配置 (生成笔记) ===
SUMMARIZATION_API_URL=https://api.siliconflow.cn/v1
SUMMARIZATION_API_KEY=your_api_key_here
SUMMARIZATION_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct

# === ASR 配置 (音频处理，可选) ===
ASR_API_URL=https://api.siliconflow.cn/v1/audio/transcriptions
ASR_API_KEY=your_api_key_here
ASR_MODEL=TeleAI/TeleSpeechASR

# === Ask AI 配置 (问答功能) ===
ASK_AI_API_URL=https://api.siliconflow.cn/v1
ASK_AI_API_KEY=your_api_key_here
ASK_AI_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
```

**（可选）配置模型价格用于费用跟踪：**

```bash
# === 模型价格配置（费用跟踪，可选） ===
# 价格单位：美元/百万 tokens
# 参考价格（硅基流动，2024-12）：
# - Qwen3-VL-8B-Instruct (OCR): 输入 $0.014/M, 输出 $0.028/M
# - Qwen3-Next-80B-A3B-Instruct: 输入 $0.14/M, 输出 $0.28/M
# - TeleSpeechASR (STT): 输入 $0.1/M, 输出 $0.1/M

OCR_INPUT_PRICE_PER_M=0.014
OCR_OUTPUT_PRICE_PER_M=0.028
SUMMARIZATION_INPUT_PRICE_PER_M=0.14
SUMMARIZATION_OUTPUT_PRICE_PER_M=0.28
ASK_AI_INPUT_PRICE_PER_M=0.14
ASK_AI_OUTPUT_PRICE_PER_M=0.28
```

### 3. 验证配置

运行以下命令验证配置是否正确：

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OCR Model:', os.getenv('OCR_MODEL'))"
```

---

## 使用方法

### GUI 模式 (推荐)

```bash
python main.py --gui
```

浏览器会自动打开 Web 界面，提供以下功能：

#### 1. PDF 处理
- 上传 PDF 文件 (支持多文件)
- 可选择输出目录
- 启用缓存加速重复处理
- 高级选项：启用分块总结处理超长文档

#### 2. 音频处理
- 上传 MP3 录音文件
- 自动转录并生成笔记

#### 3. Ask AI
- 输入问题，AI 从已处理内容中检索并回答
- 可选择特定知识库限定搜索范围

#### 4. 知识库管理
- 创建知识库 (按科目分组)
- 添加/移除 PDF 文档
- 重建搜索索引

### CLI 模式

```bash
# 处理 PDF
python main.py --type pdf --input slides/*.pdf --output ./notes

# 处理音频
python main.py --type audio --input recording.mp3 --output ./notes

# 指定缓存目录
python main.py --type pdf --input file.pdf --output ./notes --dump-dir ./cache
```

---

## 常见问题

### Q: 为什么生成的笔记质量不佳？

**A:** 这通常取决于：
1. **模型选择** - 推荐使用 `Qwen3-Next-80B-A3B-Instruct` 或 `DeepSeek-V3`
2. **音频质量** - 录音环境嘈杂会导致 STT 结果破碎
3. **PDF 质量** - 扫描件或图片质量差会影响 OCR 效果

### Q: STT 请求失败怎么办？

**A:** 本工具目前仅适配了硅基流动的 ASR API。如需使用其他服务，请检查 `.env` 中的 `ASR_API_URL` 是否正确。

### Q: 如何处理超长文档？

**A:** 在 GUI 中展开"高级选项"，勾选"启用分块总结"。文档会自动分块处理，突破模型上下文窗口限制。

### Q: 缓存文件在哪里？

**A:** 默认存储在 `./cache/` 目录，包含：
- `ocr/` - OCR 结果缓存
- `stt/` - STT 结果缓存
- `knowledge_bases.json` - 知识库配置
- `kb_indices/` - 知识库索引

---

## 功能详解

### 知识库管理

创建知识库可以按科目组织 PDF，实现独立搜索：

```
1. 在"知识库管理"标签页创建知识库
2. 选择相关的 PDF 文档加入
3. 系统自动构建语义索引
4. 在 Ask AI 中选择该知识库进行问答
```

### 分块总结

当文档超过模型上下文窗口时：

1. 文档自动按段落/句子分块
2. 每块独立生成总结
3. 合并所有总结
4. 如合并结果仍过长，进行二次总结

### 费用跟踪

启用费用跟踪后，系统会自动记录每次 API 调用的 token 使用量和费用：

**日志示例：**
```
OCR Done for image: page_0.jpg, length: 1234, usage: 1200 in + 800 out = 2000 total, cost: $0.000036
Summarization Done for text, length: 5678, usage: 4500 in + 1200 out = 5700 total, cost: $0.000468
```

**Worker 关闭时显示汇总：**
```
[OCRWorker] Cost summary: requests: 10, input_tokens: 12,000, output_tokens: 8,000, total_cost: $0.000360
```

**配置方法：** 在 `.env` 文件中设置对应服务的 `_INPUT_PRICE_PER_M` 和 `_OUTPUT_PRICE_PER_M` 环境变量。

---

## 开发计划

- [ ] 往年题&作业题库：从往年作业生成复习题和答案解析
- [ ] 更多 ASR 服务适配
- [ ] 导出 Anki 卡片格式

---

## 开源协议

本项目基于 [GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html) 协议开源。

**Don't be an asshole!**

---

## 更新日志

详见 [Changelog](#changelog) 章节。

### [0.7.2] - 2024-12-30

**新增：模型费用跟踪功能**
- BaseWorker 新增 CostTracker 类，自动跟踪 token 使用量和费用
- 支持输入和输出价格分别配置（价格单位：美元/百万 tokens）
- 每次请求日志显示详细 token 使用和费用信息
- Worker 关闭时打印费用汇总（请求数、输入 tokens、输出 tokens、总费用）
- 价格配置为可选项，未设置时功能正常使用（不显示费用）

**更新：环境变量配置**
- .env.example 新增"Model Pricing Configuration"章节
- 提供常见模型价格参考表（SiliconFlow 2024-12）
- 支持 OCR、ASR、SUMMARIZATION、ASK_AI 四个服务的独立价格配置

**测试：**
- 新增 CostTracker 单元测试（test_worker.py）
- 新增 get_pricing_config 函数测试
- 新增 BaseWorker 费用跟踪集成测试

### [0.7.1] - 2024-12-30

**修复：Qwen 模型 Markdown 代码块包装问题**
- 修复 Qwen 系列模型自动添加外层 ```markdown 代码块的问题
- 新增 `_strip_outer_markdown_block()` 函数自动检测并移除
- 保留文档内合法的代码块不受影响

**修复：VLM OCR 编造图片 URL 问题**
- 修复 VLM 在处理包含图片的文档时编造假 URL 的问题
- 更新 OCR 提示词，明确禁止编造任何 URL 或文件路径
- 新增图片处理规则，指导模型用文字描述图片内容

**优化：OCR 和总结提示词重构**
- 重写 Generic VLM OCR 提示词，结构更清晰
- 重写 OCR 总结提示词（PDF 专用），强调结构化组织
- 重写 STT 总结提示词（音频专用），强调去除口语冗余
- 添加详细的特殊内容处理表格
- 优化用户提示词，更简洁明确

### [0.7.0] - 2024-12-30

**新增：知识库管理系统**
- 新增 `utilities/knowledge_base/` 模块
- 支持按科目创建知识库，独立管理 PDF 文档
- 支持为每个知识库构建独立的语义搜索索引
- GUI 新增"知识库管理"标签页

**新增：Ask AI 功能**
- 新增 `utilities/workers/ask_ai_worker.py`：基于 LangChain 的 AI 问答 Worker
- 集成检索工具，支持从缓存的 PDF/音频内容中召回相关信息
- 支持知识库隔离搜索，限定搜索范围到特定科目
- GUI 新增"Ask AI"标签页，提供问答界面

**新增：分块总结功能**
- SummarizationWorker 新增分块总结模式，支持处理超长文本
- 智能分块：按段落和句子分割，保持语义完整性
- 两阶段总结：先分块总结，再合并摘要，突破上下文窗口限制
- GUI 新增"高级选项"，可配置分块阈值

**优化：GUI 布局重构**
- 重构为上下流式中心对齐布局
- 使用 Accordion 组件折叠高级选项
- 添加清空按钮，方便重置表单
- 优化视觉层次和间距

**架构变更：**
- `utilities/retrieval/`: 新增检索模块
  - `retriever.py`: CacheRetriever，支持关键词和语义搜索
  - `tools.py`: LangChain 工具定义
  - `cache_loader.py`: 缓存加载器
- `utilities/workers/ask_ai_worker.py`: Ask AI Worker

### [0.6.0] - 2024-12-30

**新增：Gradio Web GUI**
- 新增 `gui.py`：基于 Gradio 的 Web 图形界面
- main.py 新增 `--gui` 参数，启动 GUI 模式
- GUI 提供 PDF 和音频处理的友好界面
- 支持文件拖拽上传、实时进度显示、结果预览和下载

**架构变更：**
- `gui.py`: GradioApp 类，管理 GUI 生命周期和 Worker 初始化
- `main.py`: 支持 GUI/CLI 双模式启动

### [0.5.0] - 2024-12-30

**新增：OCR 缓存与结构化数据存储**
- 新增 `utilities/models.py`：结构化数据模型（OCRResult、OCRSection、PDFCache）
- 新增 `utilities/cache.py`：基于文件哈希的 OCR 缓存管理器
- OCRWorker 新增 `process_images_structured()` 方法，返回结构化 OCRResult
- main.py PDF 处理管线现在支持缓存，避免重复 OCR 处理
- 缓存基于 PDF 文件的 MD5 哈希，支持缓存有效性验证（修改时间检查）

### [0.4.0] - 2024-12-30

**重构：精简架构，PDF 处理合并到 main.py**
- 将 DumpPDF 的 PDF 转图片逻辑合并到 main.py
- 删除 `utilities/DumpPDF.py`

### [0.3.0] - 2024-12-30

**重构：精简架构，所有逻辑合并到 Worker**
- 将 DeepSeekOCR、GenericVL、STT、Summarization 的逻辑直接合并到各自的 Worker 中
- 删除冗余的中间层类，简化代码结构

### [0.2.0] - 2024-12-30

**重构：Worker 架构与流控**
- 新增 Worker 架构，所有 AI 服务通过独立的 daemon 线程处理
- 新增基于令牌桶算法的 RPM/TPM 流控机制
- 为每个服务添加独立的 API 配置（URL、Key、Model）
- 实现优雅关闭，确保队列中的任务完成后再退出

### [0.1.0] - 2024-12-21

**初始版本**
- PDF 转 OCR 并生成总结笔记
- 音频转 STT 并生成总结笔记
- 支持 OpenAI Like API
