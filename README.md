# 妙妙期末小工具

本项目基于[GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)协议开源，Don't be an asshole!

## What is this

本项目旨在构建一个用AI辅助期末复习的工具（没有作弊功能），当前已经实现的功能有：

1. 从老师的复习课录音生成复习（预习）笔记
2. 从老师的教学PPT生成复习(预习)笔记
3. **Ask AI**：从已经生成的笔记和原文提供的上下文中回答问题
4. **知识库管理**：按科目分组管理 PDF，支持独立索引和搜索
5. **缓存功能**：基于文件哈希的智能缓存，避免重复处理
6. **分块总结**：处理超长文本，突破模型上下文窗口限制

计划实现的功能有：

1. 往年题&作业题库：从往年的作业题和作业题库生成复习（预习）题和答案解析

## But how to use it?

1. 从任何一个提供OpenAI Like API的LLM提供商获取一个API Key
2. 将.env.example文件重命名为.env, 并修改其中的配置项
3. 使用 [uv](https://docs.astral.sh/uv/) 初始化项目:

```bash
uv venv .venv
. .venv/bin/activate.sh
uv pip install -e .
```

**注意：**

- activate具体使用哪个文件取决于你的shell，具体可参考uv文档
- 本项目依赖于pdf2image, 而pdf2image又依赖于poppler，因此你自行需要安装poppler。具体见[pdf2image文档](https://pdf2image.readthedocs.io/en/latest/installation.html#installing-poppler)


4. 运行项目：

### GUI 模式 (推荐)

```bash
python main.py --gui
```

GUI 模式将在浏览器中打开 Web 界面，提供更友好的用户体验：
- 支持拖拽上传文件
- 实时进度显示
- 结果预览和下载

### CLI 模式

```bash
python main.py --type [audio|pdf] --input [input_files] --output [output_dir]
```

| 参数 | 描述 |
| --- | --- |
| --gui | 启动 GUI 模式 |
| --type | 输入文件类型，可选值为audio和pdf (CLI 模式必需) |
| --input | 输入文件路径，支持通配符 (CLI 模式必需) |
| --output | 输出目录 (CLI 模式必需) |
| --dump | 是否将中间结果保存到cache中 (默认: True) |
| --no-dump | 不保存中间结果 |
| --dump-dir | cache目录 (默认: ./cache/) |

## FAQ

### 为什么我的笔记非常的诡异？

这通常出现在STT模式下，由于录音环境的问题，STT的结果通常非常破碎，因此需要足够强大的模型来处理这样的文本，以获得更准确的结果。目前笔者测试成功的模型有：

- Qwen3-Next-80B-A3B-Instruct
- DeepSeek-V3.2-Exp
- Gemini-2.5-Flash (这部分需要自行适配Gemini的API)

推荐使用Qwen3-Next-80B-A3B-Instruct模型，它在总结效果和价格上取得了一个不错的平衡，且推理速度非常快。

### 为什么我的STT请求失败了

实际上笔者只适配了硅基流动的ASR模型，如果需要使用别家的模型，需要自行适配。此外，ASR_API_URL作为独立的配置项出现的原因是ASR模型的URL地址需要填写完整，例如：
[https://api.siliconflow.cn/v1/audio/transcriptions](https://api.siliconflow.cn/v1/audio/transcriptions)

## Changelog

### [0.7.0] - 2024-12-30

**新增：知识库管理系统**
- 新增 `utilities/knowledge_base/` 模块：知识库管理功能
  - `manager.py`: KnowledgeBaseManager，管理知识库的 CRUD 操作
  - `models.py`: KnowledgeBase 数据模型和存储
- 支持按科目创建知识库，独立管理 PDF 文档
- 支持为每个知识库构建独立的语义搜索索引
- GUI 新增"知识库管理"标签页，提供可视化管理界面

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

**使用方式：**
```bash
# GUI 模式 (推荐)
python main.py --gui

# 新功能：
# - Ask AI: 基于已处理内容回答问题
# - 知识库管理: 按科目分组管理 PDF
# - 分块总结: 处理超长文本
```

### [0.6.0] - 2024-12-30

**新增：Gradio Web GUI**
- 新增 `gui.py`：基于 Gradio 的 Web 图形界面
- main.py 新增 `--gui` 参数，启动 GUI 模式
- GUI 提供 PDF 和音频处理的友好界面
- 支持文件拖拽上传、实时进度显示、结果预览和下载

**架构变更：**
- `gui.py`: GradioApp 类，管理 GUI 生命周期和 Worker 初始化
- `main.py`: 支持 GUI/CLI 双模式启动

**使用方式：**
```bash
# GUI 模式 (推荐)
python main.py --gui

# CLI 模式
python main.py --type pdf --input *.pdf --output ./output
```

### [0.5.0] - 2024-12-30

**新增：OCR 缓存与结构化数据存储**
- 新增 `utilities/models.py`：结构化数据模型（OCRResult、OCRSection、PDFCache）
- 新增 `utilities/cache.py`：基于文件哈希的 OCR 缓存管理器
- OCRWorker 新增 `process_images_structured()` 方法，返回结构化 OCRResult
- main.py PDF 处理管线现在支持缓存，避免重复 OCR 处理
- 缓存基于 PDF 文件的 MD5 哈希，支持缓存有效性验证（修改时间检查）

**架构变更：**
- `utilities/models.py`: 数据模型定义，包含 JSON 序列化/反序列化
- `utilities/cache.py`: OCRCache 管理器，提供 `get_pdf_cache()` 和 `save_pdf_cache()` 方法
- `utilities/workers/ocr_worker.py`: 新增 `_ocr_structured()` 和 `process_images_structured()` 方法
- `main.py`: `process_pdf_pipeline()` 现在使用缓存和结构化 OCR 结果

**测试：**
- 所有 100 个单元测试通过
- 代码覆盖率提升至 94%（models.py 100%, cache.py 96%）

### [0.4.0] - 2024-12-30

**重构：精简架构，PDF 处理合并到 main.py**
- 将 DumpPDF 的 PDF 转图片逻辑合并到 main.py
- 删除 `utilities/DumpPDF.py`，所有 PDF 处理逻辑集中在 main.py 的 `convert_pdf_to_images` 函数
- 代码覆盖率提升至 94%

**架构简化：**
- `main.py`: 新增 `convert_pdf_to_images` 函数，处理 PDF 到图片的转换
- 删除：`utilities/DumpPDF.py`

**测试：**
- 所有 60 个单元测试通过
- 代码覆盖率：91% → 94%

### [0.3.0] - 2024-12-30

**重构：精简架构，所有逻辑合并到 Worker**
- 将 DeepSeekOCR、GenericVL、STT、Summarization 的逻辑直接合并到各自的 Worker 中
- 删除冗余的中间层类，简化代码结构
- 代码覆盖率提升至 91%

**架构简化：**
- `utilities/workers/ocr_worker.py`: 包含所有 OCR 逻辑（DeepSeek + GenericVL）
- `utilities/workers/stt_worker.py`: 包含所有 STT 逻辑
- `utilities/workers/summarization_worker.py`: 包含所有 Summarization 逻辑和 TextSource 枚举
- 删除：`DeepSeekOCR.py`, `GenericVisionLanguageOCR.py`, `STT.py`, `Summarization.py`

**测试：**
- 更新所有测试以适应新架构
- 所有 60 个单元测试通过

### [0.2.0] - 2024-12-30

**重构：Worker 架构与流控**
- 新增 Worker 架构，所有 AI 服务通过独立的 daemon 线程处理
- 新增基于令牌桶算法的 RPM/TPM 流控机制
- 为每个服务添加独立的 API 配置（URL、Key、Model）
- 实现优雅关闭，确保队列中的任务完成后再退出

**新增配置项：**
- `OCR_API_URL` / `OCR_API_KEY` / `OCR_RPM` / `OCR_TPM`
- `SUMMARIZATION_API_URL` / `SUMMARIZATION_API_KEY` / `SUMMARIZATION_RPM` / `SUMMARIZATION_TPM`
- ASR 服务保持原有独立配置

**测试：**
- 新增完整的单元测试框架 (pytest)
- 60 个单元测试，覆盖核心功能
- 代码覆盖率约 70%

**架构变更：**
- `utilities/rate_limiter.py`: 令牌桶流控实现
- `utilities/worker.py`: Worker 基类
- `utilities/workers/`: OCRWorker, STTWorker, SummarizationWorker

### [0.1.0] - 2024-12-21

**初始版本**
- PDF 转 OCR 并生成总结笔记
- 音频转 STT 并生成总结笔记
- 支持 OpenAI Like API
