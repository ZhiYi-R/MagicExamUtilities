# 妙妙期末小工具

## What is this

本项目旨在构建一个用AI辅助期末复习的工具（没有作弊功能），当前已经实现的功能有：

1. 从老师的复习课录音生成复习（预习）笔记
2. 从老师的教学PPT生成复习(预习)笔记
   
计划实现的功能有：

1. ASK AI：从已经生成的笔记和原文提供的上下文中回答问题
2. 缓存功能：如果cache中已经存在ocr/stt结果且修改时间晚于输入文件，则直接从cache中读取
3. 往年题&作业题库：从往年的作业题和作业题库生成复习（预习）题和答案解析

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

```
python main.py --type [audio|pdf] --input [input_files] --dump --dump-dir [cache_dir] --output [output_dir]
```

| 参数 | 描述 |
| --- | --- |
| --type | 输入文件类型，可选值为audio和pdf |
| --input | 输入文件路径，支持通配符, 对于audio类型，仅支持mp3文件, 对于pdf类型，仅支持pdf文件 |
| --dump | 是否将结果保存到cache中 |
| --dump-dir | cache目录 |
| --output | 输出目录 |

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