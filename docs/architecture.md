# 项目架构说明

本项目由索引构建、检索与生成、离线评测、稳定性检测与 Web 展示等部分组成。以下按目录树状结构说明各部分功能与依赖关系。

## 目录树（概览）

```
.
├── 01_index.py
├── 02_chat.py
├── 03_eval.py
├── 04_stability.py
├── build_web_docs.py
├── config.yaml
├── data/
├── prompts/
├── scripts/
├── src/
│   ├── chat_engine.py
│   ├── chat_memory.py
│   ├── common.py
│   ├── loader_jsonl.py
│   ├── memory_store.py
│   ├── model_factory.py
│   ├── query_rewrite.py
│   └── rag_retrieval.py
├── web/
│   ├── index.html
│   ├── main.js
│   ├── styles.css
│   └── docs.json
└── web_app.py
```

## 顶层入口脚本

### 01_index.py
索引构建入口，读取 `config.yaml` 的路径与模型配置，加载 JSONL/文本数据并构建 FAISS 向量索引，输出持久化目录与跳过日志。

### 02_chat.py
命令行问答入口，通过 `ChatEngine` 加载索引与模型，执行检索与回答生成，并输出检索摘要与性能指标。

### 03_eval.py
离线检索评测入口，读取 `data/eval_queries.jsonl` 与 `data/eval_qrels.jsonl`，执行检索后计算 P@K、R@K、MRR、nDCG 等指标，并输出诊断报告。

### 04_stability.py
稳定性评测入口，对同一批查询重复检索与生成，统计检索结果一致性与答案相似度，输出稳定性报告。

### build_web_docs.py
将 JSONL 方法清单与额外文档合并，生成前端使用的 `web/docs.json`，包含摘要、标签与示例片段。

### web_app.py
Web 服务入口，提供静态资源与 API 服务（/api/chat、/api/health），用于前端页面与对话接口。

## 核心模块（src）

### chat_engine.py
聊天引擎核心，负责加载配置、向量索引与 LLM，并协调检索、提示词构建与回答生成，输出统一的响应结构。

### chat_memory.py
会话记忆管理器，封装对话历史的读取与写入，支持内存存储与历史合并。

### common.py
通用工具函数，提供 YAML 配置读取与文本读取能力。

### loader_jsonl.py
JSONL 加载器，将每行 JSON 转为 `Document`，并构建用于检索的文本与元数据。

### memory_store.py
会话存储接口与内存实现，提供会话读取、追加、清空、过期管理与统计信息。

### model_factory.py
模型工厂，统一构建 embedding 与 LLM，支持 Ollama 与 OpenAI 兼容接口。

### query_rewrite.py
查询改写模块，可选调用 LLM 生成改写候选，支持缓存、去重与相似度门控。

### rag_retrieval.py
检索与融合逻辑，负责基础检索、阈值过滤、关键词加权、rerank 及多查询 RRF 融合。

## Web 前端（web）

### index.html
页面结构，包含文档浏览区与对话区布局。

### main.js
前端交互逻辑，包括文档加载、筛选、详情渲染与聊天请求。

### styles.css
页面视觉样式，定义布局、面板、卡片与聊天样式。

### docs.json
前端数据源，由 `build_web_docs.py` 生成，包含可浏览的文档内容与元数据。

## 数据与配置

### config.yaml
项目配置入口，包含模型、索引路径、检索策略、生成参数与 rerank 配置。

### data/
数据与评测文件存放目录，包括原始 JSONL 文档、评测查询与相关性标注。

### prompts/
提示词模板目录，用于系统提示或自定义生成策略。

### scripts/
部署脚本目录，提供 Windows/Unix 环境的依赖安装与初始化指引。
