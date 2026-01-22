# UCC 文档助手

基于检索增强生成（RAG）的文档问答与浏览工具，包含索引构建、离线评测、稳定性检测与 Web 前端。

## 功能概览

- 构建 FAISS 向量索引，支持 JSONL 逐行载入与通用文本加载。
- 基于检索结果生成回答，支持关键词加权与可选 rerank。
- 离线评测检索效果，输出指标与诊断报告。
- 稳定性评测，统计多次运行的一致性与答案相似度。
- 提供 Web 页面浏览文档与对话式问答体验。

## 目录结构

```
.
├── 01_index.py           # 构建向量索引
├── 02_chat.py            # CLI 对话入口
├── 03_eval.py            # 离线检索评测
├── 04_stability.py       # 稳定性评测
├── build_web_docs.py     # 生成 web/docs.json
├── config.yaml           # 项目配置
├── data/                 # 数据集与评测数据
├── prompts/              # 提示词模板
├── scripts/              # 部署脚本
├── src/                  # 核心逻辑模块
├── web/                  # 前端静态资源
└── web_app.py            # Web 服务入口
```

## 快速开始

### 1. 安装依赖

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 也可以直接执行：

```shell
.\scripts\deploy.ps1
```

### 2. 配置文件

根据实际环境编辑 `config.yaml`，包含模型、索引路径、检索与生成参数等配置。

### 3. 构建索引

```shell
python 01_index.py
```

### 4. CLI 对话

```shell
python 02_chat.py
```

### 5. 离线评测

```shell
python 03_eval.py
```

### 6. 稳定性评测

```shell
python 04_stability.py
```

### 7. 启动 Web 服务

```shell
python web_app.py
```

浏览器访问：`http://localhost:8000/`。

## 架构说明

详见 `docs/architecture.md`，包含树状结构与各模块职责说明。
