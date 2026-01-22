from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, make_response, request, send_from_directory

from src.chat_engine import ChatEngine
from src.chat_memory import MemoryManager


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


def create_app() -> Flask:
    """创建并配置 Flask 应用。"""
    # 初始化应用与静态资源路径
    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/web")

    # 加载聊天引擎与内存管理
    engine = ChatEngine.from_yaml(str(BASE_DIR / "config.yaml"))
    memory = MemoryManager.in_memory()

    @app.get("/")
    def index():
        """首页入口。"""
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/health")
    def health():
        """健康检查与内存统计。"""
        stats = memory.stats()
        return jsonify(
            {
                "status": "ok",
                "sessions": stats.get("sessions"),
                "ttl_seconds": stats.get("ttl_seconds"),
                "max_messages": stats.get("max_messages"),
            }
        )

    @app.post("/api/chat")
    def chat():
        """聊天接口，处理用户问题并返回答案。"""
        # 解析输入参数
        payload: Dict[str, str] = request.get_json(force=True) or {}
        message = (payload.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400

        # 获取或生成会话 ID
        session_id = request.cookies.get("session_id")
        if not session_id:
            session_id = uuid.uuid4().hex

        # 获取历史并生成回答
        history = memory.get_history(session_id)
        response = engine.answer(message, history=history)
        memory.append_turn(session_id, message, response.answer)

        # 组装响应结构
        resp = make_response(
            jsonify(
                {
                    "answer": response.answer,
                    "docs": [
                        {
                            "source": doc.metadata.get("source", ""),
                            "id": doc.metadata.get("id", ""),
                            "title": doc.metadata.get("title", ""),
                            "section": doc.metadata.get("section", ""),
                        }
                        for doc in response.docs
                    ],
                    "metrics": (
                        {
                            "elapsed": response.metrics.elapsed,
                            "total_tokens": response.metrics.total_tokens,
                            "tokens_per_s": response.metrics.tokens_per_s,
                        }
                        if response.metrics
                        else None
                    ),
                }
            )
        )
        # 设置会话 Cookie
        resp.set_cookie("session_id", session_id, max_age=1800, httponly=True, samesite="Lax")
        return resp

    return app


if __name__ == "__main__":
    # 本地启动开发服务
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=False)
