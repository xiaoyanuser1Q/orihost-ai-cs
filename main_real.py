#!/usr/bin/env python3
"""
小桃养成记 AI 客服
- 优先使用本地 LLM (llama-cpp + Qwen2-0.5B)
- 自动降级到规则引擎 (模型不可用时)
"""
import os
import json
import time
import urllib.request
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="小桃养成记 AI 客服")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CHAT_DIR = Path("/home/container")
MODEL_PATH = CHAT_DIR / "qwen2-0.5b-instruct-q4_k_m.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0.5b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "你是「小桃养成记」游戏官网的AI客服小助手。"
    "游戏是一款免费养成类手机游戏，玩家可以养成一个叫小桃的角色。"
    "官网地址: https://xiaoyan-pet.xiaoyan-pan.eu.cc/ "
    "APK下载: https://xiaoyan-pet-site.pages.dev/download/xiaoyan-pet-v1.3.0-release.apk "
    "请用中文、亲切可爱的语气回答。"
    "如果不确定答案，就说'这个我不太确定，建议查看官网详情'。"
    "回答尽量简短，3-5句话以内。"
)

# ── 模型加载 ──────────────────────────────────────────────
llm = None
llm_ready = False

def download_model():
    """下载 Qwen2-0.5B Q4 模型（~300MB）"""
    if MODEL_PATH.exists():
        print(f"[model] 已存在: {MODEL_PATH} ({MODEL_PATH.stat().st_size//1024//1024}MB)")
        return
    print(f"[model] 下载 {MODEL_URL} ...")
    tmp = MODEL_PATH.with_suffix(".part")
    def _report(blocks, bs, total):
        done = blocks * bs
        pct = min(99, done * 100 // total) if total else 0
        print(f"\r[model] {pct}% ({done//1024//1024}/{total//1024//1024} MB)", end="", flush=True)
    urllib.request.urlretrieve(MODEL_URL, tmp, reporthook=_report)
    tmp.rename(MODEL_PATH)
    print(f"\n[model] 下载完成: {MODEL_PATH.stat().st_size//1024//1024}MB")

def init_llm():
    global llm, llm_ready
    try:
        download_model()
        from llama_cpp import Llama
        print("[llm] 加载模型...")
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=512,
            n_threads=2,
            verbose=False,
        )
        llm_ready = True
        print("[llm] 模型加载成功 ✅")
    except Exception as e:
        print(f"[llm] 模型加载失败，降级到规则引擎: {e}")
        llm_ready = False

# ── 规则引擎（降级方案）──────────────────────────────────
RULES = [
    (["下载", "下载链接", "apk", "安装"],
     "小桃养成记的 APK 下载在这里哦～\nhttps://xiaoyan-pet-site.pages.dev/download/xiaoyan-pet-v1.3.0-release.apk\n直接点就能下载，安装前记得允许「未知来源应用」安装权限 💕"),
    (["官网", "网站", "主页"],
     "官网在这里：https://xiaoyan-pet.xiaoyan-pan.eu.cc/\n可以看到游戏介绍、版本更新和安全检测信息哦～"),
    (["登录", "账号", "注册", "密码"],
     "小桃养成记支持邮箱登录，在游戏内设置页 → 账号里注册/登录。"
     "登录后可云同步存档，换手机也不丢进度！"),
    (["存档", "云同步", "备份"],
     "存档默认存在手机本地（Room 数据库），登录后可一键云同步到 Supabase。"
     "换手机时先登同一个账号就能恢复～"),
    (["更新", "新版本", "检查更新"],
     "游戏内有「检查更新」功能，会提示最新版本。"
     "也可以在官网查看最新版本下载。当前最新版 v1.3.0（四季的小屋）。"),
    (["崩溃", "闪退", "打不开", "卡顿"],
     "试试这几个：① 清除游戏缓存 ② 重启手机 ③ 确认安卓版本 ≥ 7.0。"
     "如果还闪退，到官网看看有没有已知问题说明～"),
    (["角色", "小桃", "养成", "属性", "剧情"],
     "小桃有 4 项表面属性（饥饿/心情/健康/亲密）和 3 条隐藏线（信任/探索/理解）。"
     "不同互动会影响数值，决定最终结局！当前是 v1.3.0 四季的小屋～"),
    (["结尾", "结局", "消散", "归途", "永恒"],
     "游戏有 3 种结局：「归途」「永恒」「消散」，取决于你的选择。"
     "不同剧情分支会影响隐藏属性，多互动才能看到全部结局哦！"),
    (["免费", "收费", "多少钱"],
     "小桃养成记完全免费！没有任何内购或广告，放心下载～"),
    (["安卓", "手机", "支持"],
     "目前只支持 Android 7.0+（API 24 以上）。iOS 版本还在计划中哦～"),
]

def rule_reply(msg: str) -> str:
    msg_lower = msg.lower()
    for keywords, reply in RULES:
        if any(kw in msg_lower for kw in keywords):
            return reply
    return ("我是小桃养成记的 AI 客服助手 🌸\n"
            "可以问我：下载、安装、登录、存档、更新、游戏内容、结局等。\n"
            "不确定的问题建议直接看官网：https://xiaoyan-pet.xiaoyan-pan.eu.cc/")

def get_reply(user_msg: str, history: list[dict]) -> str:
    if llm_ready:
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history[-6:])
            messages.append({"role": "user", "content": user_msg})
            resp = llm(
                messages,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[llm] 推理失败，降级规则: {e}")
    return rule_reply(user_msg)

# ── 聊天历史（内存）─────────────────────────────────────
chat_history: list[dict] = []

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (CHAT_DIR / "chat.html").read_text(encoding="utf-8")
    return html

@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "llm": llm_ready,
        "model": "qwen2-0.5b-instruct-q4" if llm_ready else "rule-based (fallback)",
        "mode": "LLM" if llm_ready else "RULES",
    }

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return JSONResponse({"error": "message is empty"}, status_code=400)

    start = time.time()
    reply = get_reply(user_msg, chat_history)
    elapsed = round(time.time() - start, 2)

    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": reply})
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    return JSONResponse({
        "reply": reply,
        "mode": "llm" if llm_ready else "rules",
        "ms": elapsed,
    })

if __name__ == "__main__":
    threading.Thread(target=init_llm, daemon=True).start()
    print("[app] 小桃养成记 AI 客服启动 http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
