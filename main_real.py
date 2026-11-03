#!/usr/bin/env python3
"""
小桃养成记 AI 客服
三级降级: llama-cpp-python → llama-server 二进制 → 规则引擎
"""
import os, json, time, urllib.request, threading, subprocess, sys, tarfile, io, zipfile
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="小桃养成记 AI 客服")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CHAT_DIR = Path("/home/container")
MODEL_PATH = CHAT_DIR / "qwen2-0_5b-instruct-q2_k.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q2_k.gguf"
LLAMA_SERVER_BIN = CHAT_DIR / "llama-server"
LLAMA_SERVER_URLS = [
    "https://github.com/ggml-org/llama.cpp/releases/download/b6471/llama-b6471-bin-ubuntu-x64.zip",
    "https://github.com/ggml-org/llama.cpp/releases/download/b6471/llama-b6471-bin-ubuntu-x64.zip",
]

SYSTEM_PROMPT = (
    "你是「小桃养成记」游戏官网的AI客服小助手。"
    "游戏是一款免费养成类手机游戏，玩家可以养成一个叫小桃的角色。"
    "官网地址: https://xiaoyan-pet.xiaoyan-pan.eu.cc/ "
    "APK下载: https://xiaoyan-pet-site.pages.dev/download/xiaoyan-pet-v1.3.0-release.apk "
    "请用中文、亲切可爱的语气回答。"
    "如果不确定答案，就说'这个我不太确定，建议查看官网详情'。"
    "回答尽量简短，3-5句话以内。"
)

llm = None
llm_backend = "none"
llama_server_proc = None

def download_model():
    if MODEL_PATH.exists():
        print(f"[model] 已存在 ({MODEL_PATH.stat().st_size//1024//1024}MB)")
        return True
    print("[model] 下载中...")
    tmp = MODEL_PATH.with_suffix(".part")
    def _r(b, bs, total):
        d = b * bs
        print(f"\r[model] {min(99,d*100//total if total else 0)}% ({d//1024//1024}/{total//1024//1024}MB)", end="", flush=True)
    urllib.request.urlretrieve(MODEL_URL, tmp, reporthook=_r)
    tmp.rename(MODEL_PATH)
    print(f"\n[model] 完成 ({MODEL_PATH.stat().st_size//1024//1024}MB)")
    return True

def try_llama_cpp():
    global llm, llm_backend
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[llm] 尝试 pip install llama-cpp-python...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--prefix", ".local", "-q", "llama-cpp-python"],
            cwd=str(CHAT_DIR),
            env={**os.environ, "LD_LIBRARY_PATH": str(CHAT_DIR)}, timeout=180, capture_output=True,
        )
        sp = CHAT_DIR / ".local/lib/python3.13/site-packages"
        if sp.exists():
            sys.path.insert(0, str(sp))
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[llm] llama-cpp-python 安装失败")
            return False
    try:
        print("[llm] 加载 llama-cpp...")
        llm = Llama(model_path=str(MODEL_PATH), n_ctx=128, n_threads=2, verbose=False)
        llm_backend = "llama_cpp"
        print("[llm] llama-cpp-python 加载成功")
        return True
    except Exception as e:
        print(f"[llm] llama-cpp 加载失败: {e}")
        return False

def try_llama_server():
    global llm_backend, llama_server_proc
    # 先测试二进制能否运行
    import shutil
    if LLAMA_SERVER_BIN.exists():
        test = subprocess.run([str(LLAMA_SERVER_BIN), "--version"], capture_output=True, text=True, timeout=5)
        print(f"[llm] binary test: rc={test.returncode} out={test.stdout[:100]} err={test.stderr[:200]}")
        if test.returncode != 0:
            print("[llm] 二进制无法运行，尝试重新下载...")
            LLAMA_SERVER_BIN.unlink()
    if not LLAMA_SERVER_BIN.exists():
        for url in LLAMA_SERVER_URLS:
            try:
                print(f"[llm] 下载 llama-server: {url.split('/')[-1]}...")
                data = urllib.request.urlopen(url, timeout=60).read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    found = False
                    for name in z.namelist():
                        if not z.getinfo(name).is_dir():
                            f = z.extractfile(name)
                            if f:
                                out_path = CHAT_DIR / name.split("/")[-1]
                                out_path.write_bytes(f.read())
                                if "llama-server" in name:
                                    out_path.chmod(0o755)
                                    found = True
                                print(f"[llm] 解压 {name}")
                    if not found:
                        print(f"[llm] 没找到 llama-server")
                        continue
                    break
            except Exception as e:
                print(f"[llm] 下载失败: {e}")
        if not LLAMA_SERVER_BIN.exists():
            print("[llm] llama-server 二进制下载失败")
            return False
        print(f"[llm] 二进制 {LLAMA_SERVER_BIN.stat().st_size//1024//1024}MB")
    try:
        err_file = CHAT_DIR / "llama-server.err"
        err_fh = open(err_file, "w")
        llama_server_proc = subprocess.Popen(
            [str(LLAMA_SERVER_BIN), "-m", str(MODEL_PATH), "--port", "8081", "-c", "128", "--threads", "2"],
            stdout=subprocess.DEVNULL, stderr=err_fh,
            cwd=str(CHAT_DIR),
        )
        time.sleep(8)
        if llama_server_proc.poll() is not None:
            err_fh.close()
            err_text = err_file.read_text() if err_file.exists() else ""
            print(f"[llm] llama-server 退出 rc={llama_server_proc.returncode}: {err_text[:300]}")
            return False
        err_fh.close()
        print("[llm] llama-server 已启动 :8081")
        llm_backend = "llama_server"
        return True
    except Exception as e:
        print(f"[llm] llama-server 启动失败: {e}")
        return False

def init_llm():
    global llm_backend
    try:
        download_model()
    except Exception as e:
        print(f"[model] 下载失败: {e}")
        return
    # 先试 llama-server（更轻量，不占 Python 内存）
    if try_llama_server():
        print("[init] 使用 llama-server")
        return
    # 再试 llama-cpp（编译型，占内存多）
    if try_llama_cpp():
        print("[init] 使用 llama-cpp")
        return
    print("[llm] 所有 LLM 方案失败，使用规则引擎")

RULES = [
    (["下载", "下载链接", "apk", "安装"],
     "小桃养成记的 APK 下载在这里哦～\nhttps://xiaoyan-pet-site.pages.dev/download/xiaoyan-pet-v1.3.0-release.apk\n直接点就能下载，安装前记得允许「未知来源应用」安装权限 💕"),
    (["官网", "网站", "主页"],
     "官网在这里：https://xiaoyan-pet.xiaoyan-pan.eu.cc/\n可以看到游戏介绍、版本更新和安全检测信息哦～"),
    (["登录", "账号", "注册", "密码"],
     "小桃养成记支持邮箱登录，在游戏内设置页 → 账号里注册/登录。登录后可云同步存档，换手机也不丢进度！"),
    (["存档", "云同步", "备份"],
     "存档默认存在手机本地（Room 数据库），登录后可一键云同步到 Supabase。换手机时先登同一个账号就能恢复～"),
    (["更新", "新版本", "检查更新"],
     "游戏内有「检查更新」功能，会提示最新版本。当前最新版 v1.3.0（四季的小屋）。"),
    (["崩溃", "闪退", "打不开", "卡顿"],
     "试试这几个：① 清除游戏缓存 ② 重启手机 ③ 确认安卓版本 ≥ 7.0。如果还闪退，到官网看看有没有已知问题说明～"),
    (["角色", "小桃", "养成", "属性", "剧情"],
     "小桃有 4 项表面属性（饥饿/心情/健康/亲密）和 3 条隐藏线（信任/探索/理解）。不同互动会影响数值，决定最终结局！"),
    (["结尾", "结局", "消散", "归途", "永恒"],
     "游戏有 3 种结局：「归途」「永恒」「消散」，取决于你的选择。多互动才能看到全部结局哦！"),
    (["免费", "收费", "多少钱"],
     "小桃养成记完全免费！没有任何内购或广告，放心下载～"),
    (["安卓", "手机", "支持"],
     "目前只支持 Android 7.0+（API 24 以上）。iOS 版本还在计划中哦～"),
]

def rule_reply(msg):
    ml = msg.lower()
    for kws, reply in RULES:
        if any(k in ml for k in kws):
            return reply
    return ("我是小桃养成记的 AI 客服助手 🌸\n"
            "可以问我：下载、安装、登录、存档、更新、游戏内容、结局等。\n"
            "不确定的问题建议直接看官网：https://xiaoyan-pet.xiaoyan-pan.eu.cc/")

def llm_reply(user_msg, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_msg})
    if llm_backend == "llama_cpp" and llm:
        resp = llm(messages, max_tokens=256, temperature=0.7, top_p=0.9)
        return resp["choices"][0]["message"]["content"].strip()
    if llm_backend == "llama_server":
        body = json.dumps({"messages": messages, "max_tokens": 256, "temperature": 0.7}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8081/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    return None

def get_reply(user_msg, history):
    if llm_backend != "none":
        try:
            r = llm_reply(user_msg, history)
            if r:
                return r
        except Exception as e:
            print(f"[llm] 推理失败: {e}")
    return rule_reply(user_msg)

chat_history: list = []

@app.get("/", response_class=HTMLResponse)
async def index():
    return (CHAT_DIR / "chat.html").read_text(encoding="utf-8")

@app.get("/api/health")
async def health():
    return {"ok": True, "backend": llm_backend, "mode": llm_backend.upper()}

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return JSONResponse({"error": "empty"}, status_code=400)
    start = time.time()
    reply = get_reply(user_msg, chat_history)
    elapsed = round(time.time() - start, 2)
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": reply})
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]
    return JSONResponse({"reply": reply, "mode": llm_backend, "ms": elapsed})

if __name__ == "__main__":
    threading.Thread(target=init_llm, daemon=True).start()
    port = int(os.environ.get("SERVER_PORT", 30086))
    print(f"[app] 小桃养成记 AI 客服 → port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
