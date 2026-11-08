import dashscope
from http import HTTPStatus
import os
import requests
import json
# 避免在代码中硬编码密钥，从环境变量读取
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

def text_rerank():
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query="什么是文本排序模型",
        documents=[
            "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
            "量子计算是计算科学的一个前沿领域",
            "预训练语言模型的发展给文本排序模型带来了新的进展"
        ],
        top_n=10,
        return_documents=True
    )
    if resp.status_code == HTTPStatus.OK:
        print(resp)
    else:
        print(resp)

def jinja():
    
    from jinja2 import Template
    template_src = """<上面模板内容之一>"""
    template = Template(template_src)

    rows = [["1","2","3"], ["4","5","6"]]
    output = template.render(rows=rows)
print(output)
def dify_test():
    # 默认走网关 2001 端口下的 /v1/chat-messages（如需 /api 前缀，请用环境变量覆盖）
    API_URL = os.getenv("DIFY_API_URL", "http://10.113.178.150:2001/v1/chat-messages")
    API_KEY = os.getenv("DIFY_API_KEY", "")

    payload = {
        "inputs": {},
        "query": "What are the specs of the iPhone 13 Pro Max?",
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "abc-123",
        "files": [
            {
                "type": "image",
                "transfer_method": "remote_url",
                "url": "https://cloud.dify.ai/logo/logo-site.png"
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json;q=0.9"
    }

    # 如果你想同步等待完整响应（非流式），可以直接使用 requests.post(...)
    # r = requests.post(API_URL, headers=headers, json=payload)
    # print(r.status_code, r.text)

    # 发起请求并流式处理
    with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=30) as r:
        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            print("ERROR: 返回了 HTML 页面，可能访问了控制台路由或未走公开 API。status=", r.status_code)
            print("HINT: 请确认使用 /api/v1/chat-messages，并确保反向代理将 /api 转发到 api 服务，且令牌为 app 级 token。")
            print("HTML SNIPPET:", r.text[:500])
            return

        r.raise_for_status()

        if "text/event-stream" in content_type:
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                line = chunk.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                try:
                    data = json.loads(line)
                    print("RECEIVED:", json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError:
                    print("NON-JSON CHUNK:", line)
        else:
            try:
                data = r.json()
                print("RECEIVED:", json.dumps(data, ensure_ascii=False))
            except ValueError:
                print("WARN: 非 JSON 响应，content-type=", content_type)
                print(r.text[:1000])

def dify_stream_from_curl():
    """
    使用 Python 复现可执行的 curl，请求 Dify /api/v1/chat-messages，流式打印返回。
    需设置环境变量：
      - DIFY_API_URL，例如: http://10.113.178.150:2001/api/v1/chat-messages
      - DIFY_API_KEY，例如: app-xxxxx
    """
    api_url = os.getenv("DIFY_API_URL", "http://10.113.178.150:2001/v1/chat-messages")
    api_key = os.getenv("DIFY_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json;q=0.9"
    }

    payload = {
        "inputs": {},
        "query": "What are the specs of the iPhone 13 Pro Max?",
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "abc-123",
        "files": [
            {
                "type": "image",
                "transfer_method": "remote_url",
                "url": "https://cloud.dify.ai/logo/logo-site.png"
            }
        ]
    }

    with requests.post(api_url, headers=headers, json=payload, stream=True, timeout=60) as r:
        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            print("ERROR: 返回了 HTML 页面，检查是否命中控制台而非 /api。status=", r.status_code)
            print(r.text[:500])
            return

        r.raise_for_status()

        # SSE 流式
        if "text/event-stream" in content_type:
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                line = chunk.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                try:
                    obj = json.loads(line)
                    # 仅输出 answer 字段内容，实时刷新
                    event_type = obj.get("event")
                    if "answer" in obj:
                        print(obj["answer"], end="", flush=True)
                    # 在消息结束时换行
                    if event_type == "message_end":
                        print("")
                except json.JSONDecodeError:
                    # 非 JSON 片段直接忽略或按需打印
                    pass
        else:
            # 非 SSE 情况：整体打印 JSON
            try:
                obj = r.json()
                if isinstance(obj, dict) and "answer" in obj:
                    print(obj["answer"])
                else:
                    print(json.dumps(obj, ensure_ascii=False))
            except ValueError:
                print(r.text[:2000])

if __name__ == '__main__':
    # 演示：等价于 curl 的流式请求
    dify_stream_from_curl()