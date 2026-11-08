import ollama
import outlines
import asyncio
import time
from ollama import Client
# Create the client or async client
client = ollama.Client()
async_client = ollama.AsyncClient()

def ollama_chat_loop(prompts, model_name="llama3.1:8b", stream=False, keep_alive="5m"):
    """使用 Ollama 的同步 client.chat 对一组 prompts 顺序循环调用。

    返回每个 prompt 的文本输出列表。
    """
    client = Client()
    outputs = []
    for prompt in prompts:
        if stream:
            parts = []
            for chunk in client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"keep_alive": keep_alive},
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    parts.append(chunk["message"]["content"])
            outputs.append("".join(parts))
        else:
            resp = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"keep_alive": keep_alive},
            )
            outputs.append(resp["message"]["content"])
    return outputs

# Create a sync model
model = outlines.from_ollama(
    client,
    "llama3.1:8b",
)
response = model("1+1=?")
print(response)
# Create an async model

async def async_test():
    model = outlines.from_ollama(
        async_client,
        "llama3.1:8b",
    )

    prompts = [f"1+{i}=?" for i in range(2, 1000)]
    start = time.time()
    results = await asyncio.gather(*[model(p) for p in prompts])
    elapsed = time.time() - start
    #print(results)
    print(f"Time taken: {elapsed:.6f} seconds")


    start = time.time()
    results = []
    for p in prompts:
        results.append(await model(p))
    elapsed = time.time() - start
    #print(results)
    print(f"Time taken: {elapsed:.6f} seconds")
    
    start = time.time()
    _ = ollama_chat_loop(prompts, model_name="llama3.1:8b", stream=False, keep_alive="5m")
    elapsed = time.time() - start
    print(f"Time taken (ollama chat loop): {elapsed:.6f} seconds")

if __name__ == "__main__":
    asyncio.run(async_test())