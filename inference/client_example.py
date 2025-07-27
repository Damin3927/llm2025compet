from openai import OpenAI


def create_client():
    return OpenAI(
        base_url="http://localhost:8000/v1", # RANK0 の VLLM サーバーの URL （RANK0 と一緒のノードで実行すると localhost で OK）
        api_key="token-abc123",
    )

def main():
    client = create_client()

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528", # 使用するモデル名
        messages=messages,
        temperature=0.7,
    )

    print(completion.choices[0].message.content)

if __name__ == "__main__":
    main()