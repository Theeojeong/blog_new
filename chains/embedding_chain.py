import httpx
import torch
import openai
import os
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_CACHE_FILE

client = OpenAI(api_key=OPENAI_API_KEY)

device = "cpu"

def get_openai_embedding(text: str) -> torch.Tensor | None:
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        embedding = response.data[0].embedding
        return torch.tensor(embedding, device="cpu")
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

def update_embedding_cache(model_name, information):
    from config import EMBEDDING_CACHE_FILE
    new_embedding = get_openai_embedding(model_name + " " + information.strip() if information else model_name)
    if new_embedding is not None:
        if os.path.exists(EMBEDDING_CACHE_FILE):
            embedding_cache = torch.load(EMBEDDING_CACHE_FILE, map_location="cpu")
        else:
            embedding_cache = {}
        embedding_cache[model_name] = new_embedding
        torch.save(embedding_cache, EMBEDDING_CACHE_FILE)
        print(f"'{model_name}'에 대한 임베딩 캐시 업데이트 완료.")
    else:
        print(f"'{model_name}'에 대한 임베딩 생성 실패. 캐시 업데이트를 건너뜹니다.")