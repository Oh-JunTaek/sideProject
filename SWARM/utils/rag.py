import faiss
import numpy as np
import pickle
from swarm import Swarm, Agent
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

# Initialize OpenAI and Swarm clients
client = OpenAI()
swarm_client = Swarm()

# FAISS 인덱스와 임베딩 데이터 로드
def load_faiss_index():
    # FAISS 인덱스 로드
    index = faiss.read_index("data/index.faiss")

    # pickle 파일에서 문서와 임베딩 데이터 로드
    with open("data/index.pkl", "rb") as f:
        doc_embeddings = pickle.load(f)

    return index, doc_embeddings

# 로드된 인덱스와 임베딩 데이터 설정
index, doc_embeddings = load_faiss_index()

def embed_query(query):
    # Using OpenAI to create embeddings for the query
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    return response.data[0].embedding

# 문서 검색 함수
def retrieve_information(query, k=5):
    query_embedding = embed_query(query)  # 실제 임베딩으로 쿼리 변환
    query_embedding = np.array(query_embedding).astype('float32')  # numpy 배열로 변환

    # FAISS에서 k개의 유사한 결과 검색
    D, I = index.search(np.array([query_embedding]), k)  # 2D 배열로 변환하여 검색

    # 검색 결과가 없을 경우 처리
    if I is None or len(I) == 0 or I[0][0] == -1:
        print("검색 결과가 없습니다.")  # 디버깅 메시지 추가
        return []

    # 검색 결과 반환, -1 인덱스 제외
    results = [doc_embeddings[i] for i in I[0] if i != -1]
    return results

# 검색 결과를 바탕으로 응답 생성 함수
def generate_response(query):
    search_results = retrieve_information(query)
    if not search_results:
        return "관련 정보를 찾을 수 없습니다. 다른 질문을 시도해 주세요."

    # 검색된 정보를 포함하여 응답 생성
    response_content = "\n\n".join(search_results)
    response = swarm_client.run(
        agent=rag_agent,
        messages=[
            {"role": "system", "content": "검색된 정보를 바탕으로 사용자의 질문에 답변해 주세요."},
            {"role": "user", "content": f"질문: {query}"},
            {"role": "assistant", "content": f"검색 결과: {response_content}"}
        ]
    )
    return response.messages[-1].content

# RAG 에이전트 설정
rag_agent = Agent(
    name="RAG Agent",
    instructions="사용자의 질문에 대해 검색된 정보를 기반으로 답변을 생성해 주세요.",
    functions=[generate_response],
)