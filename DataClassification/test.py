from gensim.models import Word2Vec

# 학습 데이터를 준비합니다. 각 문장은 단어 리스트로 나누어져 있어야 합니다.
sentences = [
    ['I', 'love', 'machine', 'learning'],
    ['Machine', 'learning', 'is', 'fun'],
    ['I', 'prefer', 'deep', 'learning', 'over', 'traditional', 'methods'],
    ['Natural', 'language', 'processing', 'is', 'a', 'subset', 'of', 'AI'],
    ['Word', 'embeddings', 'help', 'us', 'analyze', 'language']
]

# Word2Vec 모델을 학습합니다.
# vector_size는 임베딩 차원 수, window는 주변 단어를 고려할 크기, min_count는 최소 등장 횟수
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# 단어 'learning'의 벡터를 출력합니다.
print("Vector for 'learning':")
print(model.wv['learning'])

# 단어 유사도를 확인합니다.
print("\nMost similar words to 'learning':")
print(model.wv.most_similar('learning'))
