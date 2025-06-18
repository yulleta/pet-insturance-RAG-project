# app.py

import streamlit as st
import openai
import pinecone
import json
import pandas as pd
import re
from typing import List, Dict, Tuple
from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI
import json

# ─── 환경설정 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="보험 비교 RAG", layout="wide")

# OpenAI, Pinecone API 키
# openai.api_key = OPENAI_API

# pc = pinecone.Pinecone(api_key=PINECONE_API)
index = pc.Index("pet-insurance-chunks")

EMBED_MODEL = "text-embedding-3-large"
# client = OpenAI(api_key=OPENAI_API)


# ─── 유틸 함수들 ────────────────────────────────────────────────────────────
W_VEC, W_CAT, W_KEY = 0.6, 0.2, 0.2

def classify_question(question: str) -> str:
    prompt = (
        f"질문: \"{question}\"\n"
        "위 질문이 다음 중 어느 유형인지 하나만 골라서 정확하게 한 단어로 알려주세요:\n"
        "보장_범위\n보장_조건\n보장_절차\n"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def compute_keyword_score(keywords: List[str], question: str) -> float:
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", question.lower())
    matched = sum(1 for kw in keywords if any(kw.lower() in token for token in tokens))
    return matched / len(keywords) if keywords else 0.0

def embed_text(text: str) -> List[float]:
    """텍스트를 OpenAI 임베딩 벡터로 변환"""
    resp = openai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def retrieve_top_chunks_by_company(
    question: str,
    top_k_per_company: int = 10,
    search_k: int = 100
) -> Tuple[Dict[str, List[Dict]], str]:
    q_type = classify_question(question)
    
    print(q_type)
    
    q_vec = embed_text(question)

    resp = index.query(
        vector=q_vec,
        top_k=search_k,
        include_metadata=True
    )

    company_buckets: Dict[str, List[Tuple[float, Dict]]] = {}

    for match in resp.matches:
        md = match.metadata
        company = md.get("company", "unknown")
        vec_score = match.score
        cat_score = md.get(q_type, 0.0)
        key_score = compute_keyword_score(md["top_keywords"], question)

        combined = W_VEC * vec_score + W_CAT * cat_score + W_KEY * key_score
        company_buckets.setdefault(company, []).append((combined, md))

    top_chunks_by_company: Dict[str, List[Dict]] = {}

    for comp, items in company_buckets.items():
        items.sort(key=lambda x: x[0], reverse=True)
        top_chunks_by_company[comp] = [md for _, md in items[:top_k_per_company]]

    return top_chunks_by_company, q_type

def generate_comparative_answer(question: str) -> str:
    company_chunks, q_type = retrieve_top_chunks_by_company(question)

    # 🔹 회사별 문서 요약 정리
    company_sections = []
    for company, chunks in company_chunks.items():
        summaries = "\n".join([f"- {md['raw_sentences']}" for md in chunks])
        section = f"**{company}**\n{summaries}"
        company_sections.append(section)

    context = "\n\n".join(company_sections)

    # 🔹 프롬프트 구성
    prompt = (
        f"[질문]\n{question}\n\n"
        f"[약관 요약 - 질의유형: {q_type}]\n{context}\n\n"
        "위 약관 내용들을 참고하여 각 보험사(삼성화재, 메리츠화재, 한화보험 3사)의 "
        "보장 여부나 조건을 정리하고 비교해주세요. "
        "회사의 이름을 명확히 구분해서 요약해 주세요."
        "답변은 마크다운 형식의 표로 나타내어 비교해주세요. 각 회사 3사가 header이고 row의 각 행 내용은 알아서 잘 채워주세요."
        "표 하단에는 마크다운 형식으로 줄글로 각 회사를 총 정리 비교하여 가장 좋은 보험사를 추천해주세요."
    )

    # 🔹 GPT 호출 (한 번만)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()

# ✅ JSON 키워드 로드
@st.cache_data
def load_keywords(json_path='cluster_keywords.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 클러스터 구분 없이 하나의 리스트로 합치기
    keyword_set = set()
    for keyword_list in data.values():
        if isinstance(keyword_list, list):
            for kw_group in keyword_list:
                if isinstance(kw_group, str):
                    for kw in kw_group.split(','):
                        keyword_set.add(kw.strip())
    return sorted(keyword_set)

def get_recommendations(input_text, keyword_list, max_suggestions=10):
    if not input_text:
        return []

    input_chars = set(input_text.lower())

    scored = []
    for kw in keyword_list:
        overlap = len(input_chars & set(kw.lower()))
        if overlap > 2:
            scored.append((kw, overlap))

    # 겹치는 수 기준 내림차순 정렬
    scored.sort(key=lambda x: -x[1])

    return [kw for kw, _ in scored[:max_suggestions]]



# ─── Streamlit UI ─────────────────────────────────────────────────────────
st.markdown("## 📌 반려동물 보험 비교 질문 Q&A")

# 1. 질문 입력
question = st.text_input(
    "보험 관련 질문을 입력하세요:",
    value="고령 반려견(예: 10살 이상)은 보험 가입이 불가능한가요?"
)


# 2. 키워드 추천 (동시에 입력된 질문 기준)
keywords = load_keywords()
# 🔍 추천 키워드 (실시간 반응)
matched_keywords = get_recommendations(question, keywords)
if matched_keywords:
    st.markdown("#### 🔍 추천 키워드")
    st.markdown(", ".join(f"`{kw}`" for kw in matched_keywords))

# 1. 쉼표로 구분된 추천 키워드 문자열
keyword_str = ", ".join(matched_keywords)

# 2. 질문 + 키워드 문자열 묶어서 GPT에 전달
full_query = f"[질문]\n{question}\n\n[관련 키워드]\n{keyword_str}"

# 3. GPT 호출
if st.button("비교하기"):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("검색 및 비교 생성 중…"):
            try:
                answer = generate_comparative_answer(full_query)
                st.markdown("### 🧾 비교 결과")
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
# python -m streamlit run app.py