import streamlit as st
if "GEMINI_API_KEY" in st.secrets:
    api_key= st.secrets["GEMINI_API_KEY"]

from google import genai
client= genai.Client(api_key=api_key)

import pandas as pd
import requests
from google.genai import types



@st.cache_data(ttl=3600)
def fetch_cherry_blossom_data():
    url="https://www.weather.go.kr/w/theme/seasonal-observation/spring-flower.do"
    headers={'User-Agent': 'Mozilla/5.0'}

    try:
        response=requests.get(url, headers=headers)
        table_list= pd.read_html(response.text)
        df= table_list[0]

        if not df.empty and len(df)> 10:
            return "2026 실시간 데이터", df.to_string(index=False)
        else:
            raise ValueError("업데이트 전")

    except Exception:
        last_year_info= '''
        [2025년 주요도시 실제 개화일 정도]
        - 제주: 3월 22일 (만개: 3월 29일)
        - 부산: 3월 25일 (만개: 4월 1일)
        - 여수/진해: 3월 26일 (만개: 4월 2일)
        - 광주/전주: 3월 28일 (만개: 4월 4일)
        - 대전/청주: 3월 31일 (만개: 4월 7일)
        - 서울: 4월 1일 (만개: 4월 8일)
        - 강릉: 3월 30일 (만개: 4월 6일)
        - 인천/수원: 4월 4일 (만개: 4월 11일)
        - 춘천: 4월 7일 (만개: 4월 14일)
        '''
        return "2025년 실제 데이터", last_year_info

def get_ai_response(question):
    data_type, blossom_info= fetch_cherry_blossom_data()
    
    config= types.GenerateContentConfig(
    max_output_tokens=10000,
    response_mime_type='text/plain',
    system_instruction = f"""
    당신은 2026년 벚꽃 개화 시기를 안내하는 전문 챗봇 ' 벗꽃 모니터' 입니다.
    [참조 데이터 ({data_type})]
    {blossom_info}

    [핵심 지침]
    1. 지금은 2026년 2월이며, 기상청 공식 발표 전입니다. 반드시 "아직 공식 예보 전이라 작년(2025년) 실제 개화 데이터를 바탕으로 여행 계획을 도와드릴게요"라고 안내하세요.
    2. 예를 들어 서울 질문이 오면 "작년에는 서울 기준 4월 1일에 꽃이 피기 시작해서 4월 8일경에 만개했어요. 올해도 이 시기를 전후로 여행 계획을 세우시면 좋을 것 같아요"라고 구체적인 가이드를 주세요.
    3. 보통 개화 후 7일 뒤가 만개(가장 예쁠 때)라는 점을 강조하세요.
    4. 나들이하기 좋은 명소(여의도 윤중로, 석촌호수, 진해 군항제 등)를 추천하고 친절하고 설레는 말투로 대답하세요.🌸
    """
    )
    response= client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=config
        )
    return response.text

st.set_page_config(
    page_title="벚꽃 모니터",
    page_icon='./logo/image.png',
    layout="wide"
    
    )
with st.sidebar:
    st.title("벚꽃 명당 추천")
    st.info('지역별로 가장 유명한 명소를 확인해보세요')
    with st.expander("수도권"):
        st.write("**여의도 윤중로**: 말이 필요 없는 성지")
        st.write("**석촌호수**: 호수와 롯데월드 뷰")
        st.write("**경희대 본관**: 고전적인 건물의 조화")
    
    with st.expander("경상/부산"):
        st.write("**진해 여좌천**: 로망스 다리 벚꽃 터널")
        st.write("**경주 보문단지**: 고즈넉한 벚꽃길")
        st.write("**부산 온천천**: 도심 속 분홍 물결")
        
    with st.expander("충청/전라/제주"):
        st.write("**제주 전농로**: 왕벚꽃 드라이브")
        st.write("**구례 섬진강**: 강변 따라 펼쳐진 꽃길")
        st.write("**공주 계룡산**: 산과 꽃의 아름다운 조화")

col1, col2= st.columns([1,5])
with col1:
    st.image('./logo/image.png')

with col2:
    st.markdown(
        '''
        <h1 style='margin-bottom:0;'>벚꽃 모니터</h1>
        <p>2026년 벚꽃 소식과 전국 명소를 한눈에 확인하세요.</p>
        ''',
        unsafe_allow_html=True
    )

st.markdown("---")
if "messages" not in st.session_state:
    st.session_state.messages=[
        {'role':'assistant','content':'🌸안녕하세요! 2026년 벚꽃 모니터입니다.🌸 '},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

question= st.chat_input('질문을 입력하세요')
if question:
    question=question.replace('\n','  \n')
    st.session_state.messages.append({'role':'user','content':question})
    st.chat_message('user').write(question)

    with st.spinner('꽃 소식을 가져오는 중..'):
        response=get_ai_response(question)
        st.session_state.messages.append({'role':'assistant','content':response})
        st.chat_message('assistant').write(response)
