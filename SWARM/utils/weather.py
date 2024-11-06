import json
import requests
import os
from swarm import Agent
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 API 키 불러오기

# OpenWeatherMap API 키 불러오기
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(location, time="now"):
    """주어진 도시의 현재 날씨 정보를 가져오는 함수."""
    if not API_KEY:
        return "API 키가 설정되지 않았습니다. .env 파일에 유효한 OpenWeatherMap API 키를 추가해 주세요."

    # OpenWeatherMap API URL 생성
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    
    try:
        # API 요청 보내기
        response = requests.get(url)
        response.raise_for_status()  # 요청에 실패하면 예외 발생
        
        # JSON 응답 파싱
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        
        return json.dumps({
            "location": location,
            "temperature": temperature,
            "description": description,
            "time": time
        })
    
    except requests.exceptions.HTTPError as http_err:
        # HTTP 에러 발생 시
        return f"HTTP 오류 발생: {http_err} (응답 코드: {response.status_code})"
    
    except requests.exceptions.RequestException as req_err:
        # 기타 요청 에러 발생 시
        return f"요청 실패: {req_err}. URL을 확인하고 네트워크 상태를 점검해 주세요."
    
    except KeyError as key_err:
        # 응답에서 필요한 키가 누락된 경우
        return f"응답 파싱 실패: {key_err}. 응답 내용: {response.text}"
    
    except Exception as err:
        # 기타 예외 발생 시
        return f"알 수 없는 오류 발생: {err}"

# Weather Agent 설정
weather_agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent that provides weather information.",
    functions=[get_weather],
)
