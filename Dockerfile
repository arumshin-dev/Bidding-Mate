# 1. 파이썬 3.12 슬림 버전 사용
FROM python:3.12-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치
# (에러 원인이었던 software-properties-common 제거함)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 라이브러리 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트의 모든 파일 복사
COPY . .

# 6. 스트림릿 포트(8501) 열기
EXPOSE 8501

# 7. 실행 명령어
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]