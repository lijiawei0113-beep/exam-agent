# 云端部署用（阶段六后半）。本地使用不需要 Docker：python -m exam_agent web
FROM python:3.12-slim
# 中文 PDF 需要中文字体
RUN apt-get update && apt-get install -y --no-install-recommends fonts-wqy-zenhei && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY kb ./kb
RUN pip install --no-cache-dir -e .
ENV EXAM_DB_PATH=/data/exam.db EXAM_EMBEDDER=none
EXPOSE 8000
CMD ["python", "-m", "exam_agent", "web", "--host", "0.0.0.0", "--port", "8000"]
