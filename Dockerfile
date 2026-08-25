FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dxf_writer.py recon_service.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "recon_service.py"]
