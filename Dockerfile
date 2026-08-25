FROM python:3.12-slim
WORKDIR /app
COPY dxf_writer.py recon_service.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "recon_service.py"]
