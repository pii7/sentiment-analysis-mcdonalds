# Gunakan Python resmi
FROM python:3.10-slim

# Tampilkan log Python real-time
ENV PYTHONUNBUFFERED=1

# Buat working directory
WORKDIR /app

# Copy daftar library
COPY requirements.txt .

# Install library
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Command utama: jalankan script ML
CMD ["python", "main.py"]