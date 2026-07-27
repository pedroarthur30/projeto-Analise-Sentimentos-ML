# 1. Imagem base oficial do Python (versão leve)
FROM python:3.10-slim

# 2. Define o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copia apenas o arquivo de dependências primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# 4. Instala as bibliotecas sem salvar cache temporário para deixar a imagem menor
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia os arquivos de configuração e o código-fonte
COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
COPY models/ models/

# 6. Instala o projeto localmente para reconhecer o pacote 'src'
RUN pip install .

# 7. Expõe a porta que o FastAPI vai utilizar
EXPOSE 8000

# 8. Comando para iniciar o servidor quando o container subir
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]