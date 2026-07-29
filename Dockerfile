FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas o arquivo de dependências primeiro 
COPY requirements.txt .

# Instala as bibliotecas sem salvar cache
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos de configuração e o código-fonte
COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
COPY models/ models/

# Instala o projeto localmente para reconhecer o pacote 'src'
RUN pip install .

# Expõe a porta que o FastAPI vai utilizar
EXPOSE 8000

# para iniciar o servidor quando o container subir
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]