# 🧠 Análise de Sentimentos com Machine Learning

API de Machine Learning que classifica reviews de e-commerce em português como **positivos** ou **negativos**, construída com **Scikit-Learn** e servida via **FastAPI**.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação e Configuração](#-instalação-e-configuração)
- [Dataset](#-dataset)
- [Treinamento do Modelo](#-treinamento-do-modelo)
- [Executando a API](#-executando-a-api)
- [Endpoints da API](#-endpoints-da-api)
- [Execução com Docker](#-execução-com-docker)
- [Notebooks de Análise](#-notebooks-de-análise)
- [Variáveis de Ambiente](#-variáveis-de-ambiente)

---

## 🔍 Visão Geral

O projeto realiza análise de sentimentos em textos de reviews de e-commerce utilizando um pipeline de NLP clássico:

1. **Pré-processamento** — Remoção de acentos, pontuação, normalização para minúsculas.
2. **Vetorização** — Transformação do texto em features numéricas com **TF-IDF**.
3. **Classificação** — **Regressão Logística** otimizada via **GridSearchCV** com validação cruzada (5-fold).
4. **Serving** — API REST com **FastAPI** para inferência em tempo real.

---

## 🛠 Tecnologias Utilizadas

| Categoria        | Tecnologia                                    |
| ---------------- | --------------------------------------------- |
| Linguagem        | Python 3.10                                   |
| Machine Learning | Scikit-Learn, TF-IDF, Logistic Regression     |
| API              | FastAPI, Uvicorn                              |
| Dados            | Pandas, NumPy                                 |
| Validação        | Pydantic, Pydantic-Settings                   |
| Visualização     | Matplotlib, Seaborn                           |
| NLP              | NLTK, Unicodedata, Regex                      |
| Containerização  | Docker                                        |
| Serialização     | Joblib                                        |

---

## 📁 Estrutura do Projeto

```
projeto_AnaliseSentimentos_ML/
│
├── app/
│   └── main.py                 # Aplicação FastAPI (rotas e endpoints)
│
├── src/
│   ├── __init__.py             # Marca src como pacote Python
│   ├── config.py               # Configurações centralizadas (caminhos, .env)
│   ├── data_processing.py      # Carregamento e limpeza dos dados
│   ├── predict.py              # Carregamento do modelo e inferência
│   └── train.py                # Pipeline de treinamento com GridSearchCV
│
├── data/
│   ├── raw/                    # Dataset original (dataset.csv)
│   └── processed/              # Dados processados
│
├── models/
│   └── modelo_sentimento_v1.joblib  # Modelo treinado serializado
│
├── notebooks/
│   ├── 01-analise-exploratoria.ipynb   # EDA do dataset
│   └── 02-avaliacao-modelo.ipynb       # Métricas e avaliação do modelo
│
├── .dockerignore               # Arquivos ignorados no build Docker
├── .env.example                # Exemplo de variáveis de ambiente
├── .gitignore                  # Arquivos ignorados pelo Git
├── Dockerfile                  # Configuração do container Docker
├── pyproject.toml              # Configuração do pacote Python
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação do projeto
```

---

## ✅ Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- [Python 3.10+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/) (gerenciador de pacotes do Python)
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) *(opcional, para execução em container)*

---

## 🚀 Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/pedroarthur30/projeto-Analise-Sentimentos-ML.git
cd projeto_AnaliseSentimentos_ML
```

### 2. Crie e ative um ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate        
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Instale o projeto como pacote local

Esse passo é **necessário** para que o Python reconheça o pacote `src` nos imports:

```bash
pip install -e .
```

### 5. Configure as variáveis de ambiente

Copie o arquivo de exemplo e ajuste se necessário:

```bash
cp .env.example .env
```

O arquivo `.env` contém os caminhos do modelo e do dataset:

```env
MODEL_PATH=models/modelo_sentimento_v1.joblib
DATA_PATH=data/raw/dataset.csv
```

> **Nota:** Os valores padrão já estão configurados em `src/config.py`. O arquivo `.env` é opcional e serve para sobrescrever os caminhos caso necessário.

---

## 📊 Dataset

> **⚠️ Importante:** O diretório `data/raw/` está no `.gitignore` e **não é versionado** no repositório. Para utilizar o projeto, baixe o dataset no link abaixo e salve-o como `data/raw/dataset.csv`:
>
> 📥 **[Download do Dataset](https://drive.google.com/file/d/11zFHKWZTHbCFpDzFzYHIH3zUrOU_U8lo/view?usp=sharing)**

O dataset (`data/raw/dataset.csv`) contém **500 reviews** de e-commerce em português com as seguintes colunas:

| Coluna         | Descrição                                |
| -------------- | ---------------------------------------- |
| `review_id`    | Identificador único do review            |
| `texto_review` | Texto do review do produto               |
| `sentimento`   | Label do sentimento (`positivo`/`negativo`) |


---

## 🏋️ Treinamento do Modelo

Para treinar (ou re-treinar) o modelo a partir do dataset:

```bash
python -m src.train
```

O que esse comando faz:

1. Carrega o dataset de `data/raw/dataset.csv`
2. Aplica limpeza de texto (remoção de acentos, pontuação, normalização)
3. Divide os dados em treino (75%) e teste (25%) com estratificação
4. Cria um **Pipeline** com: `TF-IDF → StandardScaler → Logistic Regression`
5. Otimiza hiperparâmetros com **GridSearchCV** 
6. Salva o melhor modelo em `models/modelo_sentimento_v1.joblib`

---

## ▶️ Executando a API

Inicie o servidor localmente com Uvicorn:

```bash
uvicorn app.main:app --reload
```

A API estará disponível em: **http://127.0.0.1:8000**

A documentação interativa (Swagger UI) pode ser acessada em: **http://127.0.0.1:8000/docs**

---

## 📡 Endpoints da API

### `GET /` — Healthcheck

Verifica se a API está online.

```bash
curl http://127.0.0.1:8000/
```

**Resposta:**

```json
{
  "status": "online",
  "mensagem": "API de Análise de Sentimentos funcionando!"
}
```

### `POST /predict` — Classificar Sentimento

Recebe um texto de review e retorna a classificação de sentimento.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Produto excelente! Chegou rápido e funciona perfeitamente."}'
```

**Corpo da requisição:**

```json
{
  "texto": "Produto excelente! Chegou rápido e funciona perfeitamente."
}
```

**Resposta:**

```json
{
  "sentimento": "positivo",
  "confianca": 0.92
}
```

| Campo       | Tipo   | Descrição                                         |
| ----------- | ------ | ------------------------------------------------- |
| `sentimento`| string | Classificação: `positivo` ou `negativo`            |
| `confianca` | float  | Probabilidade do modelo para a classe prevista     |

**Códigos de erro:**

| Código | Situação                              |
| ------ | ------------------------------------- |
| 400    | Texto do review vazio                 |
| 500    | Modelo não carregado ou erro interno  |

---

## 🐳 Execução com Docker

### 1. Construa a imagem Docker

```bash
docker build -t analise-sentimentos .
```

> A imagem utiliza `python:3.10-slim` como base, instala as dependências e copia o código-fonte junto com o modelo treinado.

### 2. Execute o container

```bash
docker run -d -p 8000:8000 --name api-sentimentos analise-sentimentos
```

A API estará disponível em: **http://localhost:8000**

### 3. Teste a API no container

```bash
curl http://localhost:8000/

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Péssimo produto, não recomendo para ninguém."}'
```

### 4. Gerenciar o container

```bash
# Ver logs do container
docker logs api-sentimentos

# Parar o container
docker stop api-sentimentos

# Reiniciar o container
docker start api-sentimentos

# Remover o container
docker rm api-sentimentos
```

### 5. Remover a imagem (opcional)

```bash
docker rmi analise-sentimentos
```

---

## 📓 Notebooks de Análise

O diretório `notebooks/` contém Jupyter Notebooks com análises detalhadas:

| Notebook                              | Descrição                                               |
| ------------------------------------- | ------------------------------------------------------- |
| `01-analise-exploratoria.ipynb`       | Análise exploratória dos dados (EDA), distribuições, wordclouds |
| `02-avaliacao-modelo.ipynb`           | Avaliação do modelo treinado com métricas e visualizações      |

Para executar os notebooks:

```bash
pip install jupyter
jupyter notebook
```

---

## 🔐 Variáveis de Ambiente

| Variável      | Descrição                          | Valor Padrão                           |
| ------------- | ---------------------------------- | -------------------------------------- |
| `MODEL_PATH`  | Caminho para o modelo treinado     | `models/modelo_sentimento_v1.joblib`   |
| `DATA_PATH`   | Caminho para o dataset de treino   | `data/raw/dataset.csv`                 |

---

