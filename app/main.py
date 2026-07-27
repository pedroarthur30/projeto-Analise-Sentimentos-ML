from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import settings
from src.predict import carregar_modelo, realizar_inferencia

app = FastAPI(
    title="API de Análise de Sentimentos",
    description="Classifica reviews de e-commerce como Positivos ou Negativos",
    version="1.0.0"
)

try:
    modelo_em_producao = carregar_modelo(settings.model_path)
    print(f"Modelo carregado com sucesso de: {settings.model_path}")
except Exception as e:
    print(f"Aviso: Erro ao carregar o modelo. Detalhes: {e}")
    modelo_em_producao = None

class ReviewRequest(BaseModel):
    texto: str

class ReviewResponse(BaseModel):
    sentimento: str
    confianca: float  

@app.get("/")
def read_root():
    """Rota de Healthcheck para verificar se a API está online."""
    return {"status": "online", "mensagem": "API de Análise de Sentimentos funcionando!"}

@app.post("/predict", response_model=ReviewResponse)
def predict_sentiment(request: ReviewRequest):
    """Recebe um texto de review e retorna se é positivo ou negativo."""
    
    # (Erro 500 - Internal Server Error)
    if modelo_em_producao is None:
        raise HTTPException(
            status_code=500, 
            detail="O modelo de Machine Learning não está carregado no servidor."
        )
    
    # (Erro 400 - Bad Request)
    if not request.texto or not request.texto.strip():
        raise HTTPException(
            status_code=400, 
            detail="O texto do review não pode estar vazio."
        )
        
    try:
        resultado, probabilidade = realizar_inferencia(modelo_em_producao, request.texto)
        
        return ReviewResponse(
            sentimento=resultado,
            confianca=probabilidade
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno durante a predição: {str(e)}"
        )