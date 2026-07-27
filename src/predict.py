import joblib
from src.data_processing import limpa_texto

def carregar_modelo(caminho_modelo: str):
    """Carrega o modelo salvo em disco."""
    try:
        return joblib.load(caminho_modelo)
    except FileNotFoundError:
        return None

def realizar_inferencia(modelo, texto: str) -> tuple[str, float]:
    """Recebe o modelo carregado e uma lista de textos, retornando as previsões numéricas."""
    if not modelo:
        raise ValueError("O modelo não foi carregado corretamente.")
    
    texto_limpo = limpa_texto(texto)
    textos = [texto_limpo]  
    
    predicao = modelo.predict(textos)[0]
    probabilidade = modelo.predict_proba(textos)[0][predicao]
    confianca = max(probabilidade, 0.0)  
    
    resultado = "positivo" if predicao == 1 else "negativo"
    
    return resultado, float(confianca)