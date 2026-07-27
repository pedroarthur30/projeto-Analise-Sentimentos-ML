import pandas as pd
import unicodedata
import re

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_arquivo)
    df.dropna(subset=['texto_review'], inplace=True)
    return df

def limpa_texto(texto: str) -> str:
    """Normaliza texto, remove acentos, pontuação e converte para minúsculas."""
    if not isinstance(texto, str):
        return ""
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    texto_limpo = texto_sem_acentos.lower()
    texto_limpo = re.sub(r'[^a-z\s]', '', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    return texto_limpo

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df['texto_limpo'] = df['texto_review'].apply(limpa_texto)
    df['sentimento_label'] = df['sentimento'].map({'positivo': 1, 'negativo': 0})
    return df