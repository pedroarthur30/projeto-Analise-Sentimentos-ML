from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_path: Path = RAIZ_PROJETO / 'models' / 'modelo_sentimento_v1.joblib'
    data_path: Path = RAIZ_PROJETO / 'data' / 'raw' / 'dataset.csv'

    model_config = SettingsConfigDict(
        env_file=str(RAIZ_PROJETO / '.env'), 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

settings = Settings()