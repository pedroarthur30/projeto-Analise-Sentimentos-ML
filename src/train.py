import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.config import settings
from src.data_processing import carregar_dados, preparar_dados

def treinar_modelo():
    print(f"Carregando e preparando os dados de: {settings.data_path}")
    
    df = carregar_dados(settings.data_path)
    df = preparar_dados(df)
    
    X = df['texto_limpo']
    y = df['sentimento_label']
    
    X_treino, _, y_treino, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])),
        ('scaler', StandardScaler(with_mean=False)),
        ('logreg', LogisticRegression(solver='liblinear', random_state=42))
    ])
    
    parametros_grid = {
        'tfidf__max_features': [500, 1000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'logreg__C': [0.1, 1, 10]
    }
    
    grid_search = GridSearchCV(pipeline, parametros_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    
    print("Iniciando o treinamento...")
    grid_search.fit(X_treino, y_treino)
    melhor_modelo = grid_search.best_estimator_
    
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(melhor_modelo, settings.model_path)
    print(f"Treinamento concluído! Modelo salvo em '{settings.model_path}'")

if __name__ == "__main__":
    treinar_modelo()