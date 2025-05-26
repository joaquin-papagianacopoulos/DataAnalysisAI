import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_and_validate(state):
    df = pd.read_csv(state["file_path"])
    metadata = {
        'shape': df.shape,
        'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': list(df.select_dtypes(include=['object']).columns),
        'missing_data': df.isnull().sum().to_dict()
    }
    return {**state, "df": df, "metadata": metadata}

def clean_dataset(state):
    df = state["df"].copy()
    
    # Limpieza básica
    df = df.dropna(thresh=len(df.columns)*0.5)  # Eliminar filas con >50% NaN
    
    # Rellenar valores faltantes
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return {**state, "df": df}

def analyze_numerical_data(state):
    df = state["df"]
    numeric_cols = state["metadata"]["numeric_cols"]
    results = {}
    
    for col in numeric_cols:
        results[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'outliers': detect_outliers_iqr(df, col)
        }
    
    return {**state, "analysis_results": {**state.get("analysis_results", {}), "numeric": results}}

def analyze_categorical_data(state):
    df = state["df"]
    categorical_cols = state["metadata"]["categorical_cols"]
    results = {}
    
    for col in categorical_cols:
        results[col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict(),
            'mode': df[col].mode()[0] if not df[col].mode().empty else None
        }
    
    return {**state, "analysis_results": {**state.get("analysis_results", {}), "categorical": results}}

def create_visualizations(state):
    df = state["df"]
    visualizations = {}
    
    # Histogramas para columnas numéricas
    for col in state["metadata"]["numeric_cols"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        df[col].hist(ax=ax, bins=30)
        ax.set_title(f'Distribución {col}')
        visualizations[f"{col}_histogram"] = fig
    
    return {**state, "visualizations": visualizations}

def generate_natural_language_report(state):
    analysis = state["analysis_results"]
    metadata = state["metadata"]
    
    report = f"""
    REPORTE ANÁLISIS CSV
    
    Dataset: {metadata['shape'][0]} filas, {metadata['shape'][1]} columnas
    
    Columnas numéricas: {len(metadata['numeric_cols'])}
    Columnas categóricas: {len(metadata['categorical_cols'])}
    
    INSIGHTS PRINCIPALES:
    """
    
    if "numeric" in analysis:
        for col, stats in analysis["numeric"].items():
            report += f"\n- {col}: Media={stats['mean']:.2f}, Outliers={len(stats['outliers'])}"
    
    return {**state, "report": report}

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)].index.tolist()