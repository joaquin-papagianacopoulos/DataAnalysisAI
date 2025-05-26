import pandas as pd
import numpy as np  
from typing import Dict, List, Tuple
def detect_trends_and_patterns(df: pd.DataFrame) -> Dict:
    insights = {}
    
    # Tendencias temporales
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        for date_col in date_cols:
            df_sorted = df.sort_values(date_col)
            # Detectar estacionalidad, tendencias, etc.
    
    # Correlaciones fuertes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'vars': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr_val
                    })
        insights['strong_correlations'] = strong_correlations
    
    return insights