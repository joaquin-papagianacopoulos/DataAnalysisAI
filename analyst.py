from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict

class StatisticalAnalyzer:
    def detect_outliers(self, df: pd.DataFrame, column: str) -> List:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def analyze_distribution(self, df: pd.DataFrame, column: str) -> Dict:
        return {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'skewness': stats.skew(df[column].dropna()),
            'kurtosis': stats.kurtosis(df[column].dropna()),
            'outliers_count': len(self.detect_outliers(df, column))
        }