import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def create_distribution_plots(self, numeric_cols: List[str]) -> Dict:
        plots = {}
        for col in numeric_cols:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histograma
            self.df[col].hist(ax=ax1, bins=30)
            ax1.set_title(f'Distribución {col}')
            
            # Boxplot
            self.df.boxplot(column=col, ax=ax2)
            ax2.set_title(f'Boxplot {col}')
            
            plots[col] = fig
        return plots
    
    def correlation_heatmap(self, numeric_cols: List[str]):
        corr_matrix = self.df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación')
        return plt.gcf()