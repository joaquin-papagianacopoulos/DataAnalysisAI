import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class CSVAnalyzer:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.metadata = self._analyze_structure()
    
    def _analyze_structure(self) -> Dict:
        return {
            'shape': self.df.shape,
            'numeric_cols': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(self.df.select_dtypes(include=['object']).columns),
            'missing_data': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }