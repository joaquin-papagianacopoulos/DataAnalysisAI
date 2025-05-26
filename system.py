from agents import *
from workflow import *
from typing import Dict, List, Tuple

class IntelligentCSVAnalyzer:
    def __init__(self):
        self.workflow = create_analysis_workflow()
        self.crew = crew
        
    def analyze_csv(self, file_path: str) -> Dict:
        # Estado inicial
        initial_state = {
            "file_path": file_path,
            "df": None,
            "metadata": None,
            "analysis_results": {},
            "visualizations": {},
            "report": ""
        }
        
        # Ejecutar workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Generar reporte con agentes
        crew_result = self.crew.kickoff({
            "data_summary": final_state["metadata"],
            "analysis_results": final_state["analysis_results"]
        })
        
        return {
            "metadata": final_state["metadata"],
            "analysis": final_state["analysis_results"],
            "visualizations": final_state["visualizations"],
            "executive_report": crew_result
        }

# Uso
analyzer = IntelligentCSVAnalyzer()
results = analyzer.analyze_csv("sales_data.csv")