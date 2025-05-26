from langgraph.graph import Graph
from langchain.schema import Document
from funciones import *
def create_analysis_workflow():
    workflow = Graph()
    
    # Nodos del flujo
    workflow.add_node("load_data", load_and_validate)
    workflow.add_node("clean_data", clean_dataset) 
    workflow.add_node("analyze_numeric", analyze_numerical_data)
    workflow.add_node("analyze_categorical", analyze_categorical_data)
    workflow.add_node("generate_visuals", create_visualizations)
    workflow.add_node("write_report", generate_natural_language_report)
    
    # Flujo condicional
    workflow.add_edge("load_data", "clean_data")
    workflow.add_conditional_edges(
        "clean_data",
        decide_analysis_path,
        {
            "numeric": "analyze_numeric",
            "categorical": "analyze_categorical", 
            "mixed": ["analyze_numeric", "analyze_categorical"]
        }
    )
    
    return workflow.compile()

def decide_analysis_path(state):
    metadata = state["metadata"]
    if len(metadata["numeric_cols"]) > 0 and len(metadata["categorical_cols"]) > 0:
        return "mixed"
    elif len(metadata["numeric_cols"]) > 0:
        return "numeric"
    else:
        return "categorical"