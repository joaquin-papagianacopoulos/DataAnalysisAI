from crewai import Agent, Task, Crew
from langchain.llms import OpenAI

# Agentes especializados
data_cleaner_agent = Agent(
    role='Data Cleaner',
    goal='Pensar paso a paso en Limpiar y preparar datos CSV',
    backstory='Especialista en detección de inconsistencias',
    llm=OpenAI(temperature=0)
)

analyst_agent = Agent(
    role='Data Analyst', 
    goal='Pensar paso a paso en Realizar análisis estadístico profundo',
    backstory='Experto en estadística y detección de patrones',
    llm=OpenAI(temperature=0)
)

reporter_agent = Agent(
    role='Report Writer',
    goal='Pensar paso a paso en Generar reportes ejecutivos claros',
    backstory='Comunicador técnico especializado',
    llm=OpenAI(temperature=0.3)
)

# Tasks
cleaning_task = Task(
    description="Analizar calidad de datos y aplicar limpieza",
    agent=data_cleaner_agent
)

analysis_task = Task(
    description="Ejecutar análisis estadístico y detectar outliers",
    agent=analyst_agent
)

report_task = Task(
    description="Generar reporte ejecutivo con insights",
    agent=reporter_agent
)

crew = Crew(
    agents=[data_cleaner_agent, analyst_agent, reporter_agent],
    tasks=[cleaning_task, analysis_task, report_task],
    verbose=True
)