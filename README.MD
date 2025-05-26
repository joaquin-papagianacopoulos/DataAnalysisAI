# 🤖 Analista CSV Inteligente

Sistema de análisis automático de datos CSV con inteligencia artificial que transforma cualquier dataset en insights empresariales accionables.

## 🎯 Características Principales

### 📊 **Análisis Automático Completo**
- **Detección automática** de tipos de columnas (numéricas/categóricas)
- **Limpieza inteligente** de datos faltantes y inconsistencias
- **Estadísticas descriptivas** completas con métricas clave
- **Detección de outliers** usando método IQR

### 📈 **Visualizaciones Interactivas** 
- Histogramas y boxplots dinámicos
- Matriz de correlación con heatmap
- Gráficos adaptativos según tipo de datos
- Interfaz responsive con Plotly

### 🎯 **Insights Empresariales**
- **Análisis de concentración** de clientes/productos
- **Detección de anomalías** que pueden indicar fraudes
- **Identificación de tendencias** y patrones de negocio
- **Recomendaciones estratégicas** automáticas

### 🤖 **Asistente IA Conversacional**
- **Chat por texto** para preguntas sobre los datos
- **Procesamiento de audio** para consultas por voz
- **Respuestas contextualizadas** basadas en tu dataset
- **Conversación fluida** con historial

## 🚀 Instalación

### Prerrequisitos
```bash
Python 3.8+
```

### Instalar dependencias
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly openai gtts streamlit-mic-recorder
```

### Configuración OpenAI
1. Obtén tu API key de [OpenAI](https://platform.openai.com/)
2. Ingresa la key en la interfaz de la aplicación

## 📱 Uso

### Iniciar la aplicación
```bash
streamlit run app.py
```

### Flujo de trabajo
1. **📁 Cargar CSV**: Sube tu archivo de datos
2. **📊 Análisis Automático**: Revisa estadísticas y visualizaciones
3. **🔍 Outliers**: Identifica valores anómalos
4. **📋 Reporte Ejecutivo**: Obtén resumen completo
5. **💼 Decisiones Empresariales**: Lee recomendaciones estratégicas
6. **🤖 Chat IA**: Conversa sobre tus datos

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│  Data Processor  │───▶│   Visualizer    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Business Intel  │◀───│   AI Assistant   │───▶│ Report Generator│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Capacidades por Tipo de Datos

### 📊 **Datos Numéricos**
- Media, mediana, desviación estándar
- Detección de outliers (IQR)
- Análisis de distribución y asimetría
- Coeficiente de variación

### 📝 **Datos Categóricos**
- Conteo de valores únicos
- Valores más frecuentes (top 5)
- Análisis de concentración
- Detección de categorías dominantes

### 📈 **Patrones Empresariales**
- Concentración de ingresos (regla 80/20)
- Diversificación de clientes
- Estacionalidad en ventas
- Indicadores de riesgo

## 🎨 Interfaz de Usuario

### **Pestañas Principales**
- **📊 Estadísticas**: Métricas y resúmenes
- **📈 Visualizaciones**: Gráficos interactivos  
- **🔍 Outliers**: Detección de anomalías
- **📋 Reporte**: Resumen ejecutivo
- **💼 Decisiones**: Recomendaciones empresariales
- **🤖 Chat IA**: Asistente conversacional

## 🔧 Tecnologías Utilizadas

| Componente | Tecnología |
|------------|------------|
| **Frontend** | Streamlit |
| **Análisis** | Pandas, NumPy |
| **Visualización** | Plotly, Matplotlib, Seaborn |
| **IA Conversacional** | OpenAI GPT-3.5 |
| **Procesamiento Audio** | Whisper API, gTTS |
| **Interface Audio** | streamlit-mic-recorder |

## 💡 Casos de Uso

### 🏢 **Empresas**
- Análisis de ventas y performance
- Identificación de clientes VIP
- Detección de fraudes
- Optimización de inventario

### 📊 **Analistas**
- Exploración rápida de datasets
- Generación de reportes automáticos
- Identificación de insights
- Validación de calidad de datos

### 🎓 **Educación**
- Enseñanza de análisis de datos
- Proyectos estudiantiles
- Investigación académica
- Casos prácticos

## ⚡ Características Técnicas

- **Procesamiento**: Hasta 100k filas eficientemente
- **Formatos**: CSV con cualquier delimitador
- **Limpieza**: Automática con preservación de datos
- **Visualizaciones**: Responsivas e interactivas
- **IA**: Respuestas contextualizadas en <150ms
- **Audio**: Transcripción en tiempo real

## 🚀 Próximas Características

- [ ] Soporte para Excel (.xlsx)
- [ ] Integración con APIs de datos
- [ ] Dashboard empresarial avanzado
- [ ] Alertas automáticas por email
- [ ] Exportación a PowerBI/Tableau
- [ ] Análisis predictivo con ML

## 🤝 Contribuir

1. Fork el repositorio
2. Crea tu rama: `git checkout -b mi-caracteristica`
3. Commit: `git commit -m 'Agregar característica'`
4. Push: `git push origin mi-caracteristica`
5. Abre un Pull Request

## 📄 Licencia

MIT License - Libre para uso comercial y personal

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@joaquin-papagianacopoulos](https://github.com/joaquin-papagianacopoulos)
- LinkedIn: [Joaquin Papagianacopoulos](https://linkedin.com/in/joaquinpapagianacopoulos)

---

### 🎯 **¿Por qué este proyecto?**

> "Democratizar el análisis de datos empresariales. Cualquier persona debería poder subir un CSV y obtener insights profesionales en minutos, no horas."

**⭐ Si te resulta útil, dale una estrella al repo!**