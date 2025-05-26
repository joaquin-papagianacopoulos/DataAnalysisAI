import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
import openai
import tempfile
import base64
import os
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io
from gtts import gTTS

class StreamlitCSVAnalyzer:
    def __init__(self):
        self.df = None
        self.analysis_results = {}
    def create_audio_recorder_component(self):
        recorder_html = """
        <div id="audio-recorder">
            <button id="record-btn" onclick="toggleRecording()">🎤 Iniciar Grabación</button>
            <button id="stop-btn" onclick="stopRecording()" disabled>⏹️ Detener</button>
            <div id="status"></div>
            <audio id="audio-playback" controls style="display:none; margin-top:10px;"></audio>
        </div>

        <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        async function toggleRecording() {
            if (!isRecording) {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    document.getElementById('audio-playback').src = audioUrl;
                    document.getElementById('audio-playback').style.display = 'block';
                    
                    // Convert to base64 and send to Streamlit
                    const reader = new FileReader();
                    reader.onloadend = function() {
                        const base64Audio = reader.result.split(',')[1];
                        window.parent.postMessage({
                            type: 'audio_recorded',
                            audio: base64Audio
                        }, '*');
                    };
                    reader.readAsDataURL(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                document.getElementById('record-btn').disabled = true;
                document.getElementById('stop-btn').disabled = false;
                document.getElementById('status').innerHTML = '🔴 Grabando...';
                
            } catch (err) {
                console.error('Error accessing microphone:', err);
                document.getElementById('status').innerHTML = '❌ Error: No se puede acceder al micrófono';
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                isRecording = false;
                audioChunks = [];
                
                document.getElementById('record-btn').disabled = false;
                document.getElementById('stop-btn').disabled = true;
                document.getElementById('status').innerHTML = '✅ Grabación completada';
            }
        }
        </script>
        """
        return recorder_html
    def load_data(self, uploaded_file):
        self.df = pd.read_csv(uploaded_file)
        return self.df
    def setup_ai_assistant(self, openai_key):
        self.client = openai.OpenAI(api_key=openai_key)
    
    def get_data_context(self):
        if self.df is None:
            return "No hay datos cargados"
            
        context = f"""
        Dataset actual:
        - {self.df.shape[0]} filas, {self.df.shape[1]} columnas
        - Columnas: {list(self.df.columns)}
        - Tipos: {self.df.dtypes.to_dict()}
        - Estadísticas: {self.df.describe().to_dict()}
        """
        return context
    
    def transcribe_audio(self, audio_bytes):
        # Guardar audio temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text
        finally:
            os.unlink(tmp_file_path)
    
    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='es')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    
    def chat_with_data(self, user_question):
        context = f"""
        Dataset: {self.df.shape[0]} filas, {self.df.shape[1]} columnas
        Columnas: {list(self.df.columns)}
        Muestra: {self.df.head(2).to_dict()}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Eres un analista de datos experto. Contexto: {context}. Responde en español, máximo 100 palabras."},
                {"role": "user", "content": user_question}
            ],
            max_tokens=150
        )
        
        return response.choices[0].message.content
    
    def get_metadata(self):
        return {
            'shape': self.df.shape,
            'numeric_cols': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(self.df.select_dtypes(include=['object']).columns),
            'missing_data': self.df.isnull().sum().to_dict()
        }
    
    def clean_data(self):
        # Limpieza básica
        self.df = self.df.dropna(thresh=len(self.df.columns)*0.5)
        
        # Arreglo para pandas 3.0
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        for col in self.df.select_dtypes(include=['object']).columns:
            mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
            self.df[col] = self.df[col].fillna(mode_val)
        
    def detect_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return self.df[(self.df[column] < lower) | (self.df[column] > upper)]
    
    def analyze_numeric_data(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            outliers = self.detect_outliers(col)
            results[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'outliers_count': len(outliers),
                'outliers_pct': len(outliers) / len(self.df) * 100
            }
        return results
    
    def analyze_categorical_data(self):
        cat_cols = self.df.select_dtypes(include=['object']).columns
        results = {}
        
        for col in cat_cols:
            results[col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(5).to_dict(),
                'mode': self.df[col].mode()[0] if not self.df[col].mode().empty else None
            }
        return results
    def generate_business_insights(self):
        insights = []
        numeric_results = self.analyze_numeric_data()
        cat_results = self.analyze_categorical_data()
        
        # Análisis de ventas/ingresos
        revenue_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                    for keyword in ['venta', 'precio', 'ingreso', 'revenue', 'sales', 'amount'])]
        
        if revenue_cols:
            for col in revenue_cols:
                mean_val = self.df[col].mean()
                top_20_pct = self.df[col].quantile(0.8)
                insights.append({
                    'type': 'Revenue',
                    'insight': f"80% de {col} está por debajo de ${top_20_pct:,.0f}",
                    'decision': f"Enfocarse en clientes/productos que generen >${top_20_pct:,.0f}"
                })
        
        # Análisis de clientes/categorías
        customer_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                        for keyword in ['cliente', 'customer', 'categoria', 'region', 'producto'])]
        
        for col in customer_cols:
            if col in cat_results:
                top_category = max(cat_results[col]['top_values'], key=cat_results[col]['top_values'].get)
                pct = (cat_results[col]['top_values'][top_category] / len(self.df)) * 100
                
                if pct > 30:
                    insights.append({
                        'type': 'Concentration',
                        'insight': f"{top_category} representa {pct:.1f}% de {col}",
                        'decision': "Alta concentración = riesgo. Diversificar base"
                    })
        
        # Análisis de outliers empresariales
        for col, stats in numeric_results.items():
            if stats['outliers_pct'] > 10:
                insights.append({
                    'type': 'Anomalies',
                    'insight': f"{col} tiene {stats['outliers_pct']:.1f}% valores anómalos",
                    'decision': "Investigar causas. Posibles fraudes o errores de proceso"
                })
        
        # Análisis de tendencias temporales
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(revenue_cols) > 0:
            insights.append({
                'type': 'Trends',
                'insight': "Dataset contiene datos temporales",
                'decision': "Implementar análisis de estacionalidad y forecasting"
            })
        
        return insights

def main():
    st.set_page_config(page_title="Analista CSV Inteligente", layout="wide")
    
    st.title("🤖 Analista CSV Inteligente")
    st.markdown("Sube tu CSV y obtén análisis automático completo")
    
    analyzer = StreamlitCSVAnalyzer()
    
    # Sidebar para upload
    with st.sidebar:
        st.header("📁 Cargar Datos")
        uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=['csv'])
        
        if uploaded_file:
            st.success("Archivo cargado correctamente")
    
    if uploaded_file:
        # Cargar y mostrar datos
        df = analyzer.load_data(uploaded_file)
        metadata = analyzer.get_metadata()
        
        # Limpiar datos
        analyzer.clean_data()
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Vista Previa de Datos")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("📈 Resumen")
            st.metric("Filas", metadata['shape'][0])
            st.metric("Columnas", metadata['shape'][1])
            st.metric("Cols. Numéricas", len(metadata['numeric_cols']))
            st.metric("Cols. Categóricas", len(metadata['categorical_cols']))
        
        # Análisis automático
        st.subheader("🔍 Análisis Automático")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Estadísticas", "Visualizaciones", "Outliers", "Reporte", "Decisiones", "Chat IA"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            # Análisis numérico
            if metadata['numeric_cols']:
                numeric_results = analyzer.analyze_numeric_data()
                
                with col1:
                    st.write("**Columnas Numéricas**")
                    numeric_df = pd.DataFrame(numeric_results).T
                    st.dataframe(numeric_df.round(2))
            
            # Análisis categórico
            if metadata['categorical_cols']:
                cat_results = analyzer.analyze_categorical_data()
                
                with col2:
                    st.write("**Columnas Categóricas**")
                    for col, stats in cat_results.items():
                        st.write(f"**{col}:**")
                        st.write(f"- Valores únicos: {stats['unique_count']}")
                        st.write(f"- Valor más común: {stats['mode']}")
        
        with tab2:
            st.write("**Distribuciones**")
            
            # Visualizaciones numéricas
            numeric_cols = metadata['numeric_cols']
            if numeric_cols:
                selected_col = st.selectbox("Selecciona columna numérica:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(df, x=selected_col, title=f'Distribución {selected_col}')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(df, y=selected_col, title=f'Boxplot {selected_col}')
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Matriz de correlación
                if len(numeric_cols) > 1:
                    st.write("**Matriz de Correlación**")
                    corr_matrix = df[numeric_cols].corr()
                    fig_corr = px.imshow(corr_matrix, 
                                       text_auto=True, 
                                       aspect="auto",
                                       title="Correlaciones")
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.write("**Detección de Outliers**")
            
            if metadata['numeric_cols']:
                outlier_col = st.selectbox("Columna para análisis de outliers:", metadata['numeric_cols'])
                outliers = analyzer.detect_outliers(outlier_col)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if len(outliers) > 0:
                        st.write(f"**Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)**")
                        st.dataframe(outliers[[outlier_col]])
                    else:
                        st.success("No se detectaron outliers")
                
                with col2:
                    st.metric("Total Outliers", len(outliers))
                    st.metric("% del Dataset", f"{len(outliers)/len(df)*100:.1f}%")
        
        with tab4:
            st.write("**Reporte Ejecutivo**")
            
            # Generar reporte automático
            report = f"""
            ## Análisis de {uploaded_file.name}
            
            **Resumen del Dataset:**
            - **Tamaño:** {metadata['shape'][0]:,} filas × {metadata['shape'][1]} columnas
            - **Columnas numéricas:** {len(metadata['numeric_cols'])}
            - **Columnas categóricas:** {len(metadata['categorical_cols'])}
            
            **Calidad de Datos:**
            """
            
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            report += f"- **Datos faltantes:** {missing_pct:.1f}% del total\n"
            
            if metadata['numeric_cols']:
                numeric_results = analyzer.analyze_numeric_data()
                total_outliers = sum([stats['outliers_count'] for stats in numeric_results.values()])
                report += f"- **Outliers detectados:** {total_outliers} ({total_outliers/len(df)*100:.1f}%)\n"
                
                report += "\n**Insights Principales:**\n"
                
                # Siempre mostrar al menos algunos insights
                for col, stats in numeric_results.items():
                    # Outliers (umbral más bajo)
                    if stats['outliers_pct'] > 1:
                        report += f"- ⚠️ **{col}** tiene {stats['outliers_pct']:.1f}% outliers\n"
                    
                    # Alta variabilidad
                    if stats['std'] / stats['mean'] > 0.5:
                        report += f"- 📊 **{col}** tiene variabilidad alta (CV: {stats['std']/stats['mean']:.2f})\n"
                    
                    # Distribución sesgada
                    if abs((stats['mean'] - stats['median']) / stats['std']) > 0.5:
                        report += f"- 📈 **{col}** tiene distribución sesgada\n"
                
                # Si no hay insights, mostrar resumen básico
                if not any([stats['outliers_pct'] > 1 or stats['std']/stats['mean'] > 0.5 
                            for stats in numeric_results.values()]):
                    report += "- ✅ Datos numéricos sin anomalías significativas\n"
                    report += f"- 📊 Columna con mayor variación: **{max(numeric_results.keys(), key=lambda x: numeric_results[x]['std'])}**\n"
            st.markdown(report)
            
            # Botón de descarga del reporte
            st.download_button(
                label="📥 Descargar Reporte",
                data=report,
                file_name=f"reporte_{uploaded_file.name.replace('.csv', '')}.md",
                mime="text/markdown"
            )
        with tab5:
            st.write("**Recomendaciones Estratégicas**")
            
            business_insights = analyzer.generate_business_insights()
            
            if business_insights:
                for i, insight in enumerate(business_insights):
                    with st.expander(f"💼 {insight['type']} - Recomendación {i+1}"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write("**Hallazgo:**")
                            st.info(insight['insight'])
                        
                        with col2:
                            st.write("**Decisión Recomendada:**")
                            st.success(insight['decision'])
                
                # Resumen ejecutivo de decisiones
                st.subheader("📋 Plan de Acción")
                action_plan = f"""
                **Prioridades identificadas:**
                
                {chr(10).join([f"• {insight['decision']}" for insight in business_insights[:3]])}
                
                **Próximos pasos:**
                1. Validar hallazgos con equipo comercial
                2. Implementar métricas de seguimiento  
                3. Establecer alertas automáticas
                """
                
                st.markdown(action_plan)
            else:
                st.info("Datos insuficientes para generar recomendaciones específicas")
        with tab6:
            st.write("📞 **Conversación por Voz**")
            
            openai_key = st.text_input("OpenAI API Key:", type="password")
            
            if openai_key:
                analyzer.setup_ai_assistant(openai_key)
                
                st.write("**Presiona el micrófono para hablar:**")
                
                # Grabadora simple
                audio = mic_recorder(
                    start_prompt="🎤 Iniciar grabación",
                    stop_prompt="⏹️ Detener grabación", 
                    key='recorder'
                )
                
                if audio:
                    # Mostrar que se grabó
                    st.success("Audio grabado correctamente")
                    st.audio(audio['bytes'])
                    
                    if st.button("💬 Conversar"):
                        with st.spinner("Procesando..."):
                            try:
                                # Crear archivo temporal
                                audio_bytes = audio['bytes']
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                    tmp_file.write(audio_bytes)
                                    tmp_file_path = tmp_file.name
                                
                                # Transcribir
                                with open(tmp_file_path, "rb") as audio_file:
                                    transcript = analyzer.client.audio.transcriptions.create(
                                        model="whisper-1", 
                                        file=audio_file
                                    )
                                
                                question = transcript.text
                                st.write(f"🗣️ **Tú dijiste:** {question}")
                                
                                # Respuesta IA
                                response = analyzer.chat_with_data(question)
                                st.write(f"🤖 **IA responde:** {response}")
                                
                                # Conversión a voz
                                tts = gTTS(text=response, lang='es')
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                                    tts.save(tts_file.name)
                                    st.audio(tts_file.name, autoplay=True)
                                    os.unlink(tts_file.name)
                                
                                # Guardar conversación
                                if "conversation_history" not in st.session_state:
                                    st.session_state.conversation_history = []
                                
                                st.session_state.conversation_history.append({
                                    'question': question,
                                    'answer': response
                                })
                                
                                os.unlink(tmp_file_path)
                                
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                # Mostrar historial
                if "conversation_history" in st.session_state:
                    st.write("**Historial de conversación:**")
                    for i, conv in enumerate(st.session_state.conversation_history):
                        with st.expander(f"Intercambio {i+1}"):
                            st.write(f"**Tú:** {conv['question']}")
                            st.write(f"**IA:** {conv['answer']}")
            else:
                st.info("Ingresa tu OpenAI API Key")

if __name__ == "__main__":
    main()