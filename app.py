import streamlit as st
import os
import tempfile
import pandas as pd
from src.extractors.llm_extractor import LLMExtractor
from src.processors.whisper_engine import WhisperTranscriber
from src.processors.vosk_engine import VoskTranscriber
from src.processors.wav2vec_engine import Wav2VecTranscriber

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Transcritor UNESP",
    page_icon="🎓",
    layout="centered"
)

with st.sidebar:
    # Tenta carregar imagem se existir
    try:
        if os.path.exists("assets/unesp-logo.svg"):
            st.image("assets/unesp-logo.svg", use_container_width=True)
    except Exception:
        pass

    st.markdown("""
    ### Sobre o Projeto
    <div style='text-align: justify;'>
    Este trabalho foi desenvolvido como requisito para a disciplina de 
    <strong>Aprendizado Profundo</strong> do Programa de
    Pós-Graduação em Ciência da Computação - <strong>UNESP</strong>.
    </div>
    
    ---
    **Docente:** Prof. Dr. Denis Henrique Pinheiro Salvadeo
    
    **Alunos:**
    * André Silva
    * Carlos Eduardo Nogueira
    * Elton Júnior
    """, unsafe_allow_html=True)

st.title("Extrator de conhecimento para coleta de informações de texto e áudio")

# Container de Configurações (Expansível para não poluir)
with st.expander("⚙️ Configurações do Modelo", expanded=True):
    col_config_1, col_config_2 = st.columns(2)
    
    with col_config_1:
        st.markdown("**1. Transcrição de Áudio (ASR)**")
        engine_choice = st.selectbox(
            "Escolha o Motor de Transcrição:", 
            ["Whisper", "Wav2Vec2 (HuggingFace - PT-BR)", "Vosk"]
        )

    with col_config_2:
        st.markdown("**2. Extração de Texto (LLM)**")
        provider_choice = st.selectbox("Escolha o LLM:", ["Groq (Llama 3)", "Google Gemini 3 Pro (preview)", "OpenAI (GPT-4o-mini)"])
        
        # Gerenciamento de Chaves API aqui mesmo
        api_key = None
        if provider_choice == "Groq (Llama 3)":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                api_key = st.text_input("Insira sua Groq API Key:", type="password")
        elif "Gemini" in provider_choice:
            api_key = os.getenv("GOOGLE_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                api_key = st.text_input("Insira sua OpenAI API Key:", type="password")

# ==========================================
# 3. CARREGAMENTO DOS MODELOS (CACHE)
# ==========================================
model_instance = None

@st.cache_resource
def load_whisper():
    return WhisperTranscriber(model_size="base")

@st.cache_resource
def load_wav2vec():
    print("DEBUG: Iniciando carga do Wav2Vec2...")
    # Este modelo tem cerca de 1.2GB e será baixado na primeira execução
    return Wav2VecTranscriber()

@st.cache_resource
def load_vosk():
    return VoskTranscriber(model_path="models/vosk-model-small-pt-0.3")

# Carrega o modelo selecionado silenciosamente (ou mostra spinner se for a 1ª vez)
try:
    if engine_choice == "Whisper":
        with st.spinner("Carregando Whisper na memória..."):
            model_instance = load_whisper()
    elif "Wav2Vec2" in engine_choice: 
        with st.spinner("Carregando Wav2Vec2 na memória..."):
            model_instance = load_wav2vec()
    elif engine_choice == "Vosk":
        with st.spinner("Carregando Vosk na memória..."):
            model_instance = load_vosk()
except Exception as e:
    st.error(f"Erro crítico ao carregar modelo: {e}")

# ==========================================
# 4. ÁREA DE INPUT (TABS)
# ==========================================
st.divider()

def mostrar_processamento_audio(audio_file_input):
    """
    Renderiza o player de áudio e o botão de transcrição.
    """
    # 1. Debug: Se o modelo não carregou, avisa o usuário em vez de esconder o botão
    if model_instance is None:
        st.error("🚨 O modelo de transcrição não foi carregado!")
        st.info("Verifique se você baixou a pasta do Vosk corretamente em 'models/' ou se o Whisper instalou.")
        return # Para a execução aqui

    # 2. Se tudo estiver certo, mostra o player
    st.audio(audio_file_input)
    
    # 3. Botão de ação
    if st.button("🚀 Iniciar Transcrição", key=f"btn_{audio_file_input.name}", type="primary"):
        with st.spinner(f"Processando com {engine_choice}..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                    tmp_file.write(audio_file_input.getvalue())
                    tmp_path = tmp_file.name

                text = model_instance.transcribe(tmp_path)
                
                st.session_state.transcribed_text = text
                os.remove(tmp_path)
                st.rerun()

            except Exception as e:
                st.error(f"Erro: {e}")

# Inicializa estado da transcrição
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None

tab1, tab2, tab3 = st.tabs(["📤 Upload de Arquivo", "🎤 Gravar Áudio", "✍️ Texto Manual"])
audio_file = None

# --- ABA 1: UPLOAD ---
with tab1:
    uploaded_file = st.file_uploader("Arraste seu arquivo aqui", type=["wav", "mp3", "m4a", "ogg"])
    # A mágica acontece aqui: chamamos a função DENTRO da aba
    if uploaded_file:
        mostrar_processamento_audio(uploaded_file)

# --- ABA 2: GRAVAÇÃO ---
with tab2:
    recorded_audio = st.audio_input("Clique para gravar")
    # A mágica acontece aqui também
    if recorded_audio:
        mostrar_processamento_audio(recorded_audio)

# --- ABA 3: TEXTO MANUAL ---
with tab3:
    st.markdown("Cole o texto da ata ou reunião aqui:")
    manual_input = st.text_area("Conteúdo do texto:", height=150, label_visibility="collapsed")
    
    if st.button("Usar este texto", type="primary"):
        if manual_input.strip():
            st.session_state.transcribed_text = manual_input
            st.rerun()

# ==========================================
# 5. LÓGICA DE PROCESSAMENTO (BOTÃO DE TRANSCRIÇÃO)
# ==========================================

# Só mostra o botão de transcrever se tiver áudio E se o usuário não acabou de inserir texto manual
if audio_file and model_instance:
    # Mostra player de áudio
    st.audio(audio_file)
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        process_btn = st.button("🚀 Iniciar Transcrição", type="primary", use_container_width=True)
    
    if process_btn:
        with st.spinner(f"Processando áudio com {engine_choice}..."):
            try:
                # Arquivo temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_path = tmp_file.name

                # Transcrição
                text = model_instance.transcribe(tmp_path)
                
                # Salva no estado e limpa
                st.session_state.transcribed_text = text
                os.remove(tmp_path)
                st.rerun()

            except Exception as e:
                st.error(f"Erro durante o processamento: {e}")

# ==========================================
# 6. RESULTADOS E EXTRAÇÃO
# ==========================================

if st.session_state.transcribed_text:
    st.divider()
    st.subheader("📝 Texto Base")
    st.container(border=True).write(st.session_state.transcribed_text)

    st.subheader("🧠 Extração de Conhecimento")
    
    if st.button("Extrair Tabela de Perguntas & Respostas", type="primary"):
        if not api_key:
            st.warning("⚠️ Você precisa configurar a API Key na aba de configurações (topo da página) primeiro.")
        else:
            with st.spinner("O LLM está analisando o contexto..."):
                try:
                    # Define o provider string baseado na escolha do selectbox
                    if "Groq" in provider_choice:
                        prov_str = "groq"
                    elif "Gemini" in provider_choice:
                        prov_str = "gemini"
                    else:
                        prov_str = "openai"
                    
                    extractor = LLMExtractor(api_key=api_key, provider=prov_str)
                    data = extractor.extract_info(st.session_state.transcribed_text)
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.success("Extração concluída!")
                        
                        st.dataframe(
                            df, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "pergunta": st.column_config.TextColumn("❓ Pergunta", width="medium"),
                                "resposta": st.column_config.TextColumn("💬 Resposta", width="large"),
                                "categoria": st.column_config.TextColumn("🏷️ Tag", width="small"),
                                "citacao_exata": st.column_config.TextColumn("🔍 Evidência (Fonte)", width="large")
                            }
                        )
                        
                        # Botão Download
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Baixar CSV", csv, "extração_unesp.csv", "text/csv")
                    else:
                        st.info("O modelo não encontrou informações factuais suficientes para criar perguntas.")
                        
                except Exception as e:
                    st.error(f"Erro na API do LLM: {e}")