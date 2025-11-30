import instructor
from openai import OpenAI
import google.generativeai as genai
from src.schemas import ExtractionResult
import os

class LLMExtractor:
    def __init__(self, api_key=None, provider="groq"):
        # ... (código de inicialização igual ao anterior) ...

        self.provider = provider

        if provider == "groq":
             self.client = instructor.from_openai(
                OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key or os.getenv("GROQ_API_KEY")
                ),
                mode=instructor.Mode.JSON
            )
             self.model = "llama-3.1-8b-instant"

        elif provider == "gemini":
            # --- Configuração do Gemini ---
            key = api_key or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=key)
            
            # O instructor envolve o modelo do Google
            self.client = instructor.from_gemini(
                genai.GenerativeModel(model_name="gemini-2.5-flash"),
                mode=instructor.Mode.GEMINI_JSON
            )
            self.model = None # O modelo já está instanciado no client acima
        else:
            self.client = instructor.from_openai(
                OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            )

            self.model = "gpt-4o-mini"

    def extract_info(self, text: str) -> list:
        if not text or len(text) < 10:
            return []

        # --- O SEGREDO ESTÁ NESTE PROMPT ---
        system_prompt = (
            "Você é um extrator de fatos estrito e cético. "
            "Sua tarefa é criar pares de Perguntas e Respostas baseadas APENAS no texto fornecido. "
            "REGRAS OBRIGATÓRIAS:\n"
            "1. NÃO use conhecimento externo ou prévio. Se o texto não diz, você não sabe.\n"
            "2. Se o texto estiver incompleto ou confuso, extraia apenas o que for explícito.\n"
            "3. Para cada resposta, você DEVE encontrar a 'citacao_exata' no texto original.\n"
            "4. Responda em Português do Brasil."
        )

        try:
            # Lógica bifurcada: O Gemini usa sintaxe diferente no Instructor
            if self.provider == "gemini":
                resp = self.client.messages.create(
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\nTEXTO FONTE:\n{text}"}
                    ],
                    response_model=ExtractionResult,
                )
            else:
                # OpenAI e Groq usam chat.completions
                resp = self.client.chat.completions.create(
                    model=self.model,
                    response_model=ExtractionResult,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"TEXTO FONTE:\n{text}"},
                    ],
                )

            return [item.model_dump() for item in resp.tabela_qa]
            
        except Exception as e:
            # Retorna o erro de forma mais limpa para o Streamlit mostrar
            raise RuntimeError(f"Erro no provedor {self.provider}: {str(e)}")