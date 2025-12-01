import instructor
from openai import OpenAI
import google.generativeai as genai
from src.schemas import ExtractionResult
import os

class LLMExtractor:
    def __init__(self, api_key=None, provider="groq"):

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
            key = api_key or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=key)
            
            self.client = instructor.from_gemini(
                genai.GenerativeModel(model_name="gemini-2.5-flash"),
                mode=instructor.Mode.GEMINI_JSON
            )
            self.model = None 
        else:
            self.client = instructor.from_openai(
                OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            )

            self.model = "gpt-4o-mini"

    def extract_info(self, text: str) -> list:
        if not text or len(text) < 10:
            return []

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
            if self.provider == "gemini":
                resp = self.client.messages.create(
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\nTEXTO FONTE:\n{text}"}
                    ],
                    response_model=ExtractionResult,
                )
            else:
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
            raise RuntimeError(f"Erro no provedor {self.provider}: {str(e)}")