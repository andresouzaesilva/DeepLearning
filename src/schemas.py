from pydantic import BaseModel, Field
from typing import List

class QAItem(BaseModel):
    pergunta: str = Field(..., description="A pergunta formulada com base no texto.")
    resposta: str = Field(..., description="A resposta direta encontrada no texto.")
    categoria: str = Field(..., description="Uma categoria curta para o assunto (ex: Data, Técnico, Pessoal).")
    citacao_exata: str = Field(..., description="A frase exata ou trecho do texto original que justifica esta resposta.")

class ExtractionResult(BaseModel):
    tabela_qa: List[QAItem] = Field(..., description="Lista de pares pergunta-resposta extraídos.")