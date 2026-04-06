Estratégia de Prompts para Extração de Dados Jurídicos

Guia para configurar o modelo IA para transformar petições iniciais em dados JSON precisos.

1. O System Prompt (Instrução Mestra)

"És um assistente especializado em Direito Brasileiro e Processo Civil. A tua única tarefa é ler o texto de uma Petição Inicial e extrair metadados específicos. Deves ignorar argumentos retóricos e focar em factos. Se um dado não for encontrado, retorna 'null'. Responde estritamente em formato JSON."

2. Estrutura do Prompt de Extração

Analise o seguinte texto de petição inicial e extraia as informações solicitadas:
1. Número do Processo (se houver).
2. Valor da Causa (apenas o número).
3. Comarca/Tribunal.
4. Nome do Juiz (se mencionado no cabeçalho).
5. Assunto Principal (ex: Dano Moral, Cobrança Indevida, Voo Cancelado).
6. Nome das Partes (Autor e Réu).

TEXTO DA PETIÇÃO:
{texto_extraido_do_pdf}

RESPONDA APENAS NO FORMATO JSON ABAIXO:
{
  "numero_processo": string,
  "valor_causa": float,
  "comarca": string,
  "juiz": string,
  "assunto": string,
  "partes": { "autor": string, "reu": string }
}


3. Técnicas de Precisão

Few-Shot Prompting: Incluir exemplos de classificação (ex: "Se fala sobre bagagem, o assunto é 'Transporte Aéreo'").

Tratamento Numérico: Instruir a IA a focar na seção "Dá-se à causa o valor de..." para evitar confusão com valores de danos morais.

4. Implementação (Python)

import google.generativeai as genai

async def extrair_dados_juridicos(texto_pdf):
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    prompt = f"Extraia os dados desta petição em JSON: {texto_pdf}"
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    return response.text


5. Cuidados e LGPD

Anonimização: Máscara de CPFs e nomes via Regex antes do envio.

Filtros de Segurança: Ajuste de safety_settings para casos de segredo de justiça.

6. Próximos Passos

Após o retorno do JSON, o sistema valida os tipos de dados e usa o assunto para filtrar a busca no banco vetorial.