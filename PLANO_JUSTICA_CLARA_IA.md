# Planejamento de evolução: Justiça Clara IA

## 1. Objetivo

Evoluir o MVP jurídico existente — hoje centrado em upload de PDF, extração de metadados, busca de casos semelhantes e predição de duração — para uma plataforma que ajude o cidadão a compreender o andamento de um processo e a encontrar formas oficiais de acompanhamento.

O sistema deve explicar informações com linguagem simples e fontes rastreáveis, apresentar estimativas como faixas condicionais, sinalizar inatividade por comparação com dados históricos e descrever próximos eventos como possibilidades. Não deve substituir orientação jurídica profissional nem afirmar resultados ou prazos como certezas.

## 2. Ponto de partida no repositório

- **Backend:** FastAPI com extração de PDF/OCR, integração opcional com LLM, persistência PostgreSQL, busca vetorial opcional e inferência XGBoost.
- **Frontend:** React e TypeScript, atualmente com uma interface enxuta para as funções do MVP.
- **Dados:** tabelas para processos enviados, histórico e sentenças; há dados de duração total, mas não há ainda uma estrutura completa para histórico de movimentações.
- **Operação e privacidade:** há registro de auditoria e mascaramento mínimo de CPF. A proposta exige ampliar as salvaguardas antes de tratar documentos reais em escala.

### Fonte inicial de dados: DataJud/CNJ

O DataJud será a primeira fonte a ser avaliada para o piloto: sua API pública disponibiliza metadados de processos públicos e movimentações de tribunais brasileiros. A cobertura concreta, a atualidade e a qualidade devem ser verificadas por tribunal e classe antes de prometer acompanhamento em tempo real ou estimativas. A API não dá acesso a processos protegidos por sigilo. O CNJ também informa que a chave pública pode ser alterada; portanto, a integração deve permitir sua rotação sem alteração de código. [Página oficial do DataJud no CNJ](https://www.cnj.jus.br/sistemas/datajud/api-publica/) · [Acesso e autenticação na wiki oficial](https://datajud-wiki.cnj.jus.br/api-publica/acesso/) · [Endpoints por tribunal](https://datajud-wiki.cnj.jus.br/api-publica/endpoints/).

**Configuração proposta:** armazenar a chave vigente em `DATAJUD_API_KEY` no ambiente do backend e enviar `Authorization: APIKey ${DATAJUD_API_KEY}`. Não colocar o valor em código, documentação versionada, frontend, logs ou histórico Git; usar `.env.example` apenas com um valor fictício. Consultar a wiki oficial para obter a chave vigente e revisar o termo de uso antes de integrar. Tratar limites, erros e indisponibilidade da API e registrar a origem e o instante da consulta.

## 3. Escopo funcional

| Funcionalidade | Resultado esperado | Dados e serviços necessários |
|---|---|---|
| Explicação jurídica | Explicar cada movimentação em linguagem simples e mostrar de onde veio a informação | Texto/código e data da movimentação, glossário jurídico versionado, fonte oficial e geração assistida por IA |
| Estimativa de duração | Exibir intervalo provável e contexto de comparação, com data de referência e limitações | Processos históricos comparáveis, tribunal/órgão, classe/assunto, fase, eventos e modelo avaliado |
| Detecção de inatividade | Apontar períodos sem movimentação que destoam de processos comparáveis e permitir conferir a regra | Histórico de movimentações, fase atual, distribuição histórica de intervalos e limiares configuráveis |
| Próximas etapas | Descrever eventos que podem ocorrer em seguida, explicando que dependem do caso | Fase e última movimentação, regras procedimentais revisadas e fontes oficiais |
| Orientação ao cidadão | Indicar canais oficiais adequados e opções de atendimento ou resolução consensual quando pertinentes | Tribunal/UF, tipo de demanda, diretório oficial com URLs, requisitos e data de revisão |

### Fluxo de referência: extravio de bagagem em transporte aéreo

O primeiro protótipo analítico pode usar uma consulta de exemplo para processos públicos ligados a transporte aéreo, extravio de bagagem e pedidos de indenização por danos materiais ou morais. O nome de uma companhia aérea pode ser um filtro opcional, mas não deve definir sozinho a comparabilidade: o conjunto deve priorizar tribunal, classe processual, assunto, fase e características disponíveis do pedido. A busca serve para estimar durações históricas, não para inferir o mérito ou o resultado provável do caso.

1. **Selecionar comparáveis:** consultar os dados públicos disponíveis no DataJud e normalizar classes, assuntos e movimentos com as Tabelas Processuais Unificadas. Guardar os critérios de inclusão e exclusão para que a amostra seja reproduzível.
2. **Validar datas e eventos:** identificar processos com datas coerentes para distribuição e eventos/etapas relevantes. Excluir ou sinalizar registros incompletos, inconsistentes e duplicados. Registrar quando um processo ainda não alcançou a etapa, para não enviesar a duração apenas com casos concluídos.
3. **Calcular estatísticas sem IA generativa:** via SQL/Python, calcular quantidade de casos, mediana, percentis (por exemplo P25–P75) e duração entre etapas. Separar por tribunal, classe, assunto e período quando houver amostra suficiente; aplicar um mínimo de amostra definido no piloto.
4. **Explicar com evidências:** a IA recebe somente os resultados estatísticos aprovados, os filtros usados e a data de consulta e os converte em linguagem simples. Exemplo de formato: “Nos processos comparáveis encontrados, metade chegou a esta etapa em até X meses. Isso é uma referência histórica, não uma previsão garantida para o seu caso.” O valor de X deve ser preenchido pelo cálculo, nunca estimado pela IA.
5. **Exibir fonte e limites:** mostrar critérios da comparação, tribunal/período, número de casos, definição da etapa e possíveis lacunas de atualização da base. Se os dados forem poucos ou incompletos, informar que não há base suficiente para apresentar a referência.

O DataJud fornece metadados e movimentações, mas a presença e a qualidade de datas necessárias a cada etapa precisam ser verificadas no piloto. Se uma data ou evento não estiver disponível ou não representar com segurança o marco desejado, não calcular essa duração até validar uma fonte complementar.

**Primeiro indicador do protótipo:** dias entre a data de ajuizamento/distribuição e o primeiro movimento TPU 12740 (audiência de conciliação registrada). O retorno apresenta mediana, P25 e P75 somente sobre os casos da amostra que têm os dois marcos; conta em separado processos sem dados e duplicatas por número. É um indicador exploratório de registro até etapa, não duração total do processo nem confirmação de que a audiência ocorreu. A amostra atual da busca tem limite de 10 registros e não deve ser apresentada como estimativa confiável ao cidadão.

**Cache do piloto:** persistir em SQLite apenas os agregados estatísticos, com chave de consulta em hash, tribunal, instante da consulta e validade configurável (6 horas por padrão). Não persistir números processuais nem movimentações. Só gravar resumo quando houver pelo menos cinco casos com duração calculável; cache nunca substitui a fonte oficial e a interface exibe a data da consulta original.

## 4. Princípios de produto e segurança

1. **Rastreabilidade:** toda explicação deve apontar a movimentação, documento ou fonte oficial que a sustenta. Conteúdo sem fonte suficiente deve ser identificado como indisponível ou incerto.
2. **Incerteza explícita:** duração aparece como faixa e não como data garantida; próximos eventos são possibilidades, nunca previsões categóricas.
3. **Separação de fatos e interpretação:** mostrar primeiro o dado processual observado e depois a explicação gerada, para o cidadão conferir ambos.
4. **Privacidade desde o desenho:** minimizar coleta e retenção, mascarar dados pessoais antes de serviços externos, restringir acesso e registrar acessos sem copiar conteúdo sensível para logs.
5. **Não induzir ação jurídica indevida:** canais e orientações devem vir de fontes oficiais revisadas; situações que exijam análise individual devem recomendar atendimento adequado, sem gerar aconselhamento personalizado conclusivo.
6. **Acessibilidade e compreensão:** português claro, explicação de termos, navegação por teclado, contraste e uso em dispositivos móveis.
7. **Transparência do modelo:** indicar período e quantidade de dados usados, população comparável, erro observado e fatores que limitam a estimativa.

## 5. Fases de implementação

### Fase 0 — Descoberta, dados e critérios (1–2 semanas)

**Atividades**

- Mapear os fluxos atuais do frontend e os endpoints já existentes no FastAPI.
- Definir usuário prioritário, tribunais/UFs iniciais, tipos de processo e fontes oficiais autorizadas.
- Ler e registrar os requisitos atuais de acesso, endpoints e termo de uso do DataJud; confirmar a chave vigente na documentação oficial e guardá-la apenas em configuração local/segura.
- Inventariar dados disponíveis: movimentações, códigos, datas, documentos, desfechos e cobertura temporal.
- Definir política de retenção, consentimento/aviso ao usuário, anonimização, controle de acesso e resposta a incidentes.
- Estabelecer métricas e critérios de publicação por funcionalidade.

**Entregáveis:** mapa de fluxos e APIs, inventário/qualidade de dados, matriz de riscos e escopo do piloto.

**Critério de saída:** existe fonte confiável e base suficiente para testar um escopo limitado; lacunas de dados estão documentadas.

### Fase 1 — Base de movimentações e fontes (2–4 semanas)

**Atividades**

- Criar modelo para movimentações processuais: identificador, processo, código/tipo, descrição original, data, origem, URL/documento de referência e data de coleta.
- Criar camada de conectores para fontes oficiais, com atualização incremental, deduplicação, tratamento de indisponibilidade e proveniência.
- Implementar primeiro o conector DataJud: selecionar alias/endpoint por tribunal, pesquisar termos exploratórios em classe/assuntos/movimentações, mapear movimentos e metadados para o modelo interno e respeitar o termo de uso e limites publicados. A consulta por número de processo pode ser adicionada em seguida para acompanhamento individual.
- Versionar glossário de termos e catálogo de canais oficiais, incluindo tribunal/UF, URL, público atendido e data de revisão.
- Atualizar auditoria e controles de acesso para consultas e sincronizações.

**Entregáveis:** migração de banco, serviço de ingestão de movimentações e catálogo inicial de fontes.

**Critério de saída:** movimentações públicas de um tribunal piloto podem ser carregadas de forma reproduzível, com origem e data preservadas; cobertura, atraso de atualização e campos ausentes estão medidos e visíveis.

### Fase 2 — Explicação de movimentações (2–3 semanas)

**Atividades**

- Implementar endpoint de explicação baseado em movimentação específica.
- Priorizar explicações determinísticas via glossário; usar IA apenas para contextualizar quando necessário.
- Restringir a geração às evidências recuperadas, exigir referência/citação e validar saída estruturada.
- Apresentar texto original, explicação, fonte e aviso de que a explicação não é decisão judicial nem aconselhamento jurídico.
- Criar revisão humana de amostras com especialistas ou equipe jurídica.

**Entregáveis:** endpoint e componente de interface para explicação rastreável.

**Critério de saída:** cada explicação publicada referencia uma fonte verificável; conteúdo sem evidência não é inventado e é sinalizado.

### Fase 3 — Estimativa em faixas e explicação da incerteza (3–5 semanas)

**Atividades**

- Substituir a apresentação de um único número por intervalos calibrados por tribunal, classe, fase e coorte comparável.
- Entregar primeiro uma linha de base estatística determinística (mediana e percentis das durações observadas, incluindo duração entre etapas); manter a IA fora da geração dos números.
- Avaliar baselines simples e o XGBoost existente; não promover o modelo atual sem validação em dados separados temporalmente.
- Medir erro absoluto, cobertura do intervalo, viés entre grupos/coortes e desempenho quando faltam atributos.
- Exibir data da estimativa, quantidade de comparáveis, intervalo, fatores considerados e limites de aplicabilidade.
- Usar geração de linguagem apenas para explicar estatísticas calculadas, critérios e limitações recebidos como dados estruturados; validar que os números da resposta correspondem aos valores fornecidos.
- Exibir “estimativa indisponível” quando a amostra ou qualidade não atender ao mínimo definido.

**Entregáveis:** relatório de validação, serviço de estimativa e apresentação de faixas no frontend.

**Critério de saída:** limites de erro/cobertura e tamanho mínimo de amostra definidos na Fase 0 são atingidos no piloto; caso contrário, manter a função desativada para aquela coorte.

### Fase 4 — Detecção de inatividade e próximas etapas (2–4 semanas)

**Atividades**

- Calcular tempo desde a última movimentação válida e comparar com a distribuição histórica na mesma fase e grupo comparável.
- Começar com regras estatísticas transparentes; comunicar “acima do padrão observado” em vez de concluir que houve atraso irregular.
- Construir catálogo de eventos possíveis por fase, baseado em fontes oficiais revisadas, sem inferir que ocorrerão necessariamente.
- Diferenciar ausência de atualização no sistema de ausência de atividade real do processo.

**Entregáveis:** serviço de alertas explicáveis e componente de linha do tempo com possíveis próximos eventos.

**Critério de saída:** cada alerta mostra última movimentação, período sem atualização, comparação usada e limitações; cada etapa possível mostra fonte e linguagem condicional.

### Fase 5 — Orientação, acessibilidade e piloto (2–3 semanas)

**Atividades**

- Exibir contatos, consulta processual, Defensoria, canais de atendimento e opções de resolução consensual somente quando aplicáveis, com links oficiais.
- Implementar estados de erro, falta de dados, atualização pendente e divergência entre fontes.
- Fazer revisão de linguagem cidadã, acessibilidade, segurança, privacidade e usabilidade com pessoas do público-alvo.
- Rodar piloto limitado, coletar feedback agregado e corrigir falhas antes de expandir tribunais ou classes.

**Entregáveis:** fluxo completo Justiça Clara IA no tribunal piloto, documentação de operação e plano de expansão.

**Critério de saída:** usuários conseguem localizar a fonte, entender os limites da estimativa e encontrar canal oficial sem ajuda da equipe.

## 6. Mudanças técnicas previstas

### Backend

- Novos módulos para ingestão de movimentações, fontes oficiais, explicações, análise de inatividade e catálogo de atendimento.
- Novos contratos Pydantic para explicação, intervalo de estimativa, alerta e orientação, com campos de evidência e proveniência.
- Endpoints versionados, por exemplo: `GET /api/processos/{id}/movimentacoes`, `GET /api/processos/{id}/explicacoes`, `GET /api/processos/{id}/estimativa`, `GET /api/processos/{id}/alertas` e `GET /api/orientacoes`.
- Migrações explícitas do schema, validação de entrada e proteção contra exposição de dados pessoais.
- Processamento assíncrono e retentativas para sincronização de fontes, preservando idempotência.

### Dados e ML

- Expandir o schema atual, que guarda processos e sentenças, para eventos/movimentações e proveniência das fontes.
- Criar dataset temporal versionado; separar treino, validação e teste por tempo para evitar vazamento de informação futura.
- Usar apenas atributos disponíveis na data em que a estimativa seria produzida.
- Registrar versão do modelo, período dos dados, métricas, população de validade e motivo de indisponibilidade.
- Evitar tratar similaridade semântica como prova de equivalência jurídica; comparar classe, fase e tribunal além do embedding.

### Frontend

- Criar uma visão de processo com: resumo em linguagem simples, movimentações, estimativa em faixa, sinais de inatividade, etapas possíveis, fontes e canais oficiais.
- Incluir controles para expandir o texto original, conferir fontes e informar que uma explicação está incorreta.
- Exibir estados de carregamento, falta de dados, estimativa não confiável e fonte indisponível.
- Garantir uso responsivo e acessível.

## 7. Métricas de sucesso

- **Compreensão:** proporção de usuários que identificam corretamente o significado da última movimentação em teste de usabilidade.
- **Rastreabilidade:** proporção de explicações e orientações com fonte válida e acessível.
- **Qualidade da estimativa:** erro absoluto mediano/MAE, cobertura real dos intervalos e desempenho por coorte relevante.
- **Alertas:** taxa de alertas considerados úteis, falsos positivos e diferença entre demora de publicação e demora processual quando mensurável.
- **Segurança e privacidade:** incidentes, exposição de dados em logs e cumprimento dos prazos de retenção.
- **Acessibilidade:** conclusão das tarefas principais com teclado e leitores de tela, além de revisão de linguagem.

Metas numéricas devem ser definidas depois do diagnóstico dos dados e do piloto; não assumir previamente que um MAE específico é adequado a todos os tribunais e tipos de processo.

## 8. Dependências e decisões a fechar

- Tribunal/UF e classes processuais do primeiro piloto.
- Alias/endpoint DataJud do piloto, cobertura observada, permissões e limites de uso aplicáveis; necessidade de uma fonte oficial complementar para atualização ou campos não cobertos.
- Existência de histórico de eventos com qualidade e cobertura suficientes.
- Responsáveis pela revisão de glossário, regras processuais e canais oficiais.
- Política de autenticação e vínculo entre cidadão e processo.
- Infraestrutura de execução, monitoramento, retenção e suporte.
- Critérios quantitativos para publicação ou bloqueio de estimativas e alertas.

## 9. Sequência recomendada para o MVP cidadão

1. Iniciar com um tribunal e poucas classes processuais.
2. Disponibilizar consulta e linha do tempo com fonte oficial.
3. Publicar explicações determinísticas e revisadas das movimentações mais comuns.
4. Adicionar estimativas em faixa apenas para grupos com dados suficientes e validação satisfatória.
5. Introduzir detecção de inatividade após validar a distribuição histórica por fase.
6. Completar com próximas etapas condicionais e canais oficiais revisados.
7. Expandir gradualmente após avaliação de qualidade, equidade, privacidade e compreensão do público.
