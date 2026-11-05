# 08 — Avaliação de respostas com BEM

> **Modelos pré-treinados** · Status: **Planejado · standby**  
> Parte do portfólio [Beatriz Helena — Engenharia de IA](https://beatrizorlandi.github.io/projetos/avaliador-bem-kaggle.html)

Design para integrar um modelo de equivalência de respostas, versionar os artefatos e calibrar seu uso.

**Stack principal:** `Python` · `TensorFlow CPU (planejado)` · `KaggleHub (planejado)` · `scikit-learn` · `MLflow`

---

## O Problema

Uma resposta pode usar palavras diferentes e preservar o significado; um avaliador também pode errar em números, negações e entidades.

### Escopo Entregue

Planejamento do adaptador para o Google BERT Answer Equivalence BEM, candidato hospedado no Kaggle. Sem download, execução de pesos ou calibração realizada.

---

## Arquitetura e Fluxo

**1. Model card + licença** → **2. Artefatos + tokenizer** → **3. Adaptador** → **4. Calibração em inglês** → **5. Teste fechado** → **6. Ficha de avaliação**

### Decisões Técnicas

- Preservar handle, versão, licença e hashes dos artefatos e do preprocessamento.
- Começar em inglês e não presumir qualidade em português.
- Comparar exact match e token-F1; não usar equivalência como prova de factualidade ou autorização.

---

## Validação e Critérios de Aceite

Aceite futuro: verificar assinatura e tokenizer, calibrar o limiar em split próprio e medir erros em um teste fechado de triplas pergunta/referência/candidata.


---

## Limites Conhecidos e Próximos Passos

### Limites Conhecidos
- Depende de aquisição externa de pesos e tokenizer e de runtime TensorFlow compatível.
- O modelo é de terceiros; a contribuição autoral será a integração, o benchmark e a análise de limites.

### Próximos Incrementos
- Revalidar o model card e a licença na aquisição.
- Criar benchmark em inglês e adaptador com truncamento explícito.

---

## Estrutura do Repositório

- `informacoes_tecnicas/PLANEJAMENTO.md`: Planejamento técnico detalhado (requisitos, contratos, arquitetura, MLflow e validação).
- `00_PLANEJAMENTO.py`: Versão em notebook Databricks do planejamento técnico.
- Notebooks e módulos em Python/PySpark (quando aplicável ao estágio de implementação do estudo).
- `evidencias/` (quando aplicável): Resumo estático das execuções validadas no Databricks.

## Links

- **Portfólio Completo:** [https://beatrizorlandi.github.io](https://beatrizorlandi.github.io)
- **Estudo de Caso no Site:** [https://beatrizorlandi.github.io/projetos/avaliador-bem-kaggle.html](https://beatrizorlandi.github.io/projetos/avaliador-bem-kaggle.html)
- **Planejamento Técnico no Site:** [https://beatrizorlandi.github.io/documentacao/avaliador-bem-kaggle.html](https://beatrizorlandi.github.io/documentacao/avaliador-bem-kaggle.html)
