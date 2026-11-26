# Model Card — Previsão Preventiva de Acompanhamento ($t \rightarrow t+1$)

## Finalidade e Limites de Uso
Modelo de classificação supervisionada treinado para estimar a probabilidade de necessidade de acompanhamento adicional no ciclo $t+1$, priorizando sensibilidade (*recall* / F2-Score) para triagem preventiva com equipe humana.

## Dados e Variáveis
* Janelas temporais: treino 2022→2023, validação 2023→2024 e teste fora do tempo (OOT) 2024→2025.
* O modelo usa **23 variáveis** (indicadores de desempenho, engajamento, evolução de pedra/INDE e `escola_publica`).
* **Atributo sensível removido:** `genero_bin` não é mais variável do modelo. Ele não tinha justificativa pedagógica documentada, e o enunciado proíbe atributos sensíveis como atalho preditivo. A coluna permanece **apenas nos arquivos `.parquet`**, para auditoria de equidade, e não é aceita pela API de inferência.
* A remoção não altera o desempenho de forma relevante: PR-AUC de validação de 0,499 antes (24 variáveis) e depois (23 variáveis).
* Remover o atributo como entrada **não garante ausência de viés**: outras variáveis podem funcionar como proxy. Por isso a auditoria de equidade por grupo continua necessária.

## Modelo Campeão e Desempenho
Campeão: `candidate_random_forest` (run MLflow `c62e7a2d…`), limiar de decisão 0,38 (otimizado para F2 na validação).

| Janela | PR-AUC | ROC-AUC | Recall | Precisão | F2 |
|---|---|---|---|---|---|
| Validação 2023→2024 | 0,4992 | 0,8188 | 0,7833 | 0,4000 | 0,6573 |
| Teste OOT 2024→2025 | 0,2296 | 0,5484 | 0,4538 | 0,2288 | 0,3792 |

Referência (baseline heurístico, validação): PR-AUC 0,4207, ROC-AUC 0,7861, recall 0,5833.

## Avaliação e Critérios de Promoção (Quality Gate)
* O modelo candidato é avaliado em divisão temporal/fora do tempo e comparado a um baseline heurístico.
* A promoção exige superar margens mínimas de desempenho (`PR-AUC`, `ROC-AUC`, `Recall` e ganho frente ao baseline heurístico).
* Resultado desta versão: **aprovado** nos quatro critérios (PR-AUC 0,4992 ≥ 0,42; ROC-AUC 0,8188 ≥ 0,72; recall 0,7833 ≥ 0,65; ganho de 0,0785 sobre a heurística, com mínimo de 0,03). Os limites do quality gate não foram alterados.

## Riscos, Viés e Governança
* Desequilíbrio de classes e deslocamento de domínio (*domain shift*) podem elevar falsos positivos ou reduzir precisão em novos ciclos.
* **Limitação observada:** no teste fora do tempo (2024→2025) o desempenho cai de forma importante (ROC-AUC 0,5484, próximo de um sorteio). O quality gate avalia a validação, e o OOT indica que o modelo generaliza mal para o ciclo seguinte. Isso reforça o monitoramento de *drift* (PSI) e a necessidade de reavaliar e retreinar a cada ciclo.
* Com precisão baixa, a maioria dos alertas é falso positivo. O uso adequado é triagem com revisão humana, nunca decisão automática.
* Toda decisão final cabe aos profissionais da Passos Mágicos; o modelo não deve ser usado para exclusão ou rotulagem definitiva de estudantes.
