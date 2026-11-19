# Model Card — Previsão Preventiva de Acompanhamento ($t \rightarrow t+1$)

## Finalidade e Limites de Uso
Modelo de classificação supervisionada treinado para estimar a probabilidade de necessidade de acompanhamento adicional no ciclo $t+1$, priorizando sensibilidade (*recall* / F2-Score) para triagem preventiva com equipe humana.

## Avaliação e Critérios de Promoção (Quality Gate)
* O modelo candidato é avaliado em divisão temporal/fora do tempo e comparado a um baseline heurístico.
* A promoção exige superar margens mínimas de desempenho (`PR-AUC`, `ROC-AUC`, `Recall` e ganho frente ao baseline heurístico).

## Riscos, Viés e Governança
* Desequilíbrio de classes e deslocamento de domínio (*domain shift*) podem elevar falsos positivos ou reduzir precisão em novos ciclos.
* Toda decisão final cabe aos profissionais da Passos Mágicos; o modelo não deve ser usado para exclusão ou rotulagem definitiva de estudantes.
