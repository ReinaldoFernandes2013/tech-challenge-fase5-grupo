# Data Card — Base Longitudinal PEDE (Passos Mágicos, 2022–2025)

## Origem e Escopo
Dados educacionais e psicossociais longitudinais da **Associação Passos Mágicos** (Programa de Estudos e Desenvolvimento de Estudantes — PEDE), cobrindo ciclos anuais de 2022 a 2025.

## Dinâmica Temporal e Prevenção de Data Leakage
* A modelagem prediz o estado/necessidade no ciclo seguinte ($t+1$) usando apenas informações disponíveis até o ciclo atual ($t$).
* Variáveis derivadas ou observadas apenas em $t+1$ são excluídas do vetor de features de treino e inferência.

## Reestruturação Metodológica (2025) e Ética
* Mudanças estruturais na metodologia e na organização das fases/indicadores em 2025 alteram distribuições históricas em variáveis como tempo na associação, defasagem e indicadores específicos.
* O sistema atua exclusivamente como apoio estatístico preventivo (*Human-in-the-Loop*), sem substituir a avaliação da equipe psicopedagógica.
