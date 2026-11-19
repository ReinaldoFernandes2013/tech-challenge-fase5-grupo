# Plano de Monitoramento e Observabilidade (PSI & Auditoria)

## Métricas de Estabilidade
* **Population Stability Index (PSI)**: monitora a mudança de distribuição entre a população de referência e janelas subsequentes (`< 0.10` estável; `0.10` a `< 0.25` drift moderado/investigar; `>= 0.25` drift severo/avaliar retreinamento).
* **Estatística Kolmogorov-Smirnov (KS)**: complementa a detecção de deslocamento contínuo nas distribuições.

## Eventos de Reestruturação e Ação Operacional
* Na transição envolvendo 2024–2025, alterações metodológicas (como reestruturação de fases e metodologia Casa Mágica / PEDE 2025) geram drift severo em features específicas, acionando o gatilho de investigação e eventual retreinamento.
* Registros de inferência são auditados para rastreabilidade operacional.
