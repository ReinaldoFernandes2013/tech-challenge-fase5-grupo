from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StudentIndicatorsInput(BaseModel):
    fase_num: float = Field(..., ge=0.0, le=13.0, description="Fase educacional normalizada (0=ALFA a 9)")
    idade: float = Field(..., ge=5.0, le=30.0, description="Idade do estudante em anos")
    anos_na_ong: float = Field(..., ge=0.0, le=20.0, description="Tempo de permanencia na Passos Magicos (anos)")
    defasagem: float = Field(..., ge=-10.0, le=10.0, description="Defasagem idade-serie (Fase Efetiva - Fase Ideal)")
    inde: float = Field(..., ge=0.0, le=10.0, description="Indice de Desenvolvimento Educacional (0 a 10)")
    iaa: float = Field(..., ge=0.0, le=10.0, description="Indicador de Autoavaliacao (0 a 10)")
    ieg: float = Field(..., ge=0.0, le=10.0, description="Indicador de Engajamento (0 a 10)")
    ips: float = Field(..., ge=0.0, le=10.0, description="Indicador Psicossocial (0 a 10)")
    ida: float = Field(..., ge=0.0, le=10.0, description="Indicador de Desempenho Academico (0 a 10)")
    ipv: float = Field(..., ge=0.0, le=10.0, description="Indicador do Ponto de Virada (0 a 10)")
    ian: float = Field(..., ge=0.0, le=10.0, description="Indicador de Adequacao de Nivel (0 a 10)")
    mat: float = Field(..., ge=0.0, le=10.0, description="Nota de Matematica (0 a 10)")
    por: float = Field(..., ge=0.0, le=10.0, description="Nota de Portugues (0 a 10)")
    ing: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Nota de Ingles (opcional, 0 a 10)")
    pedra_ord: float = Field(..., ge=1.0, le=4.0, description="Pedra atual (1=Quartzo, 2=Agata, 3=Ametista, 4=Topazio)")
    pedra_anterior_ord: Optional[float] = Field(default=None, ge=1.0, le=4.0, description="Pedra no ano anterior (1 a 4)")
    delta_inde_1y: float = Field(default=0.0, ge=-10.0, le=10.0, description="Variacao do INDE no ultimo ano")
    escola_publica: float = Field(default=1.0, ge=0.0, le=1.0, description="1=Escola Publica, 0=Rede Privada/Parceira")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    necessita_acompanhamento_adicional: bool
    probabilidade_risco: float
    nivel_prioridade: str
    threshold_aplicado: float
    principais_fatores: List[Dict[str, Any]]
    aviso_uso_responsavel: str
    model_name: str
    model_version: str
    latency_ms: float
