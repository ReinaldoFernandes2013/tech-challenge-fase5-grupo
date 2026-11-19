from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == "candidate_random_forest"


def test_predict_valid_student():
    payload = {
        "fase_num": 3.0,
        "idade": 13.0,
        "anos_na_ong": 2.0,
        "defasagem": -1.0,
        "inde": 5.8,
        "iaa": 8.0,
        "ieg": 5.2,
        "ips": 5.5,
        "ida": 4.9,
        "ipv": 5.6,
        "ian": 5.0,
        "mat": 4.5,
        "por": 5.3,
        "pedra_ord": 1.0,
        "delta_inde_1y": -0.6,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["probabilidade_risco"] <= 1.0
    assert isinstance(data["necessita_acompanhamento_adicional"], bool)
    assert "Human-in-the-Loop" in data["aviso_uso_responsavel"]
    assert len(data["principais_fatores"]) == 5


def test_predict_rejects_invalid_indicator_out_of_range():
    bad_payload = {
        "fase_num": 3.0,
        "idade": 13.0,
        "anos_na_ong": 2.0,
        "defasagem": -1.0,
        "inde": 15.5,
        "iaa": 8.0,
        "ieg": 5.2,
        "ips": 5.5,
        "ida": 4.9,
        "ipv": 5.6,
        "ian": 5.0,
        "mat": 4.5,
        "por": 5.3,
        "pedra_ord": 1.0,
    }
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422
