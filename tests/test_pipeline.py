"""Testes das regras de negócio e invariantes do pipeline NexusRisk."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_credit_risk as pcr  # noqa: E402


def _rating(dias: int, score: int) -> str:
    resultado = pcr.classificar_rating_bacen(pd.Series([dias]), pd.Series([score]))
    return str(resultado.iloc[0])


@pytest.mark.parametrize(
    "dias, esperado",
    [
        (0, "AA"),
        (14, "AA"),  # abaixo de 15 dias não impõe piso
        (15, "B"),
        (30, "B"),
        (31, "C"),
        (60, "C"),
        (61, "D"),
        (90, "D"),
        (91, "E"),
        (120, "E"),
        (121, "F"),
        (150, "F"),
        (151, "G"),
        (180, "G"),
        (181, "H"),
        (240, "H"),
    ],
)
def test_rating_por_atraso_res_2682(dias, esperado):
    assert _rating(dias, score=800) == esperado


@pytest.mark.parametrize(
    "score, esperado",
    [(750, "AA"), (749, "A"), (600, "A"), (599, "B"), (300, "B")],
)
def test_rating_por_score_sem_atraso(score, esperado):
    assert _rating(0, score) == esperado


def test_rating_final_e_o_pior_entre_atraso_e_score():
    assert _rating(20, score=800) == "B"  # atraso pior que o score
    assert _rating(0, score=400) == "B"  # score pior que o atraso
    assert _rating(45, score=400) == "C"


def test_aliquotas_cobrem_todos_os_ratings_e_sao_crescentes():
    assert list(pcr.ALIQUOTAS_PCLD) == pcr.ORDEM_RATING
    valores = list(pcr.ALIQUOTAS_PCLD.values())
    assert valores == sorted(valores)
    assert valores[0] == 0.0 and valores[-1] == 1.0


@pytest.fixture(scope="module")
def carteira() -> pd.DataFrame:
    bruto = pcr.gerar_dataset_carteira_credito(n_registros=1200, semente=42)
    return pcr.aplicar_feature_engineering(bruto)


def test_dataset_e_deterministico_por_semente():
    a = pcr.gerar_dataset_carteira_credito(200, semente=7)
    b = pcr.gerar_dataset_carteira_credito(200, semente=7)
    c = pcr.gerar_dataset_carteira_credito(200, semente=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


@pytest.mark.parametrize("n", [1, 10, 333, 1001])
def test_gerador_respeita_n_registros(n):
    assert len(pcr.gerar_dataset_carteira_credito(n_registros=n)) == n


def test_invariantes_do_dataset(carteira):
    assert carteira["id_cliente"].is_unique
    assert not carteira.isna().any().any()
    assert carteira["score_credito"].between(300, 950).all()
    assert (carteira["renda_mensal"] >= pcr.SALARIO_MINIMO_2024).all()
    assert (carteira["saldo_devedor"] <= carteira["limite_concedido"]).all()
    assert (carteira["dias_atraso"] >= 0).all()


def test_flag_d90_e_status_sao_consistentes(carteira):
    d90 = carteira["status_cobranca"] == "Inadimplente (D90+)"
    assert (carteira["flag_d90"] == 1).equals(d90)
    assert (carteira.loc[d90, "dias_atraso"] > 90).all()
    em_dia = carteira["status_cobranca"] == "Em Dia (D0)"
    assert (carteira.loc[em_dia, "dias_atraso"] == 0).all()


def test_faixas_sem_valores_indefinidos(carteira):
    for coluna in ["faixa_score", "faixa_renda", "faixa_dti", "status_cobranca"]:
        assert carteira[coluna].notna().all()


@pytest.mark.parametrize(
    "score, esperada",
    [(549, "Baixo (<550)"), (550, "Médio (550-699)"), (699, "Médio (550-699)"),
     (700, "Alto (700-849)"), (849, "Alto (700-849)"), (850, "Excelente (>=850)")],
)
def test_limites_das_faixas_de_score(score, esperada):
    df = pd.DataFrame({"score_credito": [score]})
    faixa = pcr._faixa_categorica(
        df["score_credito"], [549, 699, 849], pcr.ORDEM_FAIXA_SCORE
    )
    assert faixa[0] == esperada


@pytest.mark.parametrize(
    "dias, esperado",
    [(0, "Em Dia (D0)"), (1, "Atraso Leve (D1-D15)"), (15, "Atraso Leve (D1-D15)"),
     (16, "Alerta (D16-D60)"), (60, "Alerta (D16-D60)"), (61, "Crítico (D61-D90)"),
     (90, "Crítico (D61-D90)"), (91, "Inadimplente (D90+)")],
)
def test_limites_da_regua_de_cobranca(dias, esperado):
    serie = pd.Series([dias])
    status = pcr._faixa_categorica(serie, [0, 15, 60, 90], pcr.ORDEM_STATUS_COBRANCA)
    assert status[0] == esperado


def test_pcld_e_saldo_vezes_aliquota(carteira):
    esperado = (carteira["saldo_devedor"] * carteira["aliquota_pcld"]).round(2)
    pd.testing.assert_series_equal(carteira["pcld_reais"], esperado, check_names=False)
    assert (carteira["pcld_reais"] <= carteira["saldo_devedor"]).all()


def test_alavancagem_toxica_segue_a_definicao(carteira):
    esperado = (carteira["dti"] > 0.70) & (carteira["score_credito"] < 600)
    assert (carteira["flag_alavancagem_toxica"] == 1).equals(esperado)


def test_pergunta_1_taxas_batem_com_a_base(carteira):
    por_canal = pcr.responder_pergunta_1(carteira)["por_canal"]
    assert por_canal["total_clientes"].sum() == len(carteira)
    assert por_canal["inadimplentes_d90"].sum() == carteira["flag_d90"].sum()
    assert por_canal["pcld_acumulado"].sum() == pytest.approx(carteira["pcld_reais"].sum())
    assert por_canal["taxa_default_pct"].is_monotonic_decreasing


def test_pergunta_2_cobre_toda_a_base(carteira):
    p2 = pcr.responder_pergunta_2(carteira)
    assert p2["por_faixa_score"]["clientes"].sum() == len(carteira)
    assert p2["por_faixa_dti"]["clientes"].sum() == len(carteira)


def test_pergunta_3_roi_e_breakeven():
    df = pd.DataFrame(
        {
            "id_cliente": ["a", "b"],
            "saldo_devedor": [300_000.0, 249_450.55],
            "pcld_reais": [0.0, 0.0],
            "status_cobranca": pd.Categorical(
                [pcr.STATUS_ALERTA] * 2, categories=pcr.ORDEM_STATUS_COBRANCA
            ),
        }
    )
    r = pcr.responder_pergunta_3(df)
    # reducao = 0,55 - 0,16 = 0,39; LGD = 0,70 -> breakeven = 150000 / 0,273
    assert r["saldo_alerta_breakeven"] == pytest.approx(549_450.55, abs=0.01)
    assert r["economia_projetada"] == pytest.approx(150_000.0, abs=1.0)
    assert r["roi_multiplicador"] == pytest.approx(1.0, abs=0.01)
    assert r["roi_liquido_pct"] == pytest.approx(0.0, abs=0.1)


def test_pergunta_3_sem_clientes_em_alerta_nao_quebra():
    df = pd.DataFrame(
        {
            "id_cliente": ["a"],
            "saldo_devedor": [1000.0],
            "pcld_reais": [0.0],
            "status_cobranca": pd.Categorical(
                ["Em Dia (D0)"], categories=pcr.ORDEM_STATUS_COBRANCA
            ),
        }
    )
    r = pcr.responder_pergunta_3(df)
    assert r["clientes_em_alerta"] == 0
    assert r["economia_projetada"] == 0.0


def test_formatar_brl():
    assert pcr.formatar_brl(1234567.8) == "R$ 1.234.567,80"
    assert pcr.formatar_brl(0) == "R$ 0,00"


def test_main_gera_csv_e_graficos(tmp_path):
    csv = tmp_path / "saida.csv"
    graficos = tmp_path / "g"
    codigo = pcr.main(
        ["--n-registros", "300", "--saida-csv", str(csv), "--dir-graficos", str(graficos)]
    )
    assert codigo == 0
    assert len(pd.read_csv(csv)) == 300
    assert {p.name for p in graficos.iterdir()} == {
        "default_por_canal.png",
        "default_por_faixa_score.png",
        "pcld_por_rating.png",
        "mapa_calor_default.html",
    }
    assert np.isfinite(pd.read_csv(csv)["pcld_reais"]).all()
