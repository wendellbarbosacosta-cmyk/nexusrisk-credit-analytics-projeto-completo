#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto: NexusRisk - Otimização de Carteira de Crédito e Mitigação de Inadimplência D90+
Segmento: Fintech & Bancos Digitais (Crédito PF & Meios de Pagamento)
Autor: Wendell Barbosa Costa (Portfólio de Data Analytics)
Norma: PEP 8 Compliant, Type Annotations e Docstrings Executivas

Descrição:
    Este script orquestra a geração de dados sintéticos, feature engineering,
    análise exploratória e modelagem de regras de negócio para diagnóstico de
    risco de crédito, respondendo a 3 perguntas críticas de negócio e
    calculando provisões PCLD conforme a Resolução CMN nº 2.682/1999.

    Todas as operações são vetorizadas (NumPy/Pandas); não há loops linha a
    linha. O pipeline é determinístico para uma mesma semente.

Uso:
    python pipeline_credit_risk.py
    python pipeline_credit_risk.py --n-registros 5000 --semente 7 --sem-graficos
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # backend sem janela: funciona em servidores e CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import seaborn as sns  # noqa: E402


# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS E PARÂMETROS REGULATÓRIOS
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Ratings da Resolução CMN nº 2.682, do melhor (AA) ao pior (H).
# A posição na lista é o código ordinal do rating (0 = AA ... 8 = H).
ORDEM_RATING: List[str] = ["AA", "A", "B", "C", "D", "E", "F", "G", "H"]

# Alíquotas mínimas de provisão (PCLD) por rating, alinhadas a ORDEM_RATING.
ALIQUOTAS_PCLD: Dict[str, float] = {
    "AA": 0.000,
    "A": 0.005,
    "B": 0.010,
    "C": 0.030,
    "D": 0.100,
    "E": 0.300,
    "F": 0.500,
    "G": 0.700,
    "H": 1.000,
}

ORDEM_FAIXA_SCORE: List[str] = [
    "Baixo (<550)",
    "Médio (550-699)",
    "Alto (700-849)",
    "Excelente (>=850)",
]
ORDEM_FAIXA_RENDA: List[str] = [
    "Até R$ 2.500",
    "R$ 2.501 - R$ 5.000",
    "R$ 5.001 - R$ 10.000",
    "Acima de R$ 10.000",
]
ORDEM_STATUS_COBRANCA: List[str] = [
    "Em Dia (D0)",
    "Atraso Leve (D1-D15)",
    "Alerta (D16-D60)",
    "Crítico (D61-D90)",
    "Inadimplente (D90+)",
]
ORDEM_FAIXA_DTI: List[str] = [
    "Até 0,40",
    "0,41 - 0,70",
    "0,71 - 1,00",
    "Acima de 1,00",
]

SALARIO_MINIMO_2024 = 1412.0
DTI_PRUDENCIAL = 0.40
DTI_TOXICO = 0.70
SCORE_TOXICO = 600
STATUS_ALERTA = "Alerta (D16-D60)"


@dataclass(frozen=True)
class CenarioReguaPreventiva:
    """Premissas (estilizadas) da régua preventiva de renegociação em D16-D60.

    Attributes:
        taxa_rolagem_sem_acao: % do saldo em alerta que rola para D90+ sem ação.
        taxa_rolagem_com_acao: % do saldo em alerta que rola para D90+ com a régua.
        perda_dado_default: LGD, % do saldo em default efetivamente perdido.
        custo_implementacao: Investimento em tecnologia da régua (R$).
    """

    taxa_rolagem_sem_acao: float = 0.55
    taxa_rolagem_com_acao: float = 0.16
    perda_dado_default: float = 0.70
    custo_implementacao: float = 150_000.0


# =============================================================================
# 2. GERAÇÃO DE DADOS SINTÉTICOS REALISTAS
# =============================================================================

def gerar_dataset_carteira_credito(
    n_registros: int = 1000, semente: int = 42
) -> pd.DataFrame:
    """Gera um conjunto de dados simulado de uma carteira de crédito PF.

    Args:
        n_registros: Quantidade de clientes na carteira.
        semente: Semente pseudoaleatória para reprodutibilidade estrita.

    Returns:
        pd.DataFrame com colunas cadastrais, financeiras e comportamentais.
    """
    rng = np.random.default_rng(semente)

    canais = np.array(
        ["Orgânico", "Tráfego Pago", "Parcerias & Afiliados", "Member-get-Member"]
    )
    produtos = np.array(["Cartão de Crédito", "Empréstimo Pessoal", "Cheque Especial"])

    canal = rng.choice(canais, size=n_registros, p=[0.28, 0.35, 0.22, 0.15])
    produto = rng.choice(produtos, size=n_registros, p=[0.65, 0.25, 0.10])

    # Score bimodal: 30% da base em torno de 480 e 70% em torno de 720.
    n_baixo = int(round(n_registros * 0.3))
    score_bruto = np.concatenate(
        [
            rng.normal(480, 60, size=n_baixo),
            rng.normal(720, 75, size=n_registros - n_baixo),
        ]
    )
    rng.shuffle(score_bruto)
    score = np.clip(np.round(score_bruto), 300, 950).astype(int)

    # Renda log-normal com bônus proporcional ao score, limitada a [SM, 35 mil].
    renda = np.round(
        np.exp(rng.normal(8.0, 0.55, size=n_registros)) + score * 2.5, 2
    )
    renda = np.clip(renda, SALARIO_MINIMO_2024, 35_000.0)

    # Limite concedido = renda x multiplicador de política x ruído comercial.
    # A ordem das condições define a precedência: canal antes de score.
    multiplicador = np.select(
        [
            canal == "Parcerias & Afiliados",
            canal == "Tráfego Pago",
            score >= 750,
            score < 550,
        ],
        [0.95, 0.75, 0.85, 0.35],
        default=0.50,
    )
    limite = renda * multiplicador * rng.uniform(0.7, 1.4, size=n_registros)
    limite = np.round(np.clip(limite, 500.0, 50_000.0), 2)

    saldo_devedor = np.round(limite * rng.uniform(0.35, 0.98, size=n_registros), 2)

    # Probabilidade de atraso: base + fatores de risco aditivos (teto de 85%).
    prob_atraso = np.minimum(
        0.85,
        0.05
        + 0.18 * (canal == "Parcerias & Afiliados")
        + 0.22 * (score < 550)
        + 0.15 * (limite / renda > 0.8),
    )
    atrasou = rng.random(n_registros) < prob_atraso
    dias_possiveis = rng.choice(
        [20, 45, 75, 115, 180, 240],
        size=n_registros,
        p=[0.2, 0.2, 0.15, 0.25, 0.1, 0.1],
    )
    dias_atraso = np.where(atrasou, dias_possiveis, 0)

    taxa_juros_mensal = np.select([score >= 750, score >= 600], [0.029, 0.045], 0.079)

    return pd.DataFrame(
        {
            "id_cliente": [f"CLI-{i:06d}" for i in range(1, n_registros + 1)],
            "canal_aquisicao": canal,
            "produto": produto,
            "score_credito": score,
            "renda_mensal": renda,
            "limite_concedido": limite,
            "saldo_devedor": saldo_devedor,
            "dias_atraso": dias_atraso,
            "taxa_juros_mensal": taxa_juros_mensal,
            "meses_relacionamento": rng.integers(1, 48, size=n_registros),
        }
    )


# =============================================================================
# 3. FEATURE ENGINEERING & REGRAS DE NEGÓCIO
# =============================================================================

def _faixa_categorica(
    valores: pd.Series, limites_superiores: List[float], rotulos: List[str]
) -> pd.Categorical:
    """Classifica `valores` em faixas ordenadas; cada limite superior é inclusivo."""
    posicao = np.searchsorted(limites_superiores, valores.to_numpy(), side="left")
    return pd.Categorical.from_codes(posicao, categories=rotulos, ordered=True)


def classificar_rating_bacen(
    dias_atraso: pd.Series, score_credito: pd.Series
) -> pd.Series:
    """Atribui o rating da Resolução CMN nº 2.682 de forma vetorizada.

    O rating final é o PIOR entre o rating por atraso (mínimo regulatório) e o
    rating por score (política interna de risco):

    - Atraso: 15-30d -> B; 31-60d -> C; 61-90d -> D; 91-120d -> E;
      121-150d -> F; 151-180d -> G; >180d -> H; abaixo de 15d não impõe piso.
    - Score: >= 750 -> AA; 600-749 -> A; < 600 -> B.
    """
    cod_atraso = np.select(
        [
            dias_atraso > 180,
            dias_atraso > 150,
            dias_atraso > 120,
            dias_atraso > 90,
            dias_atraso > 60,
            dias_atraso > 30,
            dias_atraso >= 15,
        ],
        [8, 7, 6, 5, 4, 3, 2],
        default=0,
    )
    cod_score = np.select(
        [score_credito >= 750, score_credito >= 600], [0, 1], default=2
    )
    codigo = np.maximum(cod_atraso, cod_score)
    return pd.Series(
        pd.Categorical.from_codes(codigo, categories=ORDEM_RATING, ordered=True),
        index=dias_atraso.index,
    )


def aplicar_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece o DataFrame com métricas regulatórias, DTI e buckets de atraso."""
    df_feat = df.copy()

    df_feat["dti"] = (df_feat["limite_concedido"] / df_feat["renda_mensal"]).round(3)
    df_feat["flag_alavancagem_toxica"] = (
        (df_feat["dti"] > DTI_TOXICO) & (df_feat["score_credito"] < SCORE_TOXICO)
    ).astype(int)

    df_feat["faixa_score"] = _faixa_categorica(
        df_feat["score_credito"], [549, 699, 849], ORDEM_FAIXA_SCORE
    )
    df_feat["faixa_renda"] = _faixa_categorica(
        df_feat["renda_mensal"], [2500, 5000, 10000], ORDEM_FAIXA_RENDA
    )
    df_feat["faixa_dti"] = _faixa_categorica(
        df_feat["dti"], [DTI_PRUDENCIAL, DTI_TOXICO, 1.0], ORDEM_FAIXA_DTI
    )
    df_feat["status_cobranca"] = _faixa_categorica(
        df_feat["dias_atraso"], [0, 15, 60, 90], ORDEM_STATUS_COBRANCA
    )

    df_feat["flag_d90"] = (df_feat["dias_atraso"] > 90).astype(int)

    df_feat["rating_bacen"] = classificar_rating_bacen(
        df_feat["dias_atraso"], df_feat["score_credito"]
    )
    df_feat["aliquota_pcld"] = df_feat["rating_bacen"].map(ALIQUOTAS_PCLD).astype(float)
    df_feat["pcld_reais"] = (df_feat["saldo_devedor"] * df_feat["aliquota_pcld"]).round(2)

    return df_feat


# =============================================================================
# 4. STORYTELLING & RESPOSTA ÀS 3 PERGUNTAS CRÍTICAS DE NEGÓCIO
# =============================================================================

def responder_pergunta_1(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pergunta 1: Canal de aquisição e faixa de renda com maior taxa de default.

    Returns:
        Dicionário com `por_canal` (visão executiva) e `por_canal_renda`
        (detalhe, ordenado do maior para o menor default).
    """
    agregacoes = dict(
        total_clientes=("id_cliente", "count"),
        volume_carteira=("saldo_devedor", "sum"),
        inadimplentes_d90=("flag_d90", "sum"),
        pcld_acumulado=("pcld_reais", "sum"),
    )

    def _com_taxa(resumo: pd.DataFrame) -> pd.DataFrame:
        resumo = resumo.reset_index()
        resumo["taxa_default_pct"] = (
            resumo["inadimplentes_d90"] / resumo["total_clientes"] * 100
        ).round(2)
        return resumo.sort_values("taxa_default_pct", ascending=False, ignore_index=True)

    return {
        "por_canal": _com_taxa(
            df.groupby("canal_aquisicao", observed=True).agg(**agregacoes)
        ),
        "por_canal_renda": _com_taxa(
            df.groupby(["canal_aquisicao", "faixa_renda"], observed=True).agg(
                **agregacoes
            )
        ),
    }


def responder_pergunta_2(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pergunta 2: Calibração de limite, DTI e score na carteira.

    Returns:
        Dicionário com `por_faixa_score` (limite, DTI, default e PCLD por faixa
        de score) e `por_faixa_dti` (default por faixa de alavancagem).
    """
    por_score = (
        df.groupby("faixa_score", observed=True)
        .agg(
            clientes=("id_cliente", "count"),
            limite_medio=("limite_concedido", "mean"),
            renda_media=("renda_mensal", "mean"),
            dti_medio=("dti", "mean"),
            pct_alavancagem_toxica=("flag_alavancagem_toxica", "mean"),
            taxa_default_pct=("flag_d90", "mean"),
            pcld_total=("pcld_reais", "sum"),
        )
        .reset_index()
    )
    por_score["taxa_default_pct"] = (por_score["taxa_default_pct"] * 100).round(2)
    por_score["pct_alavancagem_toxica"] = (
        por_score["pct_alavancagem_toxica"] * 100
    ).round(2)

    por_dti = (
        df.groupby("faixa_dti", observed=True)
        .agg(clientes=("id_cliente", "count"), taxa_default_pct=("flag_d90", "mean"))
        .reset_index()
    )
    por_dti["taxa_default_pct"] = (por_dti["taxa_default_pct"] * 100).round(2)

    return {"por_faixa_score": por_score, "por_faixa_dti": por_dti}


def responder_pergunta_3(
    df: pd.DataFrame, cenario: Optional[CenarioReguaPreventiva] = None
) -> Dict[str, float]:
    """Pergunta 3: ROI da régua preventiva no estágio D16-D60.

    A perda esperada é `saldo x taxa de rolagem para D90+ x LGD`. O benefício da
    régua é a redução dessa perda; o ROI líquido é `(benefício - custo) / custo`.
    """
    cenario = cenario or CenarioReguaPreventiva()

    clientes_alerta = df[df["status_cobranca"] == STATUS_ALERTA]
    saldo_em_alerta = float(clientes_alerta["saldo_devedor"].sum())
    pcld_atual_alerta = float(clientes_alerta["pcld_reais"].sum())

    reducao_rolagem = cenario.taxa_rolagem_sem_acao - cenario.taxa_rolagem_com_acao
    economia = saldo_em_alerta * reducao_rolagem * cenario.perda_dado_default
    custo = cenario.custo_implementacao

    return {
        "clientes_em_alerta": int(len(clientes_alerta)),
        "saldo_em_alerta": round(saldo_em_alerta, 2),
        "pcld_atual_alerta": round(pcld_atual_alerta, 2),
        "perda_esperada_sem_acao": round(
            saldo_em_alerta
            * cenario.taxa_rolagem_sem_acao
            * cenario.perda_dado_default,
            2,
        ),
        "perda_esperada_com_acao": round(
            saldo_em_alerta
            * cenario.taxa_rolagem_com_acao
            * cenario.perda_dado_default,
            2,
        ),
        "economia_projetada": round(economia, 2),
        "custo_implementacao": custo,
        "roi_multiplicador": round(economia / custo, 2),
        "roi_liquido_pct": round((economia - custo) / custo * 100, 1),
        "saldo_alerta_breakeven": round(
            custo / (reducao_rolagem * cenario.perda_dado_default), 2
        ),
    }


# =============================================================================
# 5. VISUALIZAÇÕES
# =============================================================================

def gerar_graficos(
    df: pd.DataFrame, p1: Dict[str, pd.DataFrame], dir_saida: Path
) -> List[Path]:
    """Gera gráficos estáticos (PNG) e um mapa de calor interativo (HTML)."""
    dir_saida.mkdir(parents=True, exist_ok=True)
    arquivos: List[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=p1["por_canal"],
        x="taxa_default_pct",
        y="canal_aquisicao",
        hue="canal_aquisicao",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="Taxa de default D90+ por canal de aquisição",
        xlabel="Taxa de default D90+ (%)",
        ylabel="",
    )
    arquivos.append(dir_saida / "default_por_canal.png")
    fig.savefig(arquivos[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    default_por_score = (
        df.groupby("faixa_score", observed=True)["flag_d90"].mean().mul(100).reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=default_por_score,
        x="faixa_score",
        y="flag_d90",
        hue="faixa_score",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="Taxa de default D90+ por faixa de score",
        xlabel="Faixa de score",
        ylabel="Taxa de default D90+ (%)",
    )
    arquivos.append(dir_saida / "default_por_faixa_score.png")
    fig.savefig(arquivos[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    pcld_por_rating = (
        df.groupby("rating_bacen", observed=False)["pcld_reais"].sum().reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=pcld_por_rating,
        x="rating_bacen",
        y="pcld_reais",
        hue="rating_bacen",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="PCLD (R$) por rating regulatório",
        xlabel="Rating (Res. CMN 2.682)",
        ylabel="PCLD (R$)",
    )
    arquivos.append(dir_saida / "pcld_por_rating.png")
    fig.savefig(arquivos[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    matriz = p1["por_canal_renda"].pivot(
        index="canal_aquisicao", columns="faixa_renda", values="taxa_default_pct"
    )
    fig_html = px.imshow(
        matriz,
        text_auto=".1f",
        color_continuous_scale="Reds",
        aspect="auto",
        title="Taxa de default D90+ (%) - canal x faixa de renda",
    )
    arquivos.append(dir_saida / "mapa_calor_default.html")
    fig_html.write_html(arquivos[-1], include_plotlyjs="cdn")

    return arquivos


# =============================================================================
# 6. ORQUESTRAÇÃO
# =============================================================================

def formatar_brl(valor: float) -> str:
    """Formata um valor numérico como moeda brasileira (R$ 1.234,56)."""
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


def _secao(titulo: str) -> None:
    print("\n" + "-" * 80)
    print(titulo)
    print("-" * 80)


def main(argv: Optional[List[str]] = None) -> int:
    """Executa o pipeline completo e imprime o relatório executivo."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Pipeline NexusRisk de risco de crédito")
    parser.add_argument("--n-registros", type=int, default=1200)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument(
        "--saida-csv",
        type=Path,
        default=Path("dataset_carteira_credito_analisada.csv"),
    )
    parser.add_argument("--dir-graficos", type=Path, default=Path("graficos"))
    parser.add_argument("--sem-graficos", action="store_true")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("=" * 80)
    print("NEXUSRISK: PIPELINE EXECUTIVO DE ANÁLISE DE RISCO DE CRÉDITO PF")
    print("=" * 80)

    df_bruto = gerar_dataset_carteira_credito(args.n_registros, args.semente)
    df_final = aplicar_feature_engineering(df_bruto)

    print(f"Clientes: {len(df_final):,} | Semente: {args.semente}")
    print(f"Saldo devedor total: {formatar_brl(df_final['saldo_devedor'].sum())}")
    print(f"Taxa de default D90+: {df_final['flag_d90'].mean() * 100:.2f}%")
    print(f"Score médio: {df_final['score_credito'].mean():.0f}")
    print(f"DTI médio: {df_final['dti'].mean():.2f}x renda")
    print(f"PCLD total: {formatar_brl(df_final['pcld_reais'].sum())}")

    p1 = responder_pergunta_1(df_final)
    _secao("PERGUNTA 1 - Onde está o vazamento de inadimplência? (por canal)")
    print(p1["por_canal"].round(2).to_string(index=False))
    _secao("PERGUNTA 1 - Detalhe canal x renda (>= 15 clientes, top 5)")
    detalhe = p1["por_canal_renda"].query("total_clientes >= 15").head(5)
    print(detalhe.round(2).to_string(index=False))

    p2 = responder_pergunta_2(df_final)
    _secao("PERGUNTA 2 - Calibração de limite por faixa de score")
    print(p2["por_faixa_score"].round(2).to_string(index=False))
    _secao("PERGUNTA 2 - Default por faixa de DTI")
    print(p2["por_faixa_dti"].to_string(index=False))

    p3 = responder_pergunta_3(df_final)
    _secao("PERGUNTA 3 - ROI da régua preventiva em D16-D60 (premissas estilizadas)")
    for chave, valor in p3.items():
        rotulo = chave.replace("_", " ")
        if isinstance(valor, float) and "multiplicador" not in chave and "pct" not in chave:
            valor = formatar_brl(valor)
        print(f"  {rotulo:<28}: {valor}")

    args.saida_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(args.saida_csv, index=False, encoding="utf-8-sig")
    print(f"\nDataset analisado salvo em: {args.saida_csv}")

    if not args.sem_graficos:
        arquivos = gerar_graficos(df_final, p1, args.dir_graficos)
        print(f"{len(arquivos)} gráficos salvos em: {args.dir_graficos}/")

    print("\nPipeline executado com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
