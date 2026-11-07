# 💳 NexusRisk: Otimização de Carteira de Crédito PF & Mitigação de Inadimplência D90+

[![CI](https://github.com/wendellbarbosacosta-cmyk/nexusrisk-credit-analytics-projeto-completo/actions/workflows/ci.yml/badge.svg)](https://github.com/wendellbarbosacosta-cmyk/nexusrisk-credit-analytics-projeto-completo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=for-the-badge)](https://seaborn.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![PEP 8](https://img.shields.io/badge/Code%20Style-PEP%208-blueviolet?style=for-the-badge)](https://pep8.org)
[![Regulatory](https://img.shields.io/badge/Compliance-BACEN%20CMN%202.682-008080?style=for-the-badge)](https://www.bcb.gov.br)

> **Projeto de Análise de Dados e Inteligência de Negócio para o setor financeiro / fintechs.**
> Identifica onde nasce a inadimplência crônica (D90+), avalia a calibração de limites pelo comprometimento de renda (DTI) e calcula a provisão regulatória (PCLD) sobre uma carteira **sintética e reprodutível**.

> [!NOTE]
> Todos os dados são **simulados** (gerados por `pipeline_credit_risk.py` com semente fixa). Os números da seção 4 são a saída real do código e podem ser reproduzidos com um único comando. Os números da seção 1 descrevem o **cenário fictício** que motiva a análise e **não** são produzidos pelo código.

---

## 📌 1. Contexto do Problema & Desafio de Negócio (cenário fictício)

Uma fintech brasileira de cartões e crédito digital expandiu sua base em **+600% em 18 meses**, impulsionada por tráfego pago e parcerias com afiliados. Premissas do cenário:

* O NPL 90+ (D90+) subiu de **4,2% para 9,8%**, acima da média de mercado de referência (6,5%).
* A **PCLD** (Provisão para Créditos de Liquidação Duvidosa) passou a consumir **42% do lucro operacional bruto**.

O Comitê de Risco precisa responder a três perguntas:

1. *Onde está o vazamento de inadimplência e qual canal destrói rentabilidade?*
2. *Os limites estão calibrados ou há alavancagem excessiva em perfis vulneráveis?*
3. *Qual o retorno financeiro (ROI) de uma intervenção preventiva antes do default irreversível?*

---

## 🎯 2. Regras de Negócio e Arcabouço Regulatório

1. **Régua de atraso** (`status_cobranca`):
   * `Em Dia (D0)` · `Atraso Leve (D1-D15)` · `Alerta (D16-D60)` · `Crítico (D61-D90)` · `Inadimplente (D90+)`.
   * `flag_d90 = 1` quando `dias_atraso > 90` (definição de default/NPL usada em todo o projeto).

2. **Rating e PCLD (Resolução CMN nº 2.682/1999)**
   O rating final é o **pior** entre o rating por atraso e o rating por score:

   | Atraso | Rating mínimo | Alíquota PCLD |
   | :--- | :---: | :---: |
   | < 15 dias | sem piso (usa score) | — |
   | 15–30 dias | B | 1,0% |
   | 31–60 dias | C | 3,0% |
   | 61–90 dias | D | 10,0% |
   | 91–120 dias | E | 30,0% |
   | 121–150 dias | F | 50,0% |
   | 151–180 dias | G | 70,0% |
   | > 180 dias | H | 100,0% |

   Rating por score (política interna): `>= 750 → AA (0%)`, `600–749 → A (0,5%)`, `< 600 → B (1%)`.
   `PCLD (R$) = Saldo Devedor × Alíquota do Rating`

3. **DTI (Debt-to-Income)**: `DTI = Limite Concedido / Renda Mensal`.
   * Limite prudencial adotado: `DTI <= 0,40`.
   * **Alavancagem tóxica** (`flag_alavancagem_toxica`): `DTI > 0,70` **e** `Score < 600`.

---

## 📊 3. Estrutura do Projeto

| Arquivo | Função |
| :--- | :--- |
| `pipeline_credit_risk.py` | Geração de dados, feature engineering vetorizado, respostas às 3 perguntas, gráficos e exportação |
| `dataset_carteira_credito_analisada.csv` | Saída do pipeline (1.200 clientes, semente 42); **gerado localmente e não versionado** |
| `tests/test_pipeline.py` | 54 testes `pytest`: rating/PCLD nos limites de cada faixa, invariantes do dataset, perguntas 1–3 e execução ponta a ponta |
| `requirements.txt` / `requirements-dev.txt` | Dependências com versões fixadas e validadas / idem + `pytest` |
| `run_analysis.sh` | Setup do ambiente virtual + execução (Linux/macOS/Git Bash) |

---

## 💡 4. Resultados da Amostra Sintética (1.200 clientes, semente 42)

| KPI | Valor |
| :--- | :---: |
| Saldo devedor total | R$ 3.029.467,80 |
| Taxa de default D90+ | 8,25% |
| PCLD total | R$ 196.997,33 |
| Score médio | 650 |
| DTI médio | 0,74x renda |

### 🔍 Pergunta 1: Qual canal concentra a inadimplência?

| Canal | Clientes | Default D90+ | PCLD |
| :--- | :---: | :---: | :---: |
| **Parcerias & Afiliados** | 275 | **17,45%** | R$ 104.588 |
| Tráfego Pago | 405 | 6,42% | R$ 58.855 |
| Orgânico | 338 | 5,33% | R$ 21.753 |
| Member-get-Member | 182 | 3,85% | R$ 11.801 |

* **Constatação:** Afiliados são 30% do saldo, mas **48% dos inadimplentes D90+** (48 de 99) e **53% da PCLD**. O default é 4,5x o do canal Member-get-Member.
* **Hipótese de causa raiz:** comissionamento por aprovação (CPA) estimula volume sem qualidade (premissa do cenário; os dados mostram o efeito, não a causa).
* **Ação recomendada:** comissionar afiliados com *chargeback* por inadimplência em 90 dias e limitar o limite inicial do canal.

### 🔍 Pergunta 2: A concessão de limite está calibrada?

| Faixa de score | Clientes | Limite médio | DTI médio | Alavancagem tóxica | Default D90+ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baixo (<550) | 312 | R$ 2.906 | 0,64 | 43,3% | **13,78%** |
| Médio (550-699) | 374 | R$ 3.773 | 0,74 | 9,1% | 5,88% |
| Alto (700-849) | 476 | R$ 4.226 | 0,80 | 0,0% | 6,72% |
| Excelente (>=850) | 38 | R$ 4.670 | 0,89 | 0,0% | 5,26% |

| Faixa de DTI | Clientes | Default D90+ |
| :--- | :---: | :---: |
| Até 0,40 | 133 | 6,77% |
| 0,41 – 0,70 | 423 | 3,31% |
| 0,71 – 1,00 | 428 | 10,51% |
| Acima de 1,00 | 216 | 14,35% |

* **Constatação:** o DTI médio da carteira (0,74) está **quase o dobro** do limite prudencial (0,40), e o default **triplica** acima de DTI 0,70 (3,3% → 10,5–14,4%). Score só discrimina bem abaixo de 550.
* **Ressalva estatística:** a faixa "até 0,40" tem default maior que a seguinte porque concentra clientes de score baixo (limite restrito por política); o efeito é de **confundimento**, não de causalidade do DTI. Faixas pequenas (ex.: Excelente, n=38) têm alta variância.
* **Ação recomendada:** teto de DTI em torno de 0,70 no onboarding, com revisão por score; reavaliar a política que dá multiplicador 0,95 a afiliados e 0,75 a tráfego pago independentemente do score.

### 🔍 Pergunta 3: Qual o ROI da régua preventiva em D16–D60?

Premissas estilizadas (parametrizáveis em `CenarioReguaPreventiva`): rolagem para D90+ de 55% sem ação e 16% com a régua, LGD de 70% e custo de tecnologia de R$ 150.000.

| Indicador | Valor |
| :--- | :---: |
| Clientes em alerta | 105 |
| Saldo em alerta | R$ 295.687 |
| Perda esperada sem ação → com régua | R$ 113.840 → R$ 33.117 |
| Economia projetada | R$ 80.723 |
| ROI (economia ÷ custo) | 0,54x |
| ROI líquido | −46,2% |
| **Saldo mínimo em alerta para payback (break-even)** | **R$ 549.451** |

* **Constatação:** nesta amostra o investimento **não se paga**. A régua só gera retorno com saldo em alerta ≥ R$ 549 mil (≈ 1,9x o desta amostra), ou seja, em uma carteira maior ou com custo menor.
* **Ação recomendada:** validar a premissa de cura (55% → 16%) com um piloto antes de investir; priorizar solução de baixo custo (disparo automatizado via app/WhatsApp) ou aplicar a régua apenas em carteiras acima do break-even.

---

## 🔮 5. Próximos Passos (Fase 2)

- [ ] **Modelo preditivo de default:** LightGBM/XGBoost para estimar P(Default) na esteira de concessão.
- [ ] **Explicabilidade com SHAP:** transparência regulatória e contrapropostas de limite.
- [ ] **Microserviço FastAPI** conteinerizado em Docker.
- [ ] **Testes A/B de política:** Champion/Challenger em 5% da base.

---

## 🛠️ 6. Como Executar Localmente

```bash
git clone https://github.com/wendellbarbosacosta-cmyk/nexusrisk-credit-analytics-projeto-completo.git
cd nexusrisk-credit-analytics-projeto-completo
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline_credit_risk.py
```

Testes: `pip install -r requirements-dev.txt && pytest`. O CI (GitHub Actions) valida Python 3.10 a 3.13 e o `run_analysis.sh` em Linux; Python 3.14 ainda não é suportado pelas versões fixadas.

Opções: `--n-registros 5000 --semente 7 --sem-graficos --saida-csv saida.csv --dir-graficos graficos`.
Gráficos (PNG e mapa de calor HTML interativo) são gerados em `graficos/`; o CSV e os gráficos são reproduzíveis e ignorados pelo Git.

---

## 👤 Autor

**Wendell Barbosa Costa**
*Especialista em Análise de Dados, Risco & Inteligência de Negócio*
* [LinkedIn](https://linkedin.com) • [GitHub](https://github.com/wendellbarbosacosta-cmyk) • [Email](mailto:wendellbarbosacosta@gmail.com)
