#!/usr/bin/env bash
# Script de automação para executar o NexusRisk localmente
set -e

echo "=========================================================="
echo "🚀 NEXUSRISK: Setup e Execução da Análise de Crédito"
echo "=========================================================="

# 1. Criação do Ambiente Virtual se não existir
if [ ! -d ".venv" ]; then
    echo "📦 Criando ambiente virtual Python (.venv)..."
    python3 -m venv .venv
fi

# 2. Ativação
echo "🔌 Ativando ambiente virtual..."
source .venv/bin/activate

# 3. Instalação das dependências
echo "📥 Instalando dependências do requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Execução do pipeline
echo "⚡ Executando o pipeline Python (PEP 8)..."
python pipeline_credit_risk.py

echo "=========================================================="
echo "✅ Execução finalizada! Gráficos gerados em ./graficos/"
echo "=========================================================="
