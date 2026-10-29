# -*- coding: utf-8 -*-
"""
[VERSÃO V1] Pipeline de Projeção Eleitoral (TSE 2026 - 1º e 2º Turno)
Estrutura organizada:
1. Download (apenas na 1ª execução) e Processamento dos Dados do TSE de 2026 em `dados`
   - Busca automática das URLs oficiais de 2026 na API de Dados Abertos do TSE (evita erro de timestamp no nome do .zip)
   - 2º Turno ordenado estritamente pelo timestamp do BU (DT_BU_RECEBIDO) a partir das 19:00
   - Descarte automático de todos os dados do 2º Turno após 2,5% das urnas apuradas
2. Modelo de Estimação (Pareamento + Transferência Ecológica NNLS + Regressão WLS Swing + Bootstrap 95%)
3. Simulação e Plotagem do Painel Visual (idêntico ao layout de acompanhamento em tempo real)
"""

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile

import duckdb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import statsmodels.api as sm


# ==============================================================================
# 1. CONFIGURAÇÕES, DOWNLOAD (COM CACHE) E PROCESSAMENTO DOS DADOS DO TSE 2026
# ==============================================================================

ANO_ELEICAO = 2026
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_SCRIPT)
PASTA_BASE = os.path.join(DIRETORIO_RAIZ, 'dados')
PASTA_RESULTADOS = os.path.join(DIRETORIO_RAIZ, 'resultados')
PASTA_CSV_1T = os.path.join(PASTA_BASE, 'csv_1t')
PASTA_CSV_2T = os.path.join(PASTA_BASE, 'csv_2t')

ARQUIVO_PARQUET_1T = os.path.join(PASTA_BASE, 'br_2026_1t_pivotado.parquet')
ARQUIVO_PARQUET_2T = os.path.join(PASTA_BASE, 'br_2026_2t_crono_2_5pct.parquet')
ARQUIVO_PAINEL_PNG = os.path.join(PASTA_RESULTADOS, 'painel_projecao_2026_v1.png')

HORA_INICIO_APURACAO = '19:00:00'
LIMITE_MAX_PCT_2T = 2.5

# Timestamps de fallback caso a API CKAN do Portal de Dados Abertos esteja indisponível
TIMESTAMP_FALLBACK_2026 = {
    '1t': '051020261403',
    '2t': '261020260742',
}

os.makedirs(PASTA_BASE, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

UFS = [
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT',
    'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO', 'ZZ'
]


def obter_urls_bweb_2026(turno: str) -> dict[str, str]:
    """
    Consulta a API oficial do Portal de Dados Abertos do TSE (CKAN) para obter
    as URLs exatas dos arquivos `bweb_{turno}_{UF}_*.zip` das Eleições de 2026.
    Caso a API não responda, utiliza o padrão da CDN do TSE para 2026.
    """
    urls_por_uf: dict[str, str] = {}
    api_url = (
        f"https://dadosabertos.tse.jus.br/api/3/action/"
        f"package_show?id=resultados-{ANO_ELEICAO}-boletim-de-urna"
    )

    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            dados = json.loads(resp.read().decode('utf-8'))
            for resource in dados.get('result', {}).get('resources', []):
                url = resource.get('url', '').strip()
                match = re.search(rf"bweb_{turno}_([A-Z]{{2}})_\d+\.zip$", url, re.IGNORECASE)
                if match:
                    uf = match.group(1).upper()
                    urls_por_uf[uf] = url
    except Exception as e:
        print(f"Aviso: Não foi possível consultar o catálogo dinâmico do TSE ({e}). Usando URL padrão 2026.")

    timestamp_padrao = TIMESTAMP_FALLBACK_2026[turno]
    for uf in UFS:
        if uf not in urls_por_uf:
            urls_por_uf[uf] = (
                f"https://cdn.tse.jus.br/estatistica/sead/eleicoes/"
                f"eleicoes{ANO_ELEICAO}/buweb/bweb_{turno}_{uf}_{timestamp_padrao}.zip"
            )

    return urls_por_uf


def baixar_ufs_2026(turno: str, pasta_destino: str) -> None:
    """
    Baixa e extrai os arquivos de Boletim de Urna (bweb) de 2026 do TSE para todas as UFs.
    Só baixa se o CSV da respectiva UF ainda não estiver extraído na pasta.
    """
    os.makedirs(pasta_destino, exist_ok=True)
    arquivos_existentes = os.listdir(pasta_destino)
    urls_2026 = obter_urls_bweb_2026(turno)

    for uf in UFS:
        csv_existe = any(f.endswith('.csv') and f"_{uf}_" in f for f in arquivos_existentes)
        if csv_existe:
            continue

        url = urls_2026[uf]
        caminho_zip = os.path.join(pasta_destino, f"bweb_{turno}_{uf}.zip")

        print(f"[2026 - {turno.upper()}] Baixando {uf}: {url}...")
        try:
            urllib.request.urlretrieve(url, caminho_zip)
            with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                zip_ref.extractall(pasta_destino)
            os.remove(caminho_zip)
        except Exception as e:
            print(f"Erro ao baixar {uf} ({turno}): {e}")


# Query para consolidar o 1º Turno de 2026 direto no DuckDB e salvar em Parquet
QUERY_1T_2026 = """
COPY (
    SELECT
        SG_UF, CD_MUNICIPIO, NR_ZONA, NR_SECAO,
        SUM(CASE WHEN NR_VOTAVEL = 13 THEN QT_VOTOS ELSE 0 END) AS v_A,
        SUM(CASE WHEN NR_VOTAVEL = 22 THEN QT_VOTOS ELSE 0 END) AS v_B,
        SUM(CASE WHEN NR_VOTAVEL NOT IN (13, 22, 95, 96) THEN QT_VOTOS ELSE 0 END) AS v_Outros,
        SUM(CASE WHEN NR_VOTAVEL NOT IN (95, 96) THEN QT_VOTOS ELSE 0 END) AS total_validos
    FROM read_csv('{pasta}/*.csv', delim=';', header=True, encoding='latin-1', ignore_errors=True)
    WHERE CD_CARGO_PERGUNTA = 1
    GROUP BY SG_UF, CD_MUNICIPIO, NR_ZONA, NR_SECAO
) TO '{parquet}' (FORMAT PARQUET)
"""

# Query para ordenar o 2º Turno de 2026 pelo timestamp exato do BU (DT_BU_RECEBIDO) a partir das 19:00
# e manter apenas até 2,5% das urnas apuradas, descartando os outros 97,5%
QUERY_2T_2026 = """
COPY (
    WITH presidente_2t AS (
        SELECT
            SG_UF, CD_MUNICIPIO, NR_ZONA, NR_SECAO,
            COALESCE(
                try_strptime(CAST(DT_BU_RECEBIDO AS VARCHAR), '%d/%m/%Y %H:%M:%S'),
                try_cast(DT_BU_RECEBIDO AS TIMESTAMP)
            ) AS horario_chegada,
            SUM(CASE WHEN NR_VOTAVEL = 13 THEN QT_VOTOS ELSE 0 END) AS v_A,
            SUM(CASE WHEN NR_VOTAVEL = 22 THEN QT_VOTOS ELSE 0 END) AS v_B,
            SUM(CASE WHEN NR_VOTAVEL NOT IN (13, 22, 95, 96) THEN QT_VOTOS ELSE 0 END) AS v_Outros,
            SUM(CASE WHEN NR_VOTAVEL NOT IN (95, 96) THEN QT_VOTOS ELSE 0 END) AS total_validos
        FROM read_csv('{pasta}/*.csv', delim=';', header=True, encoding='latin-1', ignore_errors=True)
        WHERE CD_CARGO_PERGUNTA = 1
        GROUP BY SG_UF, CD_MUNICIPIO, NR_ZONA, NR_SECAO, DT_BU_RECEBIDO
    ),
    totais_nacionais AS (
        SELECT
            COUNT(*) AS total_urnas_brasil,
            SUM(v_A) * 100.0 / NULLIF(SUM(total_validos), 0) AS pct_A_real_100pct,
            MIN(CASE WHEN SG_UF != 'ZZ' THEN CAST(horario_chegada AS DATE) END) AS data_eleicao
        FROM presidente_2t
        WHERE horario_chegada IS NOT NULL
    ),
    apuracao_ordenada AS (
        SELECT
            p.SG_UF,
            p.CD_MUNICIPIO,
            p.NR_ZONA,
            p.NR_SECAO,
            p.horario_chegada,
            p.v_A,
            p.v_B,
            p.v_Outros,
            p.total_validos,
            t.pct_A_real_100pct,
            ROW_NUMBER() OVER (
                ORDER BY p.horario_chegada ASC, p.SG_UF, p.CD_MUNICIPIO, p.NR_ZONA, p.NR_SECAO
            ) * 100.0 / t.total_urnas_brasil AS pct_urnas_acumulado
        FROM presidente_2t p
        CROSS JOIN totais_nacionais t
        WHERE p.horario_chegada >= CAST(t.data_eleicao || ' {hora_inicio}' AS TIMESTAMP)
    )
    SELECT *
    FROM apuracao_ordenada
    WHERE pct_urnas_acumulado <= {limite_pct}
    ORDER BY horario_chegada ASC, SG_UF, CD_MUNICIPIO, NR_ZONA, NR_SECAO
) TO '{parquet}' (FORMAT PARQUET)
"""


def preparar_dados() -> None:
    """
    Executa o download e processamento de 2026 apenas na primeira vez.
    Nas execuções seguintes, reutiliza os arquivos Parquet salvos em `dados`.
    Após salvar os primeiros 2,5% do 2º turno, remove os CSVs brutos do 2º turno para liberar disco.
    """
    start_time = time.time()
    con = duckdb.connect()

    # 1º Turno de 2026: só baixa e processa se o Parquet ainda não existir
    if not os.path.exists(ARQUIVO_PARQUET_1T):
        print("--- [2026 - 1º Turno] Parquet não encontrado. Baixando e processando CSVs do TSE ---")
        baixar_ufs_2026('1t', PASTA_CSV_1T)
        print("Consolidando 1º Turno de 2026 em Parquet via DuckDB...")
        con.execute(
            QUERY_1T_2026.format(
                pasta=PASTA_CSV_1T.replace('\\', '/'),
                parquet=ARQUIVO_PARQUET_1T.replace('\\', '/'),
            )
        )
    else:
        print(f"--- [2026 - 1º Turno] Utilizando base já existente: {ARQUIVO_PARQUET_1T} ---")

    # 2º Turno de 2026: só baixa e processa se o Parquet (cortado em 2,5%) ainda não existir
    if not os.path.exists(ARQUIVO_PARQUET_2T):
        print("--- [2026 - 2º Turno] Parquet não encontrado. Baixando e processando CSVs do TSE ---")
        baixar_ufs_2026('2t', PASTA_CSV_2T)
        print(
            f"Ordenando 2º Turno de 2026 por DT_BU_RECEBIDO (a partir das {HORA_INICIO_APURACAO}) "
            f"e mantendo apenas até {LIMITE_MAX_PCT_2T}% das urnas..."
        )
        con.execute(
            QUERY_2T_2026.format(
                pasta=PASTA_CSV_2T.replace('\\', '/'),
                hora_inicio=HORA_INICIO_APURACAO,
                limite_pct=LIMITE_MAX_PCT_2T,
                parquet=ARQUIVO_PARQUET_2T.replace('\\', '/'),
            )
        )
        # Exclui os CSVs brutos do 2º Turno (descarta definitivamente o restante após 2,5%)
        if os.path.exists(PASTA_CSV_2T):
            print("Excluindo arquivos brutos do restante do 2º Turno de 2026 para economizar espaço...")
            shutil.rmtree(PASTA_CSV_2T)
    else:
        print(f"--- [2026 - 2º Turno] Utilizando base já existente (até {LIMITE_MAX_PCT_2T}%): {ARQUIVO_PARQUET_2T} ---")

    con.close()
    print(f"Verificação/preparação dos dados de 2026 concluída em {time.time() - start_time:.1f}s!")


# ==============================================================================
# 2. MODELO DE ESTIMAÇÃO V1 (PAREAMENTO, PROJEÇÃO HÍBRIDA E BOOTSTRAP)
# ==============================================================================

def parear_eleicoes(df_base_t0: pd.DataFrame, df_atual_t1: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza a base do 1º Turno de 2026 (t0) com o 2º Turno de 2026 (t1).
    """
    chaves_secao = ['SG_UF', 'CD_MUNICIPIO', 'NR_ZONA', 'NR_SECAO']

    # Merge principal entre 1º e 2º turno
    pareado = pd.merge(
        df_base_t0, df_atual_t1, on=chaves_secao, how='left', suffixes=('_t0', '_t1')
    )

    # Identifica quais seções já chegaram na apuração do TSE
    pareado['apurada'] = pareado['total_validos_t1'].notna()

    # Razão logarítmica do 1º e 2º turno (com correção de Haldane +0.5)
    pareado['log_odds_t0'] = np.log((pareado['v_A_t0'] + 0.5) / (pareado['v_B_t0'] + 0.5))
    pareado['log_odds_t1'] = np.log((pareado['v_A_t1'] + 0.5) / (pareado['v_B_t1'] + 0.5))
    pareado['swing'] = pareado['log_odds_t1'] - pareado['log_odds_t0']

    return pareado


def projetar_com_transicao_1t_2t(df_pareado: pd.DataFrame) -> dict:
    """
    Combina o Modelo de Transferência Ecológica (1º -> 2º turno) com a Regressão de Swing (WLS),
    aplicando a rampa de transição entre 5% e 15% de urnas apuradas.
    """
    apuradas = df_pareado[df_pareado['apurada']].copy()
    pendentes = df_pareado[~df_pareado['apurada']].copy()

    pct_apurado = len(apuradas) / len(df_pareado)

    # -------------------------------------------------------------------------
    # A. MODELO 1: Transferência de Votos do 1º para o 2º Turno (NNLS)
    # -------------------------------------------------------------------------
    X_1t_apuradas = apuradas[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values
    X_1t_pendentes = pendentes[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values

    # Estima as taxas de transferência (garantindo que nenhuma taxa seja negativa)
    taxas_para_A, _ = nnls(X_1t_apuradas, apuradas['v_A_t1'].values)
    taxas_para_B, _ = nnls(X_1t_apuradas, apuradas['v_B_t1'].values)

    v_A_pred_transf = X_1t_pendentes @ taxas_para_A
    v_B_pred_transf = X_1t_pendentes @ taxas_para_B
    prop_A_transf = v_A_pred_transf / np.maximum(1.0, v_A_pred_transf + v_B_pred_transf)

    # -------------------------------------------------------------------------
    # B. MODELO 2: Regressão Ponderada de Swing em Log-Odds (WLS + Shrinkage)
    # -------------------------------------------------------------------------
    X_reg_treino = sm.add_constant(pd.get_dummies(apuradas['SG_UF'], drop_first=True, dtype=float))
    X_reg_treino['log_odds_t0'] = apuradas['log_odds_t0']

    modelo_wls = sm.WLS(apuradas['swing'], X_reg_treino, weights=apuradas['total_validos_t0']).fit()

    X_reg_pred = sm.add_constant(pd.get_dummies(pendentes['SG_UF'], drop_first=True, dtype=float))
    X_reg_pred['log_odds_t0'] = pendentes['log_odds_t0']
    X_reg_pred = X_reg_pred.reindex(columns=X_reg_treino.columns, fill_value=0.0)

    swing_wls = modelo_wls.predict(X_reg_pred).values

    # Encolhimento Empírico-Bayesiano para estabilizar UFs com poucas urnas
    peso_eb = len(apuradas) / (len(apuradas) + 80.0)
    swing_medio = np.average(apuradas['swing'], weights=apuradas['total_validos_t0'])
    swing_final = (peso_eb * swing_wls) + ((1 - peso_eb) * swing_medio)

    prop_A_reg = 1.0 / (1.0 + np.exp(-(pendentes['log_odds_t0'].values + swing_final)))

    # -------------------------------------------------------------------------
    # C. RAMPA DE TRANSIÇÃO (5% a 15% apurado)
    # Abaixo de 5%: 100% Transferência | Acima de 15%: 100% Regressão WLS
    # -------------------------------------------------------------------------
    alpha = np.clip((pct_apurado - 0.05) / (0.15 - 0.05), 0.0, 1.0)
    prop_A_combinada = (1.0 - alpha) * prop_A_transf + alpha * prop_A_reg

    # Volume de votos válidos esperado nas pendentes (ajustado pela abstenção observada)
    fator_comparecimento = apuradas['total_validos_t1'].sum() / apuradas['total_validos_t0'].sum()
    total_pred_pendentes = pendentes['total_validos_t0'].values * fator_comparecimento
    v_A_pred_final = total_pred_pendentes * prop_A_combinada

    # Consolidação Nacional (Votos já apurados + Votos projetados nas seções faltantes)
    votos_A_nacional = apuradas['v_A_t1'].sum() + v_A_pred_final.sum()
    votos_totais_nacional = apuradas['total_validos_t1'].sum() + total_pred_pendentes.sum()

    return {
        'projecao_A_pct': (votos_A_nacional / votos_totais_nacional) * 100,
        'peso_regressao_alpha': alpha,
        'taxas_transferencia_A': taxas_para_A  # [Retenção A, Migração B->A, Migração 3ªVia->A]
    }


def cluster_bootstrap_municipio(df_pareado: pd.DataFrame, n_boots: int = 200) -> dict:
    """
    Calcula a margem de erro (Intervalo de Confiança de 95%) reamostrando
    municípios inteiros (clusters) com reposição nas seções já apuradas.
    """
    apuradas = df_pareado[df_pareado['apurada']].copy()
    pendentes = df_pareado[~df_pareado['apurada']].copy()

    grupos_muni = apuradas.groupby('CD_MUNICIPIO').indices
    municipios_unicos = list(grupos_muni.keys())
    n_munis = len(municipios_unicos)

    estimativas_boot = []

    for _ in range(n_boots):
        munis_sorteados = np.random.choice(municipios_unicos, size=n_munis, replace=True)
        idx_boot = np.concatenate([grupos_muni[m] for m in munis_sorteados])
        apuradas_boot = apuradas.iloc[idx_boot]

        df_temp = pd.concat([apuradas_boot, pendentes], ignore_index=True)
        res = projetar_com_transicao_1t_2t(df_temp)
        estimativas_boot.append(res['projecao_A_pct'])

    ic_inferior, ic_superior = np.percentile(estimativas_boot, [2.5, 97.5])
    ponto_central = projetar_com_transicao_1t_2t(df_pareado)['projecao_A_pct']

    return {
        'projecao_central': round(float(ponto_central), 2),
        'ic_95_inferior': round(float(ic_inferior), 2),
        'ic_95_superior': round(float(ic_superior), 2),
        'margem_erro_pp': round(float((ic_superior - ic_inferior) / 2), 2)
    }


# ==============================================================================
# 3. PLOTAGEM NO FORMATO DO PAINEL DA IMAGEM (CARDS + GRÁFICO 0% A 100%)
# ==============================================================================

def fmt_br(valor: float, casas: int = 1) -> str:
    """Formata número decimal com vírgula no padrão brasileiro."""
    return f"{valor:.{casas}f}".replace('.', ',')


def plotar_painel_apuracao_2026(
    df_2t_parcial: pd.DataFrame,
    proj_A: float,
    ic_A_inf: float,
    ic_A_sup: float,
    caminho_saida_png: str = 'painel_projecao_2026_v1.png',
) -> None:
    cor_lula = '#e03c31'
    cor_flavio = '#1a6ed8'
    cor_texto = '#222222'
    cor_sub = '#666666'
    cor_borda = '#e2e2e2'

    df_ord = df_2t_parcial.sort_values('pct_urnas_acumulado').copy()
    v_A_acum = df_ord['v_A'].cumsum()
    tot_acum = df_ord['total_validos'].cumsum()
    filtro_valido = tot_acum > 0
    df_ord = df_ord[filtro_valido].copy()
    df_ord['pct_A_acum'] = (v_A_acum[filtro_valido] / tot_acum[filtro_valido]) * 100.0
    df_ord['pct_B_acum'] = 100.0 - df_ord['pct_A_acum']

    pct_secoes_atual = float(df_ord['pct_urnas_acumulado'].iloc[-1])
    apurado_A_agora = float(df_ord['pct_A_acum'].iloc[-1])
    apurado_B_agora = 100.0 - apurado_A_agora

    proj_B = 100.0 - proj_A
    ic_B_inf = 100.0 - ic_A_sup
    ic_B_sup = 100.0 - ic_A_inf

    ultimo_horario = pd.to_datetime(df_ord['horario_chegada'].iloc[-1])
    hora_str = ultimo_horario.strftime('%H:%M')

    idx_plot = np.unique(np.linspace(0, len(df_ord) - 1, min(150, len(df_ord)), dtype=int))
    x_apurado = df_ord['pct_urnas_acumulado'].iloc[idx_plot].values
    y_A_apurado = df_ord['pct_A_acum'].iloc[idx_plot].values
    y_B_apurado = df_ord['pct_B_acum'].iloc[idx_plot].values

    x_proj = np.linspace(pct_secoes_atual, 100.0, 200)
    t_norm = (x_proj - pct_secoes_atual) / (100.0 - pct_secoes_atual)
    transicao_base = 3 * (t_norm ** 2) - 2 * (t_norm ** 3)
    oscilacao_ilustrativa = -2.2 * np.sin(np.pi * (t_norm ** 0.85)) * ((1.0 - t_norm) ** 1.3)
    y_A_proj = apurado_A_agora + (proj_A - apurado_A_agora) * transicao_base + oscilacao_ilustrativa
    y_B_proj = 100.0 - y_A_proj

    fig = plt.figure(figsize=(10.5, 9.8), facecolor='#fafafa', dpi=120)
    gs = GridSpec(
        3, 2, figure=fig, height_ratios=[0.11, 0.28, 0.61],
        hspace=0.14, wspace=0.06, left=0.04, right=0.96, top=0.96, bottom=0.04
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(
        0.0, 0.78,
        f"Recebendo dados do TSE · {fmt_br(pct_secoes_atual, 2)}% das seções apuradas · atualizado às {hora_str}",
        fontsize=12.5, fontweight='bold', color=cor_texto, va='center'
    )
    ax_header.text(
        0.012, 0.24,
        "  projeção pouco confiável até 2,5% das seções  ",
        fontsize=9.5, color='#555555', va='center',
        bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.8', facecolor='#f3f3f3', edgecolor='#dcdcdc', lw=1)
    )

    def desenhar_card_candidato(ax, nome, partido, cor, proj, ic_inf, ic_sup, apurado):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        card_bg = FancyBboxPatch(
            (0.01, 0.02), 0.98, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor='white', edgecolor=cor_borda, linewidth=1.2
        )
        ax.add_patch(card_bg)

        ax.plot([0.06, 0.10], [0.82, 0.82], color=cor, lw=3.0, solid_capstyle='round')
        ax.text(0.12, 0.82, f"{nome} ", fontsize=12.5, fontweight='bold', color=cor_texto, va='center')
        offset_partido = 0.23 if len(nome) <= 5 else 0.27
        ax.text(offset_partido, 0.82, partido, fontsize=11.5, color='#777777', va='center')

        ax.text(0.06, 0.56, f"{fmt_br(proj, 1)}", fontsize=33, fontweight='bold', color='#767676', va='center')
        ax.text(0.305, 0.52, "%", fontsize=17, fontweight='bold', color='#767676', va='center')
        ax.plot([0.06, 0.34], [0.41, 0.41], color='#888888', linestyle=':', linewidth=1.3)

        ax.text(0.06, 0.31, "projeção pouco confiável até 2,5% das seções", fontsize=9.5, color='#767676', va='center')

        ax.text(
            0.06, 0.18,
            f"faixa de 95%: {fmt_br(ic_inf, 1)} – {fmt_br(ic_sup, 1)}",
            fontsize=11, fontweight='bold', color='#111111', va='center'
        )
        ax.text(
            0.06, 0.08,
            f"apurado agora: {fmt_br(apurado, 1)}%",
            fontsize=10.5, color='#444444', va='center'
        )

    ax_card_A = fig.add_subplot(gs[1, 0])
    desenhar_card_candidato(ax_card_A, "Lula", "PT", cor_lula, proj_A, ic_A_inf, ic_A_sup, apurado_A_agora)

    ax_card_B = fig.add_subplot(gs[1, 1])
    desenhar_card_candidato(ax_card_B, "Flávio", "PL", cor_flavio, proj_B, ic_B_inf, ic_B_sup, apurado_B_agora)

    ax_chart_container = fig.add_subplot(gs[2, :])
    ax_chart_container.set_xlim(0, 1)
    ax_chart_container.set_ylim(0, 1)
    ax_chart_container.axis('off')
    chart_bg = FancyBboxPatch(
        (0.005, 0.01), 0.99, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor='white', edgecolor=cor_borda, linewidth=1.2
    )
    ax_chart_container.add_patch(chart_bg)

    ax = fig.add_axes([0.10, 0.14, 0.67, 0.36], facecolor='white')

    for y_val in [40, 45, 50, 55, 60]:
        ax.axhline(y_val, color='#eeeeee', linewidth=1.0, zorder=1)
    ax.axhline(50.0, color='#cccccc', linewidth=1.5, zorder=2)
    ax.text(99.5, 50.35, "50% dos válidos", fontsize=8.5, color='#888888', ha='right', va='bottom')

    ax.plot(x_apurado, y_A_apurado, color=cor_lula, linestyle='-', linewidth=2.0, zorder=4)
    ax.plot(x_apurado, y_B_apurado, color=cor_flavio, linestyle='-', linewidth=2.0, zorder=4)

    ax.scatter([pct_secoes_atual], [apurado_A_agora], color=cor_lula, s=36, zorder=5)
    ax.scatter([pct_secoes_atual], [apurado_B_agora], color=cor_flavio, s=36, zorder=5)

    ax.plot(x_proj, y_A_proj, color=cor_lula, linestyle=':', linewidth=1.8, zorder=3)
    ax.plot(x_proj, y_B_proj, color=cor_flavio, linestyle=':', linewidth=1.8, zorder=3)

    ax.plot([100, 100], [ic_A_inf, ic_A_sup], color=cor_lula, linewidth=4.5, alpha=0.75, solid_capstyle='butt', zorder=4)
    ax.plot([100, 100], [ic_B_inf, ic_B_sup], color=cor_flavio, linewidth=4.5, alpha=0.75, solid_capstyle='butt', zorder=4)
    ax.scatter([100], [proj_A], color=cor_lula, edgecolors='white', linewidth=1.2, s=55, zorder=6)
    ax.scatter([100], [proj_B], color=cor_flavio, edgecolors='white', linewidth=1.2, s=55, zorder=6)

    ax.text(102.2, proj_A + 0.65, f"{fmt_br(ic_A_inf, 1)}–{fmt_br(ic_A_sup, 1)}", fontsize=9.5, color='#222222', va='bottom', clip_on=False)
    ax.text(102.2, proj_A - 0.55, f"Lula {fmt_br(proj_A, 1)}%", fontsize=10.5, fontweight='bold', color='#111111', va='center', clip_on=False)

    ax.text(102.2, proj_B + 0.45, f"Flávio {fmt_br(proj_B, 1)}%", fontsize=10.5, fontweight='bold', color='#111111', va='center', clip_on=False)
    ax.text(102.2, proj_B - 0.75, f"{fmt_br(ic_B_inf, 1)}–{fmt_br(ic_B_sup, 1)}", fontsize=9.5, color='#222222', va='top', clip_on=False)

    ax.set_xlim(0, 101)
    ax.set_ylim(40, 60.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=9.5, color=cor_sub)
    ax.set_yticks([40, 45, 50, 55, 60])
    ax.set_yticklabels(['40%', '45%', '50%', '55%', '60%'], fontsize=9.5, color=cor_sub)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(axis='both', length=0, pad=6)

    legend_elements = [
        Line2D([0], [0], color=cor_lula, lw=2.2, label='Lula'),
        Line2D([0], [0], color=cor_flavio, lw=2.2, label='Flávio'),
        Line2D([0], [0], color='#555555', lw=2.2, linestyle='-', label='apurado'),
        Line2D([0], [0], color='#555555', lw=2.0, linestyle=':', label='projeção'),
        Line2D([0], [0], color='#555555', marker='|', linestyle='None', markersize=10, markeredgewidth=3.5, label='faixa de 95% (em 100%)'),
    ]
    ax.legend(
        handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.08, 1.15),
        ncol=5, frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.4
    )

    ax_chart_container.text(
        0.025, 0.085,
        "Eixo horizontal: % das seções já totalizadas pelo TSE.",
        fontsize=9.2, color=cor_sub
    )
    ax_chart_container.text(
        0.025, 0.038,
        "A trajetória pontilhada das curvas até 100% é só ilustrativa: não se sabe a ordem em que as seções que faltam serão apuradas.",
        fontsize=9.2, color=cor_sub
    )

    plt.savefig(caminho_saida_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nPainel visual salvo em: {os.path.abspath(caminho_saida_png)}")
    plt.show(block=False)
    plt.pause(2)
    plt.close(fig)


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL (ELEIÇÕES 2026 - V1)
# ==============================================================================

if __name__ == '__main__':
    preparar_dados()

    print("\nCarregando bases Parquet de 'dados'...")
    df_1t_br = pd.read_parquet(ARQUIVO_PARQUET_1T)
    df_2t_crono_br = pd.read_parquet(ARQUIVO_PARQUET_2T)

    pct_A_real_br = float(df_2t_crono_br['pct_A_real_100pct'].iloc[0])

    limite_teste = 2.42
    df_apurado_tempo_real = df_2t_crono_br[df_2t_crono_br['pct_urnas_acumulado'] <= limite_teste].copy()
    if df_apurado_tempo_real.empty:
        df_apurado_tempo_real = df_2t_crono_br.copy()

    pct_secoes_atual = float(df_apurado_tempo_real['pct_urnas_acumulado'].max())
    hora_atualizacao = pd.to_datetime(df_apurado_tempo_real['horario_chegada'].max()).strftime('%H:%M')

    pct_bruto_A = (
        df_apurado_tempo_real['v_A'].sum() / df_apurado_tempo_real['total_validos'].sum()
    ) * 100.0
    pct_bruto_B = 100.0 - pct_bruto_A

    df_pareado_br = parear_eleicoes(df_1t_br, df_apurado_tempo_real)
    df_pareado_br = df_pareado_br.dropna(subset=['v_A_t0', 'log_odds_t0']).copy()

    print("Rodando Motor de Projeção Nacional 2026 + Cluster Bootstrap (Faixa de 95%)...")
    boot_res = cluster_bootstrap_municipio(df_pareado_br, n_boots=100)

    proj_A = boot_res['projecao_central']
    ic_A_inf = boot_res['ic_95_inferior']
    ic_A_sup = boot_res['ic_95_superior']

    proj_B = 100.0 - proj_A
    ic_B_inf = 100.0 - ic_A_sup
    ic_B_sup = 100.0 - ic_A_inf

    print("\n" + "=" * 72)
    print(f"Recebendo dados do TSE · {fmt_br(pct_secoes_atual, 2)}% das seções apuradas · atualizado às {hora_atualizacao}")
    print("[ projeção pouco confiável até 2,5% das seções ]")
    print("-" * 72)
    print(f"  — Lula PT   : {fmt_br(proj_A, 1)}%  | faixa de 95%: {fmt_br(ic_A_inf, 1)} – {fmt_br(ic_A_sup, 1)} | apurado agora: {fmt_br(pct_bruto_A, 1)}%")
    print(f"  — Flávio PL : {fmt_br(proj_B, 1)}%  | faixa de 95%: {fmt_br(ic_B_inf, 1)} – {fmt_br(ic_B_sup, 1)} | apurado agora: {fmt_br(pct_bruto_B, 1)}%")
    print("-" * 72)
    print(f"  (Gabarito 100% TSE: Lula {fmt_br(pct_A_real_br, 2)}% vs Flávio {fmt_br(100 - pct_A_real_br, 2)}%)")
    print("=" * 72)

    plotar_painel_apuracao_2026(
        df_2t_parcial=df_apurado_tempo_real,
        proj_A=proj_A,
        ic_A_inf=ic_A_inf,
        ic_A_sup=ic_A_sup,
        caminho_saida_png=ARQUIVO_PAINEL_PNG,
    )

