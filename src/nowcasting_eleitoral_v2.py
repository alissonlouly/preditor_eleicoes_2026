# -*- coding: utf-8 -*-
"""
Pipeline de Projeção Eleitoral (TSE 2026 - 1º e 2º Turno) - Versão V2 (com V1 preservada)
Estrutura organizada:
1. Download (apenas na 1ª execução) e Processamento dos Dados do TSE de 2026 em `dados`
2. Modelos de Estimação:
   - V1 (`projetar_com_transicao_1t_2t_v1`): Modelo original (NNLS nacional puro abaixo de 5%)
   - V2 (`projetar_com_transicao_1t_2t_v2`): Modelo aprimorado com:
     (1) Pós-estratificação por UF + NNLS Regionalizado com Shrinkage Empírico-Bayesiano
     (2) WLS de Swing enriquecido com proporção de 3ª via (`prop_outros_t0`) e interação com `log_odds_t0`
     (3) Rampa híbrida ativa desde o início da apuração (< 5%)
     (4) Modelagem de Comparecimento Diferencial (Differential Turnout) por Região e Perfil da Seção
3. Painel Visual V2 redesenhado segundo boas práticas de Data Visualization (`painel_projecao_2026_v2.png`)
"""

import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import warnings

import duckdb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
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

ARQUIVO_PAINEL_V1_PNG = os.path.join(PASTA_RESULTADOS, 'painel_projecao_2026_v1.png')
ARQUIVO_PAINEL_V2_PNG = os.path.join(PASTA_RESULTADOS, 'painel_projecao_2026_v2_dashboard.png')
ARQUIVO_PAINEL_PNG = os.path.join(PASTA_RESULTADOS, 'painel_projecao_2026_v1.png')
ARQUIVO_RESULTADO_V2_TXT = os.path.join(PASTA_RESULTADOS, 'resultado_v2.txt')
ARQUIVO_CACHE_BOOT_V2 = os.path.join(PASTA_BASE, 'cache_boot_v2.json')

HORA_INICIO_APURACAO = '19:00:00'
LIMITE_MAX_PCT_2T = 2.5

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

MAPA_REGIAO = {
    'AC': 'Norte', 'AM': 'Norte', 'AP': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste',
    'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul',
    'ZZ': 'Exterior',
}


def obter_urls_bweb_2026(turno: str) -> dict[str, str]:
    """
    Consulta a API oficial do Portal de Dados Abertos do TSE (CKAN) para obter
    as URLs exatas dos arquivos `bweb_{turno}_{UF}_*.zip` das Eleições de 2026.
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
    """
    start_time = time.time()
    con = duckdb.connect()

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
        if os.path.exists(PASTA_CSV_2T):
            print("Excluindo arquivos brutos do restante do 2º Turno de 2026 para economizar espaço...")
            shutil.rmtree(PASTA_CSV_2T)
    else:
        print(f"--- [2026 - 2º Turno] Utilizando base já existente (até {LIMITE_MAX_PCT_2T}%): {ARQUIVO_PARQUET_2T} ---")

    con.close()
    print(f"Verificação/preparação dos dados de 2026 concluída em {time.time() - start_time:.1f}s!")


# ==============================================================================
# 2. MODELOS DE ESTIMAÇÃO: V1 (ORIGINAL) E V2 (APRIMORADO COM 4 MELHORIAS)
# ==============================================================================

def parear_eleicoes(df_base_t0: pd.DataFrame, df_atual_t1: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza a base do 1º Turno de 2026 (t0) com o 2º Turno de 2026 (t1) e
    adiciona variáveis de Macrorregião, proporção de 3ª via e comparecimento.
    """
    chaves_secao = ['SG_UF', 'CD_MUNICIPIO', 'NR_ZONA', 'NR_SECAO']

    pareado = pd.merge(
        df_base_t0, df_atual_t1, on=chaves_secao, how='left', suffixes=('_t0', '_t1')
    )

    pareado['apurada'] = pareado['total_validos_t1'].notna()

    # Razão logarítmica do 1º e 2º turno (com correção de Haldane +0.5)
    pareado['log_odds_t0'] = np.log((pareado['v_A_t0'] + 0.5) / (pareado['v_B_t0'] + 0.5))
    pareado['log_odds_t1'] = np.log((pareado['v_A_t1'] + 0.5) / (pareado['v_B_t1'] + 0.5))
    pareado['swing'] = pareado['log_odds_t1'] - pareado['log_odds_t0']

    # Variáveis adicionais para a V2
    pareado['regiao'] = pareado['SG_UF'].map(MAPA_REGIAO).fillna('Sudeste')
    pareado['prop_outros_t0'] = pareado['v_Outros_t0'] / np.maximum(1.0, pareado['total_validos_t0'])
    pareado['razao_comparecimento'] = pareado['total_validos_t1'] / np.maximum(1.0, pareado['total_validos_t0'])

    return pareado


def projetar_com_transicao_1t_2t_v1(df_pareado: pd.DataFrame) -> dict:
    """
    [MODELO V1 - CONGELADO]
    Modelo original: NNLS nacional sem pesos regionais abaixo de 5% (alpha = 0 aos 2,42%)
    e fator de comparecimento nacional único.
    """
    apuradas = df_pareado[df_pareado['apurada']].copy()
    pendentes = df_pareado[~df_pareado['apurada']].copy()

    pct_apurado = len(apuradas) / len(df_pareado)

    X_1t_apuradas = apuradas[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values
    X_1t_pendentes = pendentes[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values

    taxas_para_A, _ = nnls(X_1t_apuradas, apuradas['v_A_t1'].values)
    taxas_para_B, _ = nnls(X_1t_apuradas, apuradas['v_B_t1'].values)

    v_A_pred_transf = X_1t_pendentes @ taxas_para_A
    v_B_pred_transf = X_1t_pendentes @ taxas_para_B
    prop_A_transf = v_A_pred_transf / np.maximum(1.0, v_A_pred_transf + v_B_pred_transf)

    X_reg_treino = sm.add_constant(pd.get_dummies(apuradas['SG_UF'], drop_first=True, dtype=float))
    X_reg_treino['log_odds_t0'] = apuradas['log_odds_t0']

    modelo_wls = sm.WLS(apuradas['swing'], X_reg_treino, weights=apuradas['total_validos_t0']).fit()

    X_reg_pred = sm.add_constant(pd.get_dummies(pendentes['SG_UF'], drop_first=True, dtype=float))
    X_reg_pred['log_odds_t0'] = pendentes['log_odds_t0']
    X_reg_pred = X_reg_pred.reindex(columns=X_reg_treino.columns, fill_value=0.0)

    swing_wls = modelo_wls.predict(X_reg_pred).values

    peso_eb = len(apuradas) / (len(apuradas) + 80.0)
    swing_medio = np.average(apuradas['swing'], weights=apuradas['total_validos_t0'])
    swing_final = (peso_eb * swing_wls) + ((1 - peso_eb) * swing_medio)

    prop_A_reg = 1.0 / (1.0 + np.exp(-(pendentes['log_odds_t0'].values + swing_final)))

    alpha = np.clip((pct_apurado - 0.05) / (0.15 - 0.05), 0.0, 1.0)
    prop_A_combinada = (1.0 - alpha) * prop_A_transf + alpha * prop_A_reg

    fator_comparecimento = apuradas['total_validos_t1'].sum() / apuradas['total_validos_t0'].sum()
    total_pred_pendentes = pendentes['total_validos_t0'].values * fator_comparecimento
    v_A_pred_final = total_pred_pendentes * prop_A_combinada

    votos_A_nacional = apuradas['v_A_t1'].sum() + v_A_pred_final.sum()
    votos_totais_nacional = apuradas['total_validos_t1'].sum() + total_pred_pendentes.sum()

    return {
        'projecao_A_pct': (votos_A_nacional / votos_totais_nacional) * 100,
        'peso_regressao_alpha': alpha,
        'taxas_transferencia_A': taxas_para_A,
    }


def projetar_com_transicao_1t_2t_v2(df_pareado: pd.DataFrame) -> dict:
    """
    [MODELO V2 - APRIMORADO]
    Implementa as 4 melhorias estatísticas para corrigir o viés de seleção das primeiras urnas:
    1. Pós-estratificação por UF (IPW) + Transferência Ecológica (NNLS) estratificada por
       Macrorregião com Shrinkage Empírico-Bayesiano para a matriz nacional.
    2. Regressão WLS de Swing enriquecida com `prop_outros_t0` (fração de 3ª via no 1T) e
       sua interação com `log_odds_t0`, mais efeitos hierárquicos de Região + UF.
    3. Rampa de transição (`alpha`) ativa desde 0,5% das urnas (combinando NNLS Regional e WLS).
    4. Comparecimento Diferencial (Differential Turnout) estimado por Região, `log_odds_t0` e `prop_outros_t0`.
    """
    apuradas = df_pareado[df_pareado['apurada']].copy()
    pendentes = df_pareado[~df_pareado['apurada']].copy()

    pct_apurado = len(apuradas) / len(df_pareado)

    # 1A. Pesos de Pós-Estratificação por UF (IPW)
    votos_uf_nacional = df_pareado.groupby('SG_UF')['total_validos_t0'].sum()
    share_uf_nacional = votos_uf_nacional / votos_uf_nacional.sum()

    votos_uf_apuradas = apuradas.groupby('SG_UF')['total_validos_t0'].sum()
    share_uf_apuradas = votos_uf_apuradas / np.maximum(1.0, votos_uf_apuradas.sum())

    razao_ipw_uf = (share_uf_nacional / share_uf_apuradas.reindex(share_uf_nacional.index).fillna(0.0))
    razao_ipw_uf = razao_ipw_uf.replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.3, 4.0)

    pesos_ipw_apuradas = apuradas['SG_UF'].map(razao_ipw_uf).values
    pesos_wls_apuradas = np.maximum(1.0, apuradas['total_validos_t0'].values * pesos_ipw_apuradas)

    # 1B. Transferência Ecológica (NNLS) Regionalizada + Shrinkage
    raiz_w = np.sqrt(pesos_ipw_apuradas)[:, None]
    X_nac_ap = apuradas[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values * raiz_w
    taxas_nac_A, _ = nnls(X_nac_ap, apuradas['v_A_t1'].values * raiz_w.ravel())
    taxas_nac_B, _ = nnls(X_nac_ap, apuradas['v_B_t1'].values * raiz_w.ravel())

    v_A_pred_transf = np.zeros(len(pendentes), dtype=float)
    v_B_pred_transf = np.zeros(len(pendentes), dtype=float)
    X_1t_pendentes = pendentes[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values
    regioes_pendentes = pendentes['regiao'].values

    for regiao in ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul', 'Exterior']:
        mask_pend = regioes_pendentes == regiao
        if not np.any(mask_pend):
            continue

        ap_reg = apuradas[apuradas['regiao'] == regiao]
        n_reg = len(ap_reg)

        if n_reg >= 15:
            X_reg_ap = ap_reg[['v_A_t0', 'v_B_t0', 'v_Outros_t0']].values
            t_reg_A, _ = nnls(X_reg_ap, ap_reg['v_A_t1'].values)
            t_reg_B, _ = nnls(X_reg_ap, ap_reg['v_B_t1'].values)

            lambda_reg = n_reg / (n_reg + 120.0)
            taxas_A_final = lambda_reg * t_reg_A + (1.0 - lambda_reg) * taxas_nac_A
            taxas_B_final = lambda_reg * t_reg_B + (1.0 - lambda_reg) * taxas_nac_B
        else:
            taxas_A_final = taxas_nac_A
            taxas_B_final = taxas_nac_B

        v_A_pred_transf[mask_pend] = X_1t_pendentes[mask_pend] @ taxas_A_final
        v_B_pred_transf[mask_pend] = X_1t_pendentes[mask_pend] @ taxas_B_final

    prop_A_transf = v_A_pred_transf / np.maximum(1.0, v_A_pred_transf + v_B_pred_transf)

    # 2. Regressão WLS de Swing com `prop_outros_t0` e Interação
    dummies_reg_ap = pd.get_dummies(apuradas['regiao'], prefix='reg', drop_first=True, dtype=float)
    dummies_uf_ap = pd.get_dummies(apuradas['SG_UF'], prefix='uf', drop_first=True, dtype=float)

    X_reg_treino = pd.concat([dummies_reg_ap, dummies_uf_ap], axis=1)
    X_reg_treino['const'] = 1.0
    X_reg_treino['log_odds_t0'] = apuradas['log_odds_t0'].values
    X_reg_treino['prop_outros_t0'] = apuradas['prop_outros_t0'].values
    X_reg_treino['interacao_outros_odds'] = (
        apuradas['log_odds_t0'].values * apuradas['prop_outros_t0'].values
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        modelo_wls = sm.WLS(apuradas['swing'].values, X_reg_treino, weights=pesos_wls_apuradas).fit()

    dummies_reg_pend = pd.get_dummies(pendentes['regiao'], prefix='reg', drop_first=True, dtype=float)
    dummies_uf_pend = pd.get_dummies(pendentes['SG_UF'], prefix='uf', drop_first=True, dtype=float)

    X_reg_pred = pd.concat([dummies_reg_pend, dummies_uf_pend], axis=1)
    X_reg_pred['const'] = 1.0
    X_reg_pred['log_odds_t0'] = pendentes['log_odds_t0'].values
    X_reg_pred['prop_outros_t0'] = pendentes['prop_outros_t0'].values
    X_reg_pred['interacao_outros_odds'] = (
        pendentes['log_odds_t0'].values * pendentes['prop_outros_t0'].values
    )
    X_reg_pred = X_reg_pred.reindex(columns=X_reg_treino.columns, fill_value=0.0)

    swing_wls = np.clip(modelo_wls.predict(X_reg_pred).values, -0.8, 0.8)

    peso_eb = len(apuradas) / (len(apuradas) + 60.0)
    swing_medio = np.average(apuradas['swing'].values, weights=pesos_wls_apuradas)
    swing_final = (peso_eb * swing_wls) + ((1.0 - peso_eb) * swing_medio)

    prop_A_reg = 1.0 / (1.0 + np.exp(-(pendentes['log_odds_t0'].values + swing_final)))

    # 3. Rampa Híbrida Ativa desde 0,5% das Urnas
    alpha = float(np.clip((pct_apurado - 0.005) / (0.06 - 0.005), 0.35, 0.85))
    prop_A_combinada = (1.0 - alpha) * prop_A_transf + alpha * prop_A_reg

    # 4. Comparecimento Diferencial por Região e Perfil da Seção
    razao_comp_ap = np.clip(apuradas['razao_comparecimento'].values, 0.70, 1.30)
    X_turnout_ap = dummies_reg_ap.copy()
    X_turnout_ap['const'] = 1.0
    X_turnout_ap['log_odds_t0'] = apuradas['log_odds_t0'].values
    X_turnout_ap['prop_outros_t0'] = apuradas['prop_outros_t0'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod_turnout = sm.WLS(razao_comp_ap, X_turnout_ap, weights=pesos_wls_apuradas).fit()

    X_turnout_pend = dummies_reg_pend.copy()
    X_turnout_pend['const'] = 1.0
    X_turnout_pend['log_odds_t0'] = pendentes['log_odds_t0'].values
    X_turnout_pend['prop_outros_t0'] = pendentes['prop_outros_t0'].values
    X_turnout_pend = X_turnout_pend.reindex(columns=X_turnout_ap.columns, fill_value=0.0)

    fator_comp_wls = mod_turnout.predict(X_turnout_pend).values
    fator_comp_medio = np.average(razao_comp_ap, weights=pesos_wls_apuradas)
    fator_comp_final = np.clip(
        peso_eb * fator_comp_wls + (1.0 - peso_eb) * fator_comp_medio,
        0.80,
        1.20,
    )

    total_pred_pendentes = pendentes['total_validos_t0'].values * fator_comp_final
    v_A_pred_final = total_pred_pendentes * prop_A_combinada

    votos_A_nacional = apuradas['v_A_t1'].sum() + v_A_pred_final.sum()
    votos_totais_nacional = apuradas['total_validos_t1'].sum() + total_pred_pendentes.sum()

    return {
        'projecao_A_pct': (votos_A_nacional / votos_totais_nacional) * 100,
        'peso_regressao_alpha': alpha,
        'taxas_transferencia_A': taxas_nac_A,
    }


projetar_com_transicao_1t_2t = projetar_com_transicao_1t_2t_v2


def cluster_bootstrap_municipio(
    df_pareado: pd.DataFrame,
    n_boots: int = 200,
    func_modelo=projetar_com_transicao_1t_2t_v2,
) -> dict:
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
        res = func_modelo(df_temp)
        estimativas_boot.append(res['projecao_A_pct'])

    ic_inferior, ic_superior = np.percentile(estimativas_boot, [2.5, 97.5])
    ponto_central = func_modelo(df_pareado)['projecao_A_pct']

    return {
        'projecao_central': round(float(ponto_central), 2),
        'ic_95_inferior': round(float(ic_inferior), 2),
        'ic_95_superior': round(float(ic_superior), 2),
        'margem_erro_pp': round(float((ic_superior - ic_inferior) / 2), 2)
    }


# ==============================================================================
# 3. PAINEL VISUAL V2 (BOAS PRÁTICAS DE DATA VISUALIZATION / EDITORIAL DESIGN)
# ==============================================================================

def fmt_br(valor: float, casas: int = 1) -> str:
    """Formata número decimal com vírgula no padrão brasileiro."""
    return f"{valor:.{casas}f}".replace('.', ',')


def plotar_painel_apuracao_2026(
    df_1t_br: pd.DataFrame,
    df_2t_parcial: pd.DataFrame,
    proj_A: float,
    ic_A_inf: float,
    ic_A_sup: float,
    proj_A_v1: float,
    pct_A_real_100: float = 51.79,
    caminho_saida_png: str = 'painel_projecao_2026_v2.png',
) -> None:
    """
    Painel V2 projetado segundo boas práticas de Data Visualization (estilo NYT / Financial Times / Nexo):
    - Hierarquia visual clara, alto data-ink ratio e paleta acessível de alto contraste
    - Barra de Termômetro ("Cabo de Guerra" rumo aos 50% dos votos válidos)
    - Cards executivos com Projeção Central, Intervalo de Confiança 95%, Placar Bruto e Correção de Viés
    - Gráfico Esquerdo (Foco Empírico Real 0,1% a 2,42%): compara a evolução real da Projeção V2 (com banda
      de IC 95%) vs Placar Bruto Oficial TSE e linha de referência do Resultado Final (100%)
    - Gráfico Direito (Diagnóstico & Auditoria do Modelo): compara Placar Bruto, V1, V2 e Gabarito 100% TSE,
      além da sub/sobre-representação regional das urnas apuradas até 2,42%.
    """
    # Paleta Editorial Moderna (Slate + Crimson + Royal Blue)
    bg_pagina = '#F8FAFC'
    bg_card = '#FFFFFF'
    borda_card = '#E2E8F0'
    texto_primario = '#0F172A'
    texto_secundario = '#475569'
    texto_muted = '#64748B'

    cor_lula = '#D92D20'       # Crimson vibrante e legível
    cor_lula_light = '#FEE4E2'
    cor_flavio = '#1570EF'     # Royal Blue de alto contraste
    cor_flavio_light = '#D1E9FF'
    cor_ouro = '#0D9488'       # Teal para referência oficial 100%

    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'

    # 1. Preparação das séries reais acumuladas (0,05% até 2,42%)
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
    me_pp = (ic_A_sup - ic_A_inf) / 2.0

    ultimo_horario = pd.to_datetime(df_ord['horario_chegada'].iloc[-1])
    primeiro_horario = pd.to_datetime(df_ord['horario_chegada'].iloc[0])
    hora_str = ultimo_horario.strftime('%H:%M')
    hora_ini_str = primeiro_horario.strftime('%H:%M')

    # Calcula a curva real do Modelo V2 em marcos de 0,15% até pct_secoes_atual
    marcos_curva = [0.15, 0.30, 0.50, 0.75, 1.00, 1.30, 1.60, 1.90, 2.20, pct_secoes_atual]
    marcos_curva = sorted(list({round(m, 2) for m in marcos_curva if m <= pct_secoes_atual}))
    if marcos_curva[-1] != round(pct_secoes_atual, 2):
        marcos_curva.append(pct_secoes_atual)

    curva_v2_A = []
    curva_bruto_A = []
    largura_ic_marcos = []

    for m in marcos_curva:
        sub_2t = df_ord[df_ord['pct_urnas_acumulado'] <= m]
        bruto_m = (sub_2t['v_A'].sum() / sub_2t['total_validos'].sum()) * 100.0
        curva_bruto_A.append(bruto_m)

        if abs(m - pct_secoes_atual) < 1e-4:
            curva_v2_A.append(proj_A)
            largura_ic_marcos.append(me_pp)
        else:
            par_m = parear_eleicoes(df_1t_br, sub_2t).dropna(subset=['v_A_t0', 'log_odds_t0'])
            res_m = projetar_com_transicao_1t_2t_v2(par_m)
            curva_v2_A.append(float(res_m['projecao_A_pct']))
            # Escala teórica de erro ~ 1/sqrt(n) ancorada no Bootstrap real em 2,42%
            fator_n = np.sqrt(pct_secoes_atual / max(0.15, m))
            largura_ic_marcos.append(min(1.6, me_pp * fator_n))

    curva_v2_A = np.array(curva_v2_A)
    curva_v2_B = 100.0 - curva_v2_A
    curva_bruto_A = np.array(curva_bruto_A)
    curva_bruto_B = 100.0 - curva_bruto_A
    largura_ic_marcos = np.array(largura_ic_marcos)

    # Série contínua do placar bruto (filtrada a partir de 0,10% para evitar ruído de poucas urnas)
    df_ord_plot = df_ord[df_ord['pct_urnas_acumulado'] >= 0.10]
    idx_plot = np.unique(np.linspace(0, len(df_ord_plot) - 1, min(180, len(df_ord_plot)), dtype=int))
    x_bruto_cont = df_ord_plot['pct_urnas_acumulado'].iloc[idx_plot].values
    y_A_bruto_cont = df_ord_plot['pct_A_acum'].iloc[idx_plot].values
    y_B_bruto_cont = df_ord_plot['pct_B_acum'].iloc[idx_plot].values

    # 2. Layout Principal da Figura
    fig = plt.figure(figsize=(13.5, 9.6), facecolor=bg_pagina, dpi=130)
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[0.11, 0.27, 0.08, 0.54],
        width_ratios=[1.0, 1.0],
        hspace=0.16, wspace=0.05,
        left=0.035, right=0.965, top=0.955, bottom=0.045
    )

    # ==========================================================================
    # SEÇÃO 1: CABEÇALHO EDITORIAL + BARRA DE PROGRESSO DE APURAÇÃO
    # ==========================================================================
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis('off')

    ax_header.text(
        0.0, 0.80,
        "MONITOR DE PROJEÇÃO ELEITORAL (MODELO V2) · 2º TURNO PRESIDENCIAL 2026",
        fontsize=13.5, fontweight='bold', color=texto_primario, va='center'
    )
    ax_header.text(
        0.0, 0.32,
        f"Fonte: Boletins de Urna (TSE)  ·  Janela: {hora_ini_str} às {hora_str} (horário de Brasília)  ·  "
        f"{len(df_ord):,} seções apuradas ({fmt_br(pct_secoes_atual, 2)}% do total nacional)".replace(',', '.'),
        fontsize=10.0, color=texto_secundario, va='center'
    )

    # Badge de status no canto superior direito
    ax_header.text(
        0.995, 0.62,
        f"  ESTÁGIO INICIAL: {fmt_br(pct_secoes_atual, 2)}% APURADO  ",
        fontsize=9.5, fontweight='bold', color='#B45309', ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.5', facecolor='#FEF3C7', edgecolor='#F59E0B', lw=1.0)
    )

    # ==========================================================================
    # SEÇÃO 2: CARDS EXECUTIVOS DOS CANDIDATOS (LULA PT vs FLÁVIO PL)
    # ==========================================================================
    def desenhar_card_v2(ax, nome, partido, numero, cor, cor_suave, proj, ic_inf, ic_sup, apurado, gabarito, lider):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Fundo branco com borda sutil
        card = FancyBboxPatch(
            (0.005, 0.02), 0.99, 0.95,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            facecolor=bg_card, edgecolor=borda_card, linewidth=1.3
        )
        ax.add_patch(card)

        # Faixa superior de destaque na cor do candidato
        barra_topo = Rectangle((0.02, 0.92), 0.96, 0.045, facecolor=cor, edgecolor='none', zorder=3)
        ax.add_patch(barra_topo)

        # Nome, Partido e Tag de Projeção
        ax.text(0.05, 0.77, f"{nome}", fontsize=15, fontweight='bold', color=texto_primario, va='center')
        ax.text(0.19 if len(nome) <= 5 else 0.23, 0.77, f"{partido} · {numero}", fontsize=11, fontweight='bold', color=texto_muted, va='center')

        status_txt = "PROJEÇÃO V2 (LÍDER)" if lider else "PROJEÇÃO V2"
        ax.text(
            0.95, 0.77, f"  {status_txt}  ",
            fontsize=8.5, fontweight='bold', color=cor, ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.4', facecolor=cor_suave, edgecolor='none')
        )

        # Número Principal (Hero KPI) com 1 casa decimal + valor exato (2 casas) ao lado
        ax.text(0.05, 0.46, f"{fmt_br(proj, 1)}%", fontsize=34, fontweight='bold', color=cor, va='center')
        ax.text(0.37, 0.43, f"({fmt_br(proj, 2)}%)", fontsize=12.5, fontweight='bold', color=texto_muted, va='center')

        # Caixa de Intervalo de Confiança 95% à direita dentro do card
        ic_box = FancyBboxPatch(
            (0.57, 0.34), 0.38, 0.26,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=1.0
        )
        ax.add_patch(ic_box)
        ax.text(0.76, 0.52, "INTERVALO DE CONFIANÇA (95%)", fontsize=7.8, fontweight='bold', color=texto_muted, ha='center', va='center')
        ax.text(0.76, 0.41, f"{fmt_br(ic_inf, 1)}%  a  {fmt_br(ic_sup, 1)}%", fontsize=11.5, fontweight='bold', color=texto_primario, ha='center', va='center')

        # Linha divisória interna
        ax.plot([0.05, 0.95], [0.24, 0.24], color='#F1F5F9', lw=1.2)

        # Rodapé do Card: Apurado Bruto vs Correção do Modelo vs Gabarito 100%
        ajuste_modelo = proj - apurado
        sinal_ajuste = "+" if ajuste_modelo >= 0 else ""
        ax.text(
            0.05, 0.12,
            f"Apurado bruto agora: {fmt_br(apurado, 2)}%",
            fontsize=9.8, fontweight='bold', color=texto_secundario, va='center'
        )
        ax.text(
            0.50, 0.12,
            f"Correção V2: {sinal_ajuste}{fmt_br(ajuste_modelo, 2)} p.p.",
            fontsize=9.5, color=cor, fontweight='bold', va='center'
        )
        ax.text(
            0.95, 0.12,
            f"Real 100%: {fmt_br(gabarito, 2)}%",
            fontsize=9.5, color=texto_muted, ha='right', va='center'
        )

    ax_card_A = fig.add_subplot(gs[1, 0])
    desenhar_card_v2(
        ax_card_A, "Lula", "PT", 13, cor_lula, cor_lula_light,
        proj_A, ic_A_inf, ic_A_sup, apurado_A_agora, pct_A_real_100, lider=(proj_A >= proj_B)
    )

    ax_card_B = fig.add_subplot(gs[1, 1])
    desenhar_card_v2(
        ax_card_B, "Flávio", "PL", 22, cor_flavio, cor_flavio_light,
        proj_B, ic_B_inf, ic_B_sup, apurado_B_agora, 100.0 - pct_A_real_100, lider=(proj_B > proj_A)
    )

    # ==========================================================================
    # SEÇÃO 3: BARRA DE TERMÔMETRO ("CABO DE GUERRA" DOS VOTOS VÁLIDOS - 50%)
    # ==========================================================================
    ax_thermo = fig.add_subplot(gs[2, :])
    ax_thermo.set_xlim(0, 100)
    ax_thermo.set_ylim(0, 1)
    ax_thermo.axis('off')

    # Barras horizontais opostas encontrando-se na proporção projetada
    ax_thermo.barh([0.48], [proj_A], height=0.42, color=cor_lula, align='center', edgecolor='white', linewidth=1.5)
    ax_thermo.barh([0.48], [proj_B], left=[proj_A], height=0.42, color=cor_flavio, align='center', edgecolor='white', linewidth=1.5)

    # Marcador central de 50% (Maioria)
    ax_thermo.plot([50, 50], [0.12, 0.84], color=texto_primario, lw=2.2, linestyle='--', zorder=5)
    ax_thermo.text(50, 0.93, "▼ Maioria (50,0%)", fontsize=9.0, fontweight='bold', color=texto_primario, ha='center', va='bottom')

    ax_thermo.text(2.0, 0.48, f"Lula {fmt_br(proj_A, 2)}%", fontsize=10.5, fontweight='bold', color='white', va='center')
    ax_thermo.text(98.0, 0.48, f"{fmt_br(proj_B, 2)}% Flávio", fontsize=10.5, fontweight='bold', color='white', ha='right', va='center')

    # ==========================================================================
    # SEÇÃO 4A (ESQUERDA): EVOLUÇÃO REAL DA PROJEÇÃO V2 vs PLACAR BRUTO (0,1% a 2,42%)
    # ==========================================================================
    gs_bottom = gs[3, :].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.06)

    ax_left_card = fig.add_subplot(gs_bottom[0, 0])
    ax_left_card.set_xlim(0, 1)
    ax_left_card.set_ylim(0, 1)
    ax_left_card.axis('off')
    ax_left_card.add_patch(FancyBboxPatch(
        (0.005, 0.01), 0.99, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor=bg_card, edgecolor=borda_card, linewidth=1.3
    ))
    ax_left_card.text(
        0.035, 0.93,
        "Evolução em Tempo Real (0,1% a 2,42% das Seções): Projeção V2 vs. Placar Bruto TSE",
        fontsize=11.2, fontweight='bold', color=texto_primario, va='center'
    )
    ax_left_card.text(
        0.035, 0.875,
        "A linha sólida mostra a estabilidade da Projeção V2 (com banda de 95%) corrigindo o viés inicial do placar bruto (tracejado)",
        fontsize=8.8, color=texto_muted, va='center'
    )

    ax_evo = fig.add_axes([0.075, 0.11, 0.445, 0.315], facecolor='white')

    # Bandas de Intervalo de Confiança 95%
    ax_evo.fill_between(
        marcos_curva, curva_v2_A - largura_ic_marcos, curva_v2_A + largura_ic_marcos,
        color=cor_lula, alpha=0.13, label='IC 95% (Modelo V2)'
    )
    ax_evo.fill_between(
        marcos_curva, curva_v2_B - largura_ic_marcos, curva_v2_B + largura_ic_marcos,
        color=cor_flavio, alpha=0.13
    )

    # Linha de 50% e Gabarito Oficial 100%
    ax_evo.axhline(50.0, color='#94A3B8', linewidth=1.2, linestyle='-', zorder=2)
    ax_evo.axhline(pct_A_real_100, color=cor_lula, linewidth=1.1, linestyle=':', alpha=0.75, zorder=2)
    ax_evo.axhline(100.0 - pct_A_real_100, color=cor_flavio, linewidth=1.1, linestyle=':', alpha=0.75, zorder=2)

    # Curvas do Placar Bruto Oficial (tracejadas finas)
    ax_evo.plot(x_bruto_cont, y_A_bruto_cont, color=cor_lula, linestyle='--', linewidth=1.4, alpha=0.55, zorder=3)
    ax_evo.plot(x_bruto_cont, y_B_bruto_cont, color=cor_flavio, linestyle='--', linewidth=1.4, alpha=0.55, zorder=3)

    # Curvas da Projeção V2 (sólidas espessas com marcadores)
    ax_evo.plot(marcos_curva, curva_v2_A, color=cor_lula, linestyle='-', marker='o', markersize=4.5, linewidth=2.5, zorder=5)
    ax_evo.plot(marcos_curva, curva_v2_B, color=cor_flavio, linestyle='-', marker='o', markersize=4.5, linewidth=2.5, zorder=5)

    # Rótulos finais no ponto atual (2,42%)
    ax_evo.scatter([pct_secoes_atual], [proj_A], color=cor_lula, s=60, edgecolors='white', linewidth=1.5, zorder=6)
    ax_evo.scatter([pct_secoes_atual], [proj_B], color=cor_flavio, s=60, edgecolors='white', linewidth=1.5, zorder=6)

    ax_evo.annotate(
        f"V2: {fmt_br(proj_A, 2)}%\n(Real: {fmt_br(pct_A_real_100, 2)}%)",
        xy=(pct_secoes_atual, proj_A), xytext=(8, 4), textcoords='offset points',
        fontsize=8.8, fontweight='bold', color=cor_lula, va='center'
    )
    ax_evo.annotate(
        f"Bruto: {fmt_br(apurado_A_agora, 2)}%",
        xy=(pct_secoes_atual, apurado_A_agora), xytext=(8, -6), textcoords='offset points',
        fontsize=8.2, color=texto_secundario, va='center'
    )
    ax_evo.annotate(
        f"V2: {fmt_br(proj_B, 2)}%\n(Real: {fmt_br(100 - pct_A_real_100, 2)}%)",
        xy=(pct_secoes_atual, proj_B), xytext=(8, -4), textcoords='offset points',
        fontsize=8.8, fontweight='bold', color=cor_flavio, va='center'
    )

    ax_evo.set_xlim(0.10, pct_secoes_atual + 0.42)
    ax_evo.set_ylim(46.5, 53.5)
    xticks_evo = [0.2, 0.5, 1.0, 1.5, 2.0, round(pct_secoes_atual, 2)]
    ax_evo.set_xticks(xticks_evo)
    ax_evo.set_xticklabels([f"{fmt_br(x, 2 if x == round(pct_secoes_atual, 2) else 1)}%" for x in xticks_evo], fontsize=8.8, color=texto_secundario)
    ax_evo.set_yticks([47, 48, 49, 50, 51, 52, 53])
    ax_evo.set_yticklabels(['47%', '48%', '49%', '50%', '51%', '52%', '53%'], fontsize=8.8, color=texto_secundario)
    ax_evo.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')
    ax_evo.set_xlabel("% de Seções Totalizadas pelo TSE (Zoom 0,1% a 2,42%)", fontsize=9.2, fontweight='bold', color=texto_secundario, labelpad=6)

    for spine in ['top', 'right']:
        ax_evo.spines[spine].set_visible(False)

    leg_evo = [
        Line2D([0], [0], color=cor_lula, lw=2.5, marker='o', markersize=4, label='Projeção V2 (Lula)'),
        Line2D([0], [0], color=cor_flavio, lw=2.5, marker='o', markersize=4, label='Projeção V2 (Flávio)'),
        Line2D([0], [0], color='#64748B', lw=1.5, linestyle='--', label='Placar Bruto TSE'),
        Line2D([0], [0], color='#64748B', lw=1.2, linestyle=':', label='Gabarito 100% TSE'),
    ]
    ax_evo.legend(handles=leg_evo, loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.2, ncol=2)

    # ==========================================================================
    # SEÇÃO 4B (DIREITA): AUDITORIA DE PRECISÃO (BRUTO vs V1 vs V2 vs 100% TSE)
    # ==========================================================================
    ax_right_card = fig.add_subplot(gs_bottom[0, 1])
    ax_right_card.set_xlim(0, 1)
    ax_right_card.set_ylim(0, 1)
    ax_right_card.axis('off')
    ax_right_card.add_patch(FancyBboxPatch(
        (0.005, 0.01), 0.99, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor=bg_card, edgecolor=borda_card, linewidth=1.3
    ))
    ax_right_card.text(
        0.045, 0.93,
        "Auditoria de Precisão aos 2,42% Apurados (Lula %)",
        fontsize=11.2, fontweight='bold', color=texto_primario, va='center'
    )
    ax_right_card.text(
        0.045, 0.875,
        "Comparação direta com o resultado final oficial de 100% das urnas (51,79%)",
        fontsize=8.8, color=texto_muted, va='center'
    )

    ax_bar = fig.add_axes([0.65, 0.13, 0.285, 0.29], facecolor='white')

    categorias = [
        'Placar Bruto\n(2,42% TSE)',
        'Modelo V1\n(NNLS Nacional)',
        'Modelo V2\n(4 Melhorias)',
        'Resultado Final\n(100% TSE)',
    ]
    valores = [apurado_A_agora, proj_A_v1, proj_A, pct_A_real_100]
    erros = [v - pct_A_real_100 for v in valores]
    cores_barras = ['#94A3B8', '#F59E0B', cor_lula, cor_ouro]

    y_pos = np.arange(len(categorias))
    barras = ax_bar.barh(y_pos, valores, height=0.52, color=cores_barras, edgecolor='none', zorder=3)

    # Linha vertical cravada no Resultado 100% TSE (51,79%)
    ax_bar.axvline(pct_A_real_100, color=cor_ouro, linestyle='--', linewidth=1.5, zorder=4)

    for idx, (barra, val, err) in enumerate(zip(barras, valores, erros)):
        sinal = "+" if err >= 0 else ""
        erro_str = "Gabarito Oficial" if idx == 3 else f"Erro: {sinal}{fmt_br(err, 2)} p.p."
        ax_bar.text(
            val + 0.06, barra.get_y() + barra.get_height() / 2.0,
            f"{fmt_br(val, 2)}%  ({erro_str})",
            va='center', ha='left', fontsize=8.8,
            fontweight='bold' if idx in (2, 3) else 'normal',
            color=texto_primario if idx in (2, 3) else texto_secundario
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(categorias, fontsize=8.8, fontweight='bold', color=texto_secundario)
    ax_bar.set_xlim(50.0, 53.1)
    ax_bar.set_xticks([50.0, 50.5, 51.0, 51.5, 52.0, 52.5])
    ax_bar.set_xticklabels(['50,0%', '50,5%', '51,0%', '51,5%', '52,0%', '52,5%'], fontsize=8.2, color=texto_muted)
    ax_bar.grid(True, axis='x', linestyle='--', alpha=0.35, color='#CBD5E1')
    ax_bar.set_xlabel("Percentual de Votos Válidos — Lula (13)", fontsize=9.0, fontweight='bold', color=texto_secundario, labelpad=6)

    for spine in ['top', 'right']:
        ax_bar.spines[spine].set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    png_bytes = buf.getvalue()
    with open(caminho_saida_png, 'wb') as f_img:
        f_img.write(png_bytes)
    print(f"\nPainel visual V2 salvo em: {os.path.abspath(caminho_saida_png)}")
    plt.show(block=False)
    plt.pause(2)
    plt.close(fig)


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL (COMPARATIVO V1 vs V2 - ELEIÇÕES 2026)
# ==============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    preparar_dados()

    print("\nCarregando bases Parquet de 'dados'...")
    df_1t_br = pd.read_parquet(ARQUIVO_PARQUET_1T)
    df_2t_crono_br = pd.read_parquet(ARQUIVO_PARQUET_2T)

    pct_A_real_br = 51.79  # Resultado Oficial TSE 100% (64.256.232 vs 59.826.041)

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

    print("1) Avaliando Modelo V1 (Original)...")
    res_v1 = projetar_com_transicao_1t_2t_v1(df_pareado_br)
    proj_A_v1 = float(res_v1['projecao_A_pct'])

    print("2) Obtendo Projeção Modelo V2 (4 Melhorias) + Faixa de 95%...")
    if os.path.exists(ARQUIVO_CACHE_BOOT_V2):
        with open(ARQUIVO_CACHE_BOOT_V2, 'r', encoding='utf-8') as f_cache:
            boot_v2 = json.load(f_cache)
    else:
        boot_v2 = cluster_bootstrap_municipio(
            df_pareado_br, n_boots=100, func_modelo=projetar_com_transicao_1t_2t_v2
        )
        with open(ARQUIVO_CACHE_BOOT_V2, 'w', encoding='utf-8') as f_cache:
            json.dump(boot_v2, f_cache, indent=2)

    proj_A = boot_v2['projecao_central']
    ic_A_inf = boot_v2['ic_95_inferior']
    ic_A_sup = boot_v2['ic_95_superior']

    proj_B = 100.0 - proj_A
    ic_B_inf = 100.0 - ic_A_sup
    ic_B_sup = 100.0 - ic_A_inf

    resumo_v2 = (
        "========================================================================\n"
        f"Recebendo dados do TSE · {fmt_br(pct_secoes_atual, 2)}% das seções apuradas · atualizado às {hora_atualizacao}\n"
        "[ projeção pouco confiável até 2,5% das seções ]\n"
        "------------------------------------------------------------------------\n"
        f"  — Lula PT   : {fmt_br(proj_A, 1)}%  ({fmt_br(proj_A, 2)}%) | faixa de 95%: {fmt_br(ic_A_inf, 1)} – {fmt_br(ic_A_sup, 1)} | apurado agora: {fmt_br(pct_bruto_A, 1)}%\n"
        f"  — Flávio PL : {fmt_br(proj_B, 1)}%  ({fmt_br(proj_B, 2)}%) | faixa de 95%: {fmt_br(ic_B_inf, 1)} – {fmt_br(ic_B_sup, 1)} | apurado agora: {fmt_br(pct_bruto_B, 1)}%\n"
        "------------------------------------------------------------------------\n"
        f"  Comparativo aos {fmt_br(pct_secoes_atual, 2)}% das seções:\n"
        f"  • Placar Bruto TSE:  {fmt_br(pct_bruto_A, 2)}% (Erro: {pct_bruto_A - pct_A_real_br:+.2f} p.p.)\n"
        f"  • Modelo V1 (Antigo): {fmt_br(proj_A_v1, 2)}% (Erro: {proj_A_v1 - pct_A_real_br:+.2f} p.p.)\n"
        f"  • Modelo V2 (Novo):   {fmt_br(proj_A, 2)}% (Erro: {proj_A - pct_A_real_br:+.2f} p.p.)\n"
        f"  • Resultado 100% TSE: {fmt_br(pct_A_real_br, 2)}%\n"
        "========================================================================\n"
    )
    print("\n" + resumo_v2)

    with open(ARQUIVO_RESULTADO_V2_TXT, 'w', encoding='utf-8') as f_out:
        f_out.write(resumo_v2)

    plotar_painel_apuracao_2026(
        df_1t_br=df_1t_br,
        df_2t_parcial=df_apurado_tempo_real,
        proj_A=proj_A,
        ic_A_inf=ic_A_inf,
        ic_A_sup=ic_A_sup,
        proj_A_v1=proj_A_v1,
        pct_A_real_100=pct_A_real_br,
        caminho_saida_png=ARQUIVO_PAINEL_V2_PNG,
    )
