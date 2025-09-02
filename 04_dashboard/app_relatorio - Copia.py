# app_relatorio.py (VERSÃO COMPLETA E CORRIGIDA)

# --- 1. IMPORTAÇÕES ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import skbio.diversity
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.lines import Line2D

# --- 2. CONFIGURAÇÃO DA PÁGINA E FUNÇÕES ---
st.set_page_config(page_title="Relatório Biótico - OPYTA", page_icon="🔬", layout="wide")

@st.cache_data
def carregar_dados_consolidados():
    # (O código desta função permanece o mesmo - conecta e carrega os dados)
    load_dotenv()
    db_user=os.getenv('DB_USER'); db_password=os.getenv('DB_PASSWORD')
    db_host=os.getenv('DB_HOST'); db_name=os.getenv('DB_NAME')
    if not all([db_user, db_password, db_host, db_name]):
        st.error("Erro de Configuração: Variáveis de ambiente do banco de dados não encontradas. Verifique seu arquivo .env.")
        return pd.DataFrame()
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    try:
        engine = create_engine(db_url)
        query = text("""
            SELECT 
                t1.grupo_biologico, t1.nome_empresa, t1.nome_projeto, t1.codigo_opyta,
                t1.nome_campanha, t1.nome_ponto, t1.latitude, t1.longitude,
                t1.nome_cientifico, t1.contagem, t1.biomassa, t1.bmwp_score,
                t2.filo, t2.ordem, t2.familia, t2.origem, t2.nome_popular
            FROM public.biota_analise_consolidada AS t1
            LEFT JOIN public.especies AS t2 ON t1.nome_cientifico = t2.nome_cientifico
        """)
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
        return df
    except Exception as e:
        st.error(f"Erro de Conexão com o Banco de Dados: {e}")
        return pd.DataFrame()

def padronizar_dados(df):
    # (Esta função permanece a mesma)
    df_limpo = df.copy()
    mapa_nomes = {'nome_campanha': 'campanha', 'nome_ponto': 'ponto_amostral', 'contagem': 'densidade', 'nome_cientifico': 'especie'}
    df_limpo.rename(columns=mapa_nomes, inplace=True)
    if 'campanha' in df_limpo.columns:
        df_limpo['campanha'] = df_limpo['campanha'].str.replace('º', '', regex=False).str.replace('ª', '', regex=False)
    return df_limpo

def jackknife_1_estimator(matrix):
    # (Função do Bloco 9)
    S_obs = (matrix.sum(axis=0) > 0).sum()
    k = matrix.shape[0]
    if k == 0: return 0
    uniques = (matrix > 0).sum(axis=0)
    Q1 = (uniques == 1).sum()
    return S_obs + Q1 * ((k - 1) / k)

# --- 3. INÍCIO DO DASHBOARD ---
st.title("🔬 Relatório de Monitoramento Biótico - OPYTA")
df_completo = carregar_dados_consolidados()

# --- 4. BARRA LATERAL E FILTROS ---
st.sidebar.header("Filtros de Seleção")
if df_completo.empty:
    st.sidebar.warning("Nenhum dado carregado para filtrar.")
    df_final = df_completo
    grupo_selecionado = "Nenhum"
else:
    # (Filtros em cascata permanecem os mesmos)
    cliente_selecionado = st.sidebar.selectbox('1. Cliente:', sorted(df_completo['nome_empresa'].unique()))
    df_cliente = df_completo[df_completo['nome_empresa'] == cliente_selecionado]
    projeto_selecionado = st.sidebar.selectbox('2. Projeto:', sorted(df_cliente['nome_projeto'].unique()))
    df_projeto = df_cliente[df_cliente['nome_projeto'] == projeto_selecionado]
    grupos_disponiveis = sorted(df_projeto['grupo_biologico'].unique())
    grupo_selecionado = st.sidebar.selectbox('3. Grupo Biológico:', grupos_disponiveis)
    df_grupo = df_projeto[df_projeto['grupo_biologico'] == grupo_selecionado]
    campanhas_disponiveis = sorted(df_grupo['nome_campanha'].unique())
    campanha_selecionada = st.sidebar.multiselect('4. Campanha(s):', campanhas_disponiveis, default=campanhas_disponiveis)
    df_filtrado = df_grupo[df_grupo['nome_campanha'].isin(campanha_selecionada)]
    df_final = padronizar_dados(df_filtrado)

# --- 5. ÁREA PRINCIPAL DE CONTEÚDO ---
st.header(f"Análises para: {grupo_selecionado}")

if df_final.empty:
    st.warning("A seleção de filtros não retornou dados. Por favor, ajuste os filtros na barra lateral.")
else:
    with st.spinner("Gerando todas as análises..."):
        
        # O código das seções 1, 2 e 3 (Tabela de Composição, Riqueza, Densidade) permanece o mesmo.
        # Adicionei os títulos corretos e expandi o conteúdo.
        
        # --- ANÁLISES COMPLETAS ---

        with st.expander("Tabela de Composição Taxonômica", expanded=True):
            # Lógica do Bloco 3
            # ... (código já presente na versão anterior, sem alterações)
            df_especies = df_final.dropna(subset=['especie']).copy()
            campanhas_unicas = sorted(df_especies['campanha'].unique())
            mapa_campanhas = {nome: f"C{i+1}" for i, nome in enumerate(campanhas_unicas)}
            df_especies['campanha_curta'] = df_especies['campanha'].map(mapa_campanhas)
            tabela_campanhas = df_especies.groupby('especie')['campanha_curta'].unique().apply(lambda x: ' e '.join(sorted(x))).reset_index()
            tabela_campanhas.rename(columns={'campanha_curta': 'Campanha'}, inplace=True)
            atributos_especies = df_especies.drop_duplicates(subset='especie').copy()
            atributos_especies['Origem'] = atributos_especies['origem'].fillna('-')
            tabela_composicao = pd.merge(atributos_especies, tabela_campanhas, on='especie')
            colunas_finais_map = {'filo': 'Filo', 'ordem': 'Ordem', 'familia': 'Família', 'especie': 'Espécie', 'nome_popular': 'Nome Comum', 'Origem': 'Origem', 'Campanha': 'Campanha'}
            colunas_existentes = [col for col in colunas_finais_map.keys() if col in tabela_composicao.columns and tabela_composicao[col].notna().any()]
            tabela_composicao_final = tabela_composicao[colunas_existentes].rename(columns=colunas_finais_map)
            st.dataframe(tabela_composicao_final)

        with st.expander("Gráficos de Riqueza", expanded=True):
            # Lógica dos Blocos 4 e 5
            # ... (código já presente na versão anterior, sem alterações)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Riqueza por Ponto Amostral")
                df_riqueza_ponto = df_final.groupby(['ponto_amostral', 'campanha'])['especie'].nunique().reset_index()
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=df_riqueza_ponto, x='ponto_amostral', y='especie', hue='campanha', palette=["#007032", "#82C21F"], ax=ax)
                ax.set_ylabel("Riqueza de Espécies"); plt.xticks(rotation=45); st.pyplot(fig)
            with col2:
                st.subheader("Riqueza por Grupo Taxonômico")
                grupo_tax = 'filo' if 'filo' in df_final.columns and df_final['filo'].notna().any() else 'ordem'
                if grupo_tax in df_final.columns:
                    df_riqueza_tax = df_final.groupby([grupo_tax, 'campanha'])['especie'].nunique().reset_index()
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=df_riqueza_tax, x=grupo_tax, y='especie', hue='campanha', palette=["#007032", "#82C21F"], ax=ax)
                    ax.set_ylabel("Riqueza de Espécies"); plt.xticks(rotation=45); st.pyplot(fig)

        with st.expander("Gráficos de Densidade / Abundância", expanded=True):
            # Lógica do Bloco 6
            # ... (código já presente, mas agora com os 2 gráficos)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Por Ponto Amostral")
                df_densidade_ponto = df_final.groupby(['ponto_amostral', 'campanha'])['densidade'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=df_densidade_ponto, x='ponto_amostral', y='densidade', hue='campanha', palette=["#007032", "#82C21F"], ax=ax)
                ax.set_ylabel("Densidade / Abundância"); plt.xticks(rotation=45); st.pyplot(fig)
            with col2:
                st.subheader("Por Espécie")
                df_densidade_especie = df_final.groupby(['especie', 'campanha'])['densidade'].sum().reset_index()
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=df_densidade_especie, x='especie', y='densidade', hue='campanha', palette=["#007032", "#82C21F"], ax=ax)
                ax.set_ylabel("Densidade / Abundância"); plt.xticks(rotation=90); st.pyplot(fig)

        with st.expander("Índices de Diversidade Alfa (Shannon & Pielou)"):
            # Lógica do Bloco 8
            st.subheader("Diversidade por Ponto Amostral e Campanha")
            df_div = df_final.dropna(subset=['especie'])
            matriz_pontos = df_div.pivot_table(index='ponto_amostral', columns='especie', values='densidade', aggfunc='sum', fill_value=0)
            matriz_campanhas = df_div.pivot_table(index='campanha', columns='especie', values='densidade', aggfunc='sum', fill_value=0)
            matriz_total = pd.concat([matriz_pontos, matriz_campanhas])
            shannon_h = matriz_total.apply(lambda row: skbio.diversity.alpha.shannon(row, base=np.e), axis=1)
            pielou_j = matriz_total.apply(lambda row: skbio.diversity.alpha.pielou_e(row), axis=1)
            df_res_div = pd.DataFrame({'Diversidade': shannon_h, 'Equitabilidade': pielou_j}, index=matriz_total.index)
            fig, ax1 = plt.subplots(figsize=(15, 6))
            ax1.bar(df_res_div.index, df_res_div['Diversidade'], color="#007032", label='Diversidade')
            ax1.tick_params(axis='x', rotation=90)
            ax2 = ax1.twinx()
            ax2.plot(df_res_div.index, df_res_div['Equitabilidade'], marker='o', ls='None', color="#82C21F", label='Equitabilidade')
            st.pyplot(fig)
        
        with st.expander("Análise de Similaridade (Diversidade Beta)"):
            # Lógica do Bloco 9
            st.subheader("Dendrograma de Similaridade (Bray-Curtis)")
            matriz_sim = df_final.pivot_table(index='ponto_amostral', columns='especie', values='densidade', aggfunc='sum', fill_value=0)
            matriz_sim = matriz_sim.loc[matriz_sim.sum(axis=1) > 0]
            if not matriz_sim.empty:
                dist_bc = skbio.diversity.beta_diversity('braycurtis', matriz_sim.values, matriz_sim.index)
                linked = linkage(dist_bc.condensed_form(), method='average')
                fig, ax = plt.subplots(figsize=(10, 8))
                dendrogram(linked, orientation='left', labels=matriz_sim.index.tolist(), ax=ax, color_threshold=0, above_threshold_color='black')
                ax.set_title('Similaridade (Bray-Curtis)'); st.pyplot(fig)
            else:
                st.warning("Não há dados suficientes para a análise de similaridade.")

        with st.expander("Curva de Suficiência Amostral"):
            # Lógica do Bloco 10
            st.subheader("Curva do Coletor com Estimador Jackknife 1")
            matriz_suf = df_final.pivot_table(index=['campanha', 'ponto_amostral'], columns='especie', values='densidade', aggfunc='sum', fill_value=0)
            n_samples = matriz_suf.shape[0]
            if n_samples > 1:
                n_randomizations = 100 # Reduzido para performance no dashboard
                sobs_curves = np.zeros((n_randomizations, n_samples))
                sest_curves = np.zeros((n_randomizations, n_samples))
                for rand in range(n_randomizations):
                    shuffled_indices = np.random.permutation(n_samples)
                    shuffled_matrix = matriz_suf.iloc[shuffled_indices]
                    for i in range(1, n_samples + 1):
                        subset = shuffled_matrix.iloc[:i, :]
                        sobs_curves[rand, i-1] = (subset.sum(axis=0) > 0).sum()
                        sest_curves[rand, i-1] = jackknife_1_estimator(subset)
                mean_sobs = sobs_curves.mean(axis=0)
                mean_sest = sest_curves.mean(axis=0)
                std_sest = sest_curves.std(axis=0)
                fig, ax = plt.subplots(figsize=(15, 6))
                x_axis = np.arange(1, n_samples + 1)
                ax.plot(x_axis, mean_sobs, label='Curva de rarefação', color="#007032", lw=2.5)
                ax.plot(x_axis, mean_sest, label='Curva estimada', color="#82C21F", lw=2.5)
                ax.errorbar(x_axis, mean_sest, yerr=std_sest, fmt='none', ecolor='black', capsize=5)
                ax.set_xlabel('Número de unidades amostrais'); ax.set_ylabel('Riqueza'); st.pyplot(fig)
            else:
                st.warning("Não há amostras suficientes para a curva de suficiência.")
                
    st.success("Relatório gerado com sucesso!")
    st.balloons()
