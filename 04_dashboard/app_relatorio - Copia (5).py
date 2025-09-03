# app_relatorio.py (VERSÃO FINAL COM CORREÇÃO DE DADOS QUALITATIVOS)

# --- 1. IMPORTAÇÕES E 2. FUNÇÕES ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import skbio.diversity
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

st.set_page_config(page_title="Relatório Biótico - OPYTA", page_icon="🔬", layout="wide")
plt.rcParams['font.family'] = 'Arial'

@st.cache_data
def carregar_dados_consolidados():
    load_dotenv()
    db_user=os.getenv('DB_USER'); db_password=os.getenv('DB_PASSWORD')
    db_host=os.getenv('DB_HOST'); db_name=os.getenv('DB_NAME')
    if not all([db_user, db_password, db_host, db_name]):
        st.error("Erro de Configuração: Variáveis de ambiente do banco de dados não encontradas."); return pd.DataFrame()
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    try:
        engine = create_engine(db_url)
        query = text("""
            SELECT 
                t1.grupo_biologico, t1.nome_empresa, t1.nome_projeto, t1.codigo_opyta,
                t1.nome_campanha, t1.nome_ponto, t1.latitude, t1.longitude,
                t1.nome_cientifico, t1.contagem, t1.biomassa, t1.bmwp_score,
                t1.tipo_amostragem,
                t2.filo, t2.ordem, t2.familia, t2.origem, t2.nome_popular
            FROM public.biota_analise_consolidada AS t1
            LEFT JOIN public.especies AS t2 ON t1.nome_cientifico = t2.nome_cientifico
        """)
        with engine.connect() as connection: df = pd.read_sql_query(query, connection)
        return df
    except Exception as e:
        st.error(f"Erro de Conexão com o Banco de Dados: {e}"); return pd.DataFrame()

def padronizar_dados(df):
    df_limpo = df.copy(); mapa_nomes = {'nome_campanha': 'campanha', 'nome_ponto': 'ponto_amostral', 'contagem': 'densidade', 'nome_cientifico': 'especie'}; df_limpo.rename(columns=mapa_nomes, inplace=True)
    if 'campanha' in df_limpo.columns: df_limpo['campanha'] = df_limpo['campanha'].str.replace('º', '', regex=False).str.replace('ª', '', regex=False)
    return df_limpo

def jackknife_1_estimator(matrix):
    S_obs = (matrix.sum(axis=0) > 0).sum(); k = matrix.shape[0]
    if k == 0: return 0
    uniques = (matrix > 0).sum(axis=0); Q1 = (uniques == 1).sum()
    return S_obs + Q1 * ((k - 1) / k)

# --- 3. INÍCIO DO DASHBOARD E 4. FILTROS ---
st.title("🔬 Relatório de Monitoramento Biótico - OPYTA")
df_completo = carregar_dados_consolidados()
st.sidebar.header("Filtros de Seleção")
if df_completo.empty:
    st.sidebar.warning("Nenhum dado carregado para filtrar."); df_final = df_completo; grupo_selecionado = "Nenhum"
else:
    cliente_selecionado = st.sidebar.selectbox('1. Cliente:', sorted(df_completo['nome_empresa'].unique())); df_cliente = df_completo[df_completo['nome_empresa'] == cliente_selecionado]
    projeto_selecionado = st.sidebar.selectbox('2. Projeto:', sorted(df_cliente['nome_projeto'].unique())); df_projeto = df_cliente[df_cliente['nome_projeto'] == projeto_selecionado]
    grupos_disponiveis = sorted(df_projeto['grupo_biologico'].unique()); grupo_selecionado = st.sidebar.selectbox('3. Grupo Biológico:', grupos_disponiveis); df_grupo = df_projeto[df_projeto['grupo_biologico'] == grupo_selecionado]
    campanhas_disponiveis = sorted(df_grupo['nome_campanha'].unique()); campanha_selecionada = st.sidebar.multiselect('4. Campanha(s):', campanhas_disponiveis, default=campanhas_disponiveis)
    df_filtrado = df_grupo[df_grupo['nome_campanha'].isin(campanha_selecionada)]; df_final = padronizar_dados(df_filtrado)

# --- 5. ÁREA PRINCIPAL DE CONTEÚDO ---
st.header(f"Análises para: {grupo_selecionado}")
if df_final.empty:
    st.warning("A seleção de filtros não retornou dados.")
else:
    with st.spinner("Gerando todas as análises..."):
        
        with st.expander("Tabela de Composição Taxonômica", expanded=True):
            df_especies = df_final.dropna(subset=['especie']).copy(); campanhas_unicas = sorted(df_especies['campanha'].unique()); mapa_campanhas = {nome: f"C{i+1}" for i, nome in enumerate(campanhas_unicas)}; df_especies['campanha_curta'] = df_especies['campanha'].map(mapa_campanhas); tabela_campanhas = df_especies.groupby('especie')['campanha_curta'].unique().apply(lambda x: ' e '.join(sorted(x))).reset_index(name='Campanha'); atributos_especies = df_especies.drop_duplicates(subset='especie').copy(); atributos_especies['Origem'] = atributos_especies['origem'].fillna('-'); tabela_composicao = pd.merge(atributos_especies, tabela_campanhas, on='especie'); colunas_finais_map = {'filo': 'Filo', 'ordem': 'Ordem', 'familia': 'Família', 'especie': 'Espécie', 'nome_popular': 'Nome Comum', 'Origem': 'Origem', 'Campanha': 'Campanha'}; colunas_existentes = [col for col in colunas_finais_map.keys() if col in tabela_composicao.columns and tabela_composicao[col].notna().any()]
            if colunas_existentes:
                tabela_selecionada = tabela_composicao[colunas_existentes]; tabela_renomeada = tabela_selecionada.rename(columns=colunas_finais_map); coluna_para_ordenar = colunas_finais_map[colunas_existentes[0]]; tabela_composicao_final = tabela_renomeada.sort_values(by=coluna_para_ordenar); st.dataframe(tabela_composicao_final, use_container_width=True)
            else: st.warning("Não há colunas de taxonomia para exibir.")

        with st.expander("Tabela de Ocorrência Detalhada"):
            st.subheader("Ocorrência e Densidade por Ponto Amostral (com 'X' para qualitativos)"); campanhas_ordenadas = sorted(df_final['campanha'].unique()); pontos_ordenados = sorted(df_final['ponto_amostral'].unique()); especies_ordenadas = sorted(df_final['especie'].dropna().unique()); lista_tabelas_por_campanha = []
            for campanha in campanhas_ordenadas:
                df_campanha_atual = df_final[df_final['campanha'] == campanha]; tabela_base = pd.DataFrame(index=especies_ordenadas, columns=pontos_ordenados); df_quant = df_campanha_atual[df_campanha_atual['tipo_amostragem'] == 'Quantitativo']; matriz_quant = df_quant.pivot_table(index='especie', columns='ponto_amostral', values='densidade', aggfunc='sum'); 
                df_qual = df_campanha_atual[df_campanha_atual['tipo_amostragem'].isin(['Qualitativo', 'Qualitativa'])]; 
                matriz_qual = df_qual.pivot_table(index='especie', columns='ponto_amostral', values='densidade', aggfunc=lambda x: 'X'); tabela_pivot_completa = tabela_base.fillna(matriz_qual); tabela_pivot_completa.fillna(matriz_quant, inplace=True); total_pontos = len(pontos_ordenados); tabela_pivot_completa['OC'] = tabela_pivot_completa[pontos_ordenados].notna().sum(axis=1)
                if total_pontos > 0: tabela_pivot_completa['%OC'] = (tabela_pivot_completa['OC'] / total_pontos) * 100
                else: tabela_pivot_completa['%OC'] = 0
                densidade_total = tabela_pivot_completa[pontos_ordenados].apply(pd.to_numeric, errors='coerce').sum(axis=0); densidade_total.name = 'Densidade Total'; riqueza = tabela_pivot_completa[pontos_ordenados].notna().sum(axis=0); riqueza.name = 'Riqueza'; tabela_campanha_final = pd.concat([tabela_pivot_completa.astype(object), densidade_total.to_frame().T.astype(object), riqueza.to_frame().T.astype(object)]); lista_tabelas_por_campanha.append(tabela_campanha_final)
            if lista_tabelas_por_campanha:
                tabela_ocorrencia = pd.concat(lista_tabelas_por_campanha, axis=1, keys=campanhas_ordenadas); tabela_ocorrencia.index.name = 'Espécie'; tabela_ocorrencia = tabela_ocorrencia.astype(object).fillna('')
                for campanha in campanhas_ordenadas:
                    col_oc_percent = (campanha, '%OC')
                    if col_oc_percent in tabela_ocorrencia.columns: tabela_ocorrencia[col_oc_percent] = tabela_ocorrencia[col_oc_percent].apply(lambda x: f"{float(x):.0f}%" if x != '' else '')
                st.dataframe(tabela_ocorrencia, use_container_width=True)

        with st.expander("Tabela de Ocorrência de Cianobactérias"):
            filo_alvo = 'CYANOBACTERIA'
            df_ciano = df_final[df_final['filo'] == filo_alvo].copy()
            if df_ciano.empty:
                st.info(f"Nenhum registro encontrado para o filo '{filo_alvo}' na seleção atual.")
            else:
                campanhas_ordenadas = sorted(df_final['campanha'].unique()); pontos_ordenados = sorted(df_final['ponto_amostral'].unique()); especies_ordenadas_ciano = sorted(df_ciano['especie'].dropna().unique()); lista_tabelas_por_campanha = []
                for campanha in campanhas_ordenadas:
                    df_campanha_atual = df_ciano[df_ciano['campanha'] == campanha]; df_quant = df_campanha_atual[df_campanha_atual['tipo_amostragem'] == 'Quantitativo']; matriz_quant = df_quant.pivot_table(index='especie', columns='ponto_amostral', values='densidade', aggfunc='sum'); 
                    df_qual = df_campanha_atual[df_campanha_atual['tipo_amostragem'].isin(['Qualitativo', 'Qualitativa'])]; 
                    matriz_qual = df_qual.pivot_table(index='especie', columns='ponto_amostral', values='densidade', aggfunc=lambda x: 'X'); tabela_pivot_completa = matriz_quant.reindex(index=especies_ordenadas_ciano, columns=pontos_ordenados); tabela_pivot_completa.fillna(matriz_qual, inplace=True); total_pontos = len(pontos_ordenados); tabela_pivot_completa['OC'] = tabela_pivot_completa[pontos_ordenados].notna().sum(axis=1)
                    if total_pontos > 0: tabela_pivot_completa['%OC'] = (tabela_pivot_completa[pontos_ordenados].notna().sum(axis=1) / total_pontos) * 100
                    else: tabela_pivot_completa['%OC'] = 0
                    densidade_total = tabela_pivot_completa[pontos_ordenados].apply(pd.to_numeric, errors='coerce').sum(axis=0); densidade_total.name = 'Abundância'; riqueza = tabela_pivot_completa[pontos_ordenados].notna().sum(axis=0); riqueza.name = 'Riqueza'; tabela_campanha_final = pd.concat([tabela_pivot_completa.astype(object), densidade_total.to_frame().T.astype(object), riqueza.to_frame().T.astype(object)]); lista_tabelas_por_campanha.append(tabela_campanha_final)
                tabela_final_ciano = pd.concat(lista_tabelas_por_campanha, axis=1, keys=campanhas_ordenadas); tabela_final_ciano.index.name = 'Táxon'; tabela_final_ciano = tabela_final_ciano.astype(object).fillna('')
                for col in tabela_final_ciano.columns:
                    if col[1] not in ['OC', '%OC', 'Táxon']: tabela_final_ciano[col] = tabela_final_ciano[col].apply(lambda x: f'{x:,.2f}'.replace('.',',') if isinstance(x, (int, float)) else x)
                for campanha in campanhas_ordenadas:
                    col_oc_percent = (campanha, '%OC')
                    if col_oc_percent in tabela_final_ciano.columns: tabela_final_ciano[col_oc_percent] = tabela_final_ciano[col_oc_percent].apply(lambda x: f"{float(x):.0f}%" if x != '' else '')
                st.dataframe(tabela_final_ciano, use_container_width=True)

        with st.expander("Tabela de Espécies Indicadoras de Qualidade Ambiental"):
            dados_indicadoras = [ {'Filo': 'Bacillariophyta', 'Espécie': 'Achnanthidium minutissimum', 'Indicação': 'Altamente tolerante a contaminação antiga e/ou recente por metais.', 'Referência': 'Cantonati, et al., 2014'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Cyclotella meneghiniana', 'Indicação': 'Ambientes eutrofizados. Comum em ambientes rasos, turvos, enriquecidos por nutrientes e favorecidos por condições moderadamente alcalinas.', 'Referência': 'Bilous, et al., 2021; Dantas et al., 2008'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Eunotia zygodon', 'Indicação': 'Ambientes mesotróficos e ácidos.', 'Referência': 'Costa, 2015; Eurey, 2008'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Frustulia sp.', 'Indicação': 'Ambientes ácidos, gênero comum em riachos de cabeceira.', 'Referência': 'Blinn e Poff, 2005'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Gomphonema gracile', 'Indicação': 'Ambientes de pH neutros e mesotróficos.', 'Referência': 'Van Dam et al., 1994'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Iconella linearis', 'Indicação': 'Preferência por ambientes não poluídos ou pouco poluídos, presentes em águas oligo-mesotróficas.', 'Referência': 'Niyatbekov and Barinova, 2018'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Iconella tenera', 'Indicação': 'Preferência por ambientes mais alcalinos, de águas com menor correnteza, não poluídos e oligotróficos.', 'Referência': 'Niyatbekov and Barinova, 2018'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Navicula sp.', 'Indicação': 'Ambientes oligo-mesotróficos.', 'Referência': 'Cordeiro-Araújo, 2010'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Pinnularia sp.', 'Indicação': 'Ambientes ácidos e oligotróficos.', 'Referência': 'Pereira et al., 2012'}, {'Filo': 'Bacillariophyta', 'Espécie': 'Ulnaria ulna', 'Indicação': 'Baixa a média tolerância a eutrofização.', 'Referência': 'Guimarães e Garcia, 2016'}, {'Filo': 'Charophyta', 'Espécie': 'Mougeotia sp.', 'Indicação': 'Ocorre em grande maioria (mas não exclusivamente) em ambientes oligotróficos. Favorecida em ambientes mais ácidos.', 'Referência': 'Zohary, et al., 2018'}, {'Filo': 'Euglenophyta', 'Espécie': 'Trachelomonas volvocina', 'Indicação': 'Ambientes de pH ácido a neutros, ricos em ferro e manganês e matéria orgânica.', 'Referência': 'Zohary, et al., 2018'} ]; df_referencia = pd.DataFrame(dados_indicadoras)
            especies_encontradas = set(df_final['especie'].unique()); tabela_indicadoras_final = df_referencia[df_referencia['Espécie'].isin(especies_encontradas)].copy()
            if tabela_indicadoras_final.empty:
                st.info("Nenhuma das espécies indicadoras de referência foi encontrada na seleção atual.")
            else:
                tabela_indicadoras_final.sort_values(by=['Filo', 'Espécie'], inplace=True); st.dataframe(tabela_indicadoras_final, use_container_width=True)

        with st.expander("Gráficos de Riqueza", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Riqueza por Ponto Amostral"); df_riqueza_ponto = df_final.groupby(['ponto_amostral', 'campanha'])['especie'].nunique().reset_index(); fig, ax = plt.subplots(figsize=(12, 8)); sns.set_theme(style="white", context="talk"); sns.barplot(data=df_riqueza_ponto, x='ponto_amostral', y='especie', hue='campanha', palette=["#007032", "#82C21F"], ax=ax, width=0.8); ax.set_ylabel("Número de Espécies (Riqueza)", fontsize=14); ax.set_xlabel("Ponto Amostral", fontsize=14); ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', direction='out', left=True, labelsize=12); ax.grid(False); sns.despine(ax=ax);
                for container in ax.containers: ax.bar_label(container, fmt='%d', fontsize=12)
                if ax.get_legend() is not None: ax.get_legend().remove()
                handles, labels = ax.get_legend_handles_labels(); novos_labels = [l.replace('1 ','1ª ',1).replace('2 ','2ª ',1) for l in labels]; fig.legend(handles, novos_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(novos_labels), frameon=False, fontsize=12); plt.tight_layout(rect=[0, 0.05, 1, 1]); st.pyplot(fig)
            with col2:
                st.subheader("Riqueza por Grupo Taxonômico"); grupo_tax = 'filo' if 'filo' in df_final.columns and df_final['filo'].notna().any() else 'ordem'
                if grupo_tax in df_final.columns and df_final[grupo_tax].notna().any():
                    df_riqueza_tax = df_final.groupby([grupo_tax, 'campanha'])['especie'].nunique().reset_index(); total_riqueza_tax = df_final.groupby(grupo_tax)['especie'].nunique().sort_values(ascending=False).index; fig, ax = plt.subplots(figsize=(12, 8)); sns.set_theme(style="white", context="talk"); sns.barplot(data=df_riqueza_tax, x=grupo_tax, y='especie', hue='campanha', palette=["#007032", "#82C21F"], ax=ax, order=total_riqueza_tax, width=0.8); ax.set_ylabel("Número de Espécies (Riqueza)", fontsize=14); ax.set_xlabel(grupo_tax.title(), fontsize=14); plt.xticks(rotation=45, ha='right'); ax.tick_params(axis='y', direction='out', left=True, labelsize=12); ax.grid(False); sns.despine();
                    for container in ax.containers: ax.bar_label(container, fmt='%d', fontsize=12)
                    if ax.get_legend() is not None: ax.get_legend().remove()
                    handles, labels = ax.get_legend_handles_labels(); novos_labels = [l.replace('1 ','1ª ',1).replace('2 ','2ª ',1) for l in labels]; fig.legend(handles, novos_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(novos_labels), frameon=False, fontsize=12); plt.tight_layout(rect=[0, 0.05, 1, 1]); st.pyplot(fig)

        with st.expander("Gráficos de Densidade / Abundância", expanded=False):
            df_quant = df_final[df_final['tipo_amostragem'] == 'Quantitativo'].copy()
            if df_quant.empty or df_quant['densidade'].sum() == 0:
                st.warning("Nenhum dado quantitativo com densidade > 0 foi encontrado para esta análise.")
            else:
                df_densidade_filo = df_quant.groupby(['campanha', 'ponto_amostral', 'filo'])['densidade'].sum().reset_index(); pivot_abs = df_densidade_filo.pivot_table(index=['campanha', 'ponto_amostral'], columns='filo', values='densidade', fill_value=0); pivot_rel = pivot_abs.div(pivot_abs.sum(axis=1), axis=0) * 100; pivot_rel.fillna(0, inplace=True)
                cores_principais = ['#3E7369', '#007032', '#00A859', '#82C21F']; n_filos = len(pivot_abs.columns); cores = cores_principais[:n_filos] if n_filos <= len(cores_principais) else [mcolors.LinearSegmentedColormap.from_list("paleta", cores_principais, N=n_filos)(i) for i in range(n_filos)]
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Densidade Absoluta por Ponto"); fig1, ax1 = plt.subplots(figsize=(12, 8)); sns.set_theme(style="white", context="talk"); pivot_abs.plot(kind='bar', stacked=True, ax=ax1, color=cores, width=0.8); ax1.set_ylabel('Densidade (cél/mL)', fontsize=14); ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax1.tick_params(axis='y', labelsize=12, direction='out', left=True); ax1.set_xlabel(None); ax1.set_xticklabels([p[1] for p in pivot_abs.index], rotation=45, ha='right', fontsize=12); ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False, fontsize=12, title=None); ax1.grid(False); sns.despine(ax=ax1); espessura_eixo = ax1.spines['bottom'].get_linewidth(); boundaries = [0]; campanhas = pivot_abs.index.get_level_values('campanha')
                    for i in range(1, len(campanhas)):
                        if campanhas[i] != campanhas[i-1]:
                            ax1.plot([i - 0.5, i - 0.5], [0, -0.10], color='black', linewidth=espessura_eixo * 1.2, transform=ax1.get_xaxis_transform(), clip_on=False); boundaries.append(i)
                    boundaries.append(len(campanhas))
                    for i in range(len(boundaries) - 1):
                        start, end = boundaries[i], boundaries[i+1]; mid = start + (end-start)/2 - 0.5; label = campanhas[start].replace('1 ','1ª ',1).replace('2 ','2ª ',1); ax1.text(mid, -0.25, label, ha='center', transform=ax1.get_xaxis_transform(), fontsize=14)
                    plt.tight_layout(rect=[0, 0.1, 1, 1]); st.pyplot(fig1)
                with col2:
                    st.subheader("Densidade Relativa por Ponto"); fig2, ax2 = plt.subplots(figsize=(12, 8)); sns.set_theme(style="white", context="talk"); pivot_rel.plot(kind='bar', stacked=True, ax=ax2, color=cores, width=0.8); ax2.set_ylabel('Densidade Relativa (%)', fontsize=14); ax2.yaxis.set_major_formatter(mticker.PercentFormatter()); ax2.tick_params(axis='y', labelsize=12, direction='out', left=True); ax2.set_xlabel(None); ax2.set_xticklabels([p[1] for p in pivot_rel.index], rotation=45, ha='right', fontsize=12); ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False, fontsize=12, title=None); ax2.grid(False); sns.despine(ax=ax2); espessura_eixo = ax2.spines['bottom'].get_linewidth(); boundaries = [0]; campanhas = pivot_rel.index.get_level_values('campanha')
                    for i in range(1, len(campanhas)):
                        if campanhas[i] != campanhas[i-1]:
                            ax2.plot([i - 0.5, i - 0.5], [0, -0.10], color='black', linewidth=espessura_eixo * 1.2, transform=ax2.get_xaxis_transform(), clip_on=False); boundaries.append(i)
                    boundaries.append(len(campanhas))
                    for i in range(len(boundaries) - 1):
                        start, end = boundaries[i], boundaries[i+1]; mid = start + (end-start)/2 - 0.5; label = campanhas[start].replace('1 ','1ª ',1).replace('2 ','2ª ',1); ax2.text(mid, -0.25, label, ha='center', transform=ax2.get_xaxis_transform(), fontsize=14)
                    plt.tight_layout(rect=[0, 0.1, 1, 1]); st.pyplot(fig2)

        with st.expander("Visualização Hierárquica (Sunburst)", expanded=False):
            st.subheader("Distribuição da Riqueza por Grupo Taxonômico"); grupo_tax_sunburst = 'filo' if 'filo' in df_final.columns and df_final['filo'].notna().any() else 'ordem'
            if grupo_tax_sunburst in df_final.columns and df_final[grupo_tax_sunburst].notna().any():
                sunburst_data = df_final.dropna(subset=[grupo_tax_sunburst, 'especie']).copy(); total_riqueza = sunburst_data['especie'].nunique(); centro_label = f"Total ({total_riqueza} spp.)"; sunburst_data = sunburst_data.groupby([grupo_tax_sunburst])['especie'].nunique().reset_index(name='Riqueza'); sunburst_data['Centro'] = centro_label; fig_sunburst = px.sunburst(sunburst_data, path=['Centro', grupo_tax_sunburst], values='Riqueza', color=grupo_tax_sunburst, color_discrete_sequence=px.colors.qualitative.Pastel); fig_sunburst.update_layout(title_text='Riqueza de Espécies', title_x=0.5); fig_sunburst.update_traces(textinfo='label+percent root', insidetextorientation='radial', hovertemplate='<b>%{label}</b><br>Riqueza: %{value} espécies<br>Proporção da Riqueza Total: %{percentRoot:.1%}'); st.plotly_chart(fig_sunburst, use_container_width=True)
            else: st.warning("Não há dados taxonômicos (Filo/Ordem) para gerar o gráfico Sunburst.")

        with st.expander("Índices de Diversidade Alfa (Shannon & Pielou)", expanded=False):
            st.subheader("Diversidade por Ponto Amostral e Campanha"); df_div = df_final.dropna(subset=['especie']); matriz_pontos = df_div.pivot_table(index='ponto_amostral', columns='especie', values='densidade', aggfunc='sum', fill_value=0); matriz_campanhas = df_div.pivot_table(index='campanha', columns='especie', values='densidade', aggfunc='sum', fill_value=0); matriz_total = pd.concat([matriz_pontos, matriz_campanhas]); shannon_h = matriz_total.apply(lambda row: skbio.diversity.alpha.shannon(row, base=np.e), axis=1); pielou_j = matriz_total.apply(lambda row: skbio.diversity.alpha.pielou_e(row), axis=1); df_res_div = pd.DataFrame({'Diversidade': shannon_h, 'Equitabilidade': pielou_j}, index=matriz_total.index); fig, ax1 = plt.subplots(figsize=(15, 8)); sns.set_theme(style="white", context="talk"); ax1.bar(df_res_div.index, df_res_div['Diversidade'], color="#007032", width=0.8); ax1.set_ylabel('Diversidade (Shannon)', fontsize=14); ax1.tick_params(axis='y', direction='in', labelsize=12); ax1.set_xlabel(None); rotulos_x_formatados = [l.replace('1 ','1ª ',1).replace('2 ','2ª ',1) for l in df_res_div.index]; ax1.set_xticks(range(len(rotulos_x_formatados))); ax1.set_xticklabels(rotulos_x_formatados, rotation=90, ha='center', fontsize=12); ax1.tick_params(axis='x', direction='in', pad=5); ax2 = ax1.twinx(); ax2.plot(df_res_div.index, df_res_div['Equitabilidade'], marker='o', ls='None', color="#82C21F", markersize=8); ax2.set_ylabel('Equitabilidade (Pielou)', fontsize=14); ax2.set_ylim(0, 1.05); ax2.tick_params(axis='y', direction='in', labelsize=12); posicao_linha = len(matriz_pontos) - 0.5; ax1.axvline(x=posicao_linha, color='darkgray', ls='-', lw=1.2); ax1.grid(False); ax2.grid(False); sns.despine(ax=ax1); sns.despine(ax=ax2, left=True); patch_handle = Patch(color="#007032", label='Diversidade'); line_handle = ax2.get_lines()[0]; line_handle.set_label('Equitabilidade'); fig.legend(handles=[patch_handle, line_handle], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=12); plt.tight_layout(rect=[0, 0, 1, 1]); st.pyplot(fig)
        
        with st.expander("Análise de Similaridade (Diversidade Beta)", expanded=False):
            st.subheader("Dendrograma de Similaridade (Bray-Curtis)"); matriz_sim = df_final[df_final['tipo_amostragem'] == 'Quantitativo'].pivot_table(index='ponto_amostral', columns='especie', values='densidade', aggfunc='sum', fill_value=0); matriz_sim = matriz_sim.loc[matriz_sim.sum(axis=1) > 0]
            if not matriz_sim.empty and matriz_sim.shape[0] > 1:
                dist_bc = skbio.diversity.beta_diversity('braycurtis', matriz_sim.values, ids=matriz_sim.index); linked = linkage(dist_bc.condensed_form(), method='average'); fig, ax = plt.subplots(figsize=(10, 8)); sns.set_theme(style="white"); dendrogram(linked, orientation='left', labels=matriz_sim.index.tolist(), ax=ax, color_threshold=0, above_threshold_color='black'); ax.invert_xaxis(); 
                def format_similarity(x, pos): return f'{(1 - x) * 100:.0f}%'
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_similarity)); ax.set_xlabel('Similaridade', fontsize=14); ax.tick_params(axis='x', labelsize=12); sns.despine(left=True); plt.tight_layout(); st.pyplot(fig)
            else: st.warning("Não há dados quantitativos suficientes para a análise de similaridade.")

        with st.expander("Curva de Suficiência Amostral", expanded=False):
            st.subheader("Curva do Coletor com Estimador Jackknife 1"); df_suficiencia = df_final.copy(); df_suficiencia['presenca'] = 1; matriz_suf = df_suficiencia.pivot_table(index=['campanha', 'ponto_amostral'], columns='especie', values='presenca', aggfunc='max', fill_value=0); n_samples = matriz_suf.shape[0]
            if n_samples > 1:
                n_randomizations = 50; sobs_curves = np.zeros((n_randomizations, n_samples)); sest_curves = np.zeros((n_randomizations, n_samples))
                for rand in range(n_randomizations):
                    shuffled_indices = np.random.permutation(n_samples); shuffled_matrix = matriz_suf.iloc[shuffled_indices]
                    for i in range(1, n_samples + 1):
                        subset = shuffled_matrix.iloc[:i, :]; sobs_curves[rand, i-1] = (subset.sum(axis=0) > 0).sum(); sest_curves[rand, i-1] = jackknife_1_estimator(subset)
                mean_sobs = sobs_curves.mean(axis=0); mean_sest = sest_curves.mean(axis=0); std_sest_original = sest_curves.std(axis=0); std_sest_ordenado = np.sort(std_sest_original); fig, ax = plt.subplots(figsize=(12, 8)); sns.set_theme(style="white"); x_axis = np.arange(1, n_samples + 1); cor_rarefacao = "#007032"; cor_estimada = "#82C21F"
                ax.plot(x_axis, mean_sobs, color=cor_rarefacao, linewidth=2.5, label='Riqueza Observada'); ax.plot(x_axis, mean_sest, color=cor_estimada, linewidth=2.5, label='Riqueza Estimada (Jackknife 1)', marker='o', markersize=4); ax.fill_between(x_axis, mean_sest - std_sest_ordenado, mean_sest + std_sest_ordenado, color=cor_estimada, alpha=0.2)
                valor_final_obs = mean_sobs[-1]; valor_final_est = mean_sest[-1]; ax.text(n_samples + 0.2, valor_final_obs, f'{valor_final_obs:.0f}', color=cor_rarefacao, fontsize=9, va='center', weight='bold'); ax.text(n_samples + 0.2, valor_final_est, f'{valor_final_est:.1f}', color=cor_estimada, fontsize=9, va='center', weight='bold'); ax.set_xlim(right=n_samples * 1.05); ax.set_xlabel('Número de Unidades Amostrais', fontsize=14); ax.set_ylabel('Riqueza', fontsize=14); ax.tick_params(axis='both', which='major', labelsize=12); ax.tick_params(axis='y', direction='out', left=True); sns.despine()
                if ax.get_legend() is not None: ax.get_legend().remove()
                handles, labels = ax.get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=12); plt.tight_layout(rect=[0, 0.05, 1, 1]); st.pyplot(fig)
            else: st.warning("Não há amostras suficientes para a curva de suficiência.")
                
        st.success("Relatório gerado com sucesso!")
        st.balloons()
