import os
import pandas as pd
import matplotlib.pyplot as plt
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import streamlit as st
from dotenv import load_dotenv

# Configuração inicial do Streamlit
st.title("Análise de Dados de Radares")
st.write("Sistema de análise de dados de radares (pardais) no Brasil")

load_dotenv()

# Configuração da API OpenAI
api_key = st.text_input("Digite sua chave da API OpenAI:", type="password")
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    st.success('Chave API configurada com sucesso!')

# Seleção do modelo
modelo = st.selectbox(
    "Selecione o modelo de IA:",
    [
        "o3-2025-04-16",
        "o3-mini-2025-01-31",
        "gpt-4.1-2025-04-14",
        "gpt-4o-2024-08-06",
        "o4-mini-2025-04-16",
        "gpt-4o-mini",
    ],
)
if modelo:
    st.success(f'Modelo {modelo} selecionado com sucesso!')

# Interface de upload de arquivo
arquivo = st.file_uploader("Escolha seu arquivo de dados", type=['csv', 'xlsx', 'xls', 'parquet'])

# Variáveis globais para o DataFrame e Agente
df = None
agente_analista = None

if arquivo is not None:
    # Obtém a extensão do arquivo
    extensao = arquivo.name.split('.')[-1].lower()
    
    try:
        # Lê o arquivo baseado na extensão
        if extensao in ['xlsx', 'xls']:
            df = pd.read_excel(arquivo)
        elif extensao == 'csv':
            df = pd.read_csv(arquivo)
        elif extensao == 'parquet':
            df = pd.read_parquet(arquivo)
        
        st.success(f'Arquivo carregado com sucesso! Formato: {extensao}')
        
        # Criar o agente apenas quando tivermos dados e API key
        if api_key and df is not None:
            # Criar o agente com base na seleção do usuário
            llm = ChatOpenAI(
                temperature=0,
                model=modelo
            )

            
            agente_analista = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                handle_parsing_errors=True,  # Passa as configurações do agente
                prefix="""Você é um analista de dados que fala português.
                Ao analisar os dados:
                1. SEMPRE responda em português de forma clara e detalhada
                2. Se a resposta incluir números ou estatísticas:
                   - Formate os números adequadamente (use vírgula para decimais)
                   - Explique o que cada valor significa
                   - Dê contexto aos resultados
                3. Se possível, inclua insights adicionais relevantes
                4. Para médias e estatísticas, explique cada coluna
                5. Quando relevante, sugira visualizações apropriadas"""
            )
            
    except Exception as e:
        st.error(f'Erro ao carregar o arquivo: {str(e)}')
else:
    st.warning('Por favor, faça o upload de um arquivo para começar a análise. Nenhum arquivo carregado ainda.')

def descritiva():
    if df is None:
        return
        
    st.write("Primeiras linhas do conjunto de dados:")
    st.write(df.head())

    st.subheader("Estatísticas Descritivas")
    st.write(df.describe())

    # Mostrar visualizações básicas
    st.subheader("Visualizações dos Dados")
    
    # Gráfico de distribuição para cada coluna numérica
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[col], edgecolor='black')
        ax.set_title(f'Distribuição de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Contagem')
        st.pyplot(fig)
        plt.close()
        



def create_plot(plot_type, x_col, y_col=None):
    """Cria diferentes tipos de gráficos baseado nos parâmetros"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'bar':
        if y_col:
            df.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax)
        else:
            df[x_col].value_counts().plot(kind='bar', ax=ax)
    elif plot_type == 'scatter':
        ax.scatter(df[x_col], df[y_col])
    elif plot_type == 'box':
        df.boxplot(column=y_col, by=x_col, ax=ax)
    elif plot_type == 'hist':
        ax.hist(df[x_col], bins=30, edgecolor='black')
        ax.set_title(f'Distribuição de {x_col}')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def consulta():
    if df is None:
        return
    if agente_analista is None:
        st.warning('Por favor, configure sua chave da API OpenAI primeiro.')
        return
        
    # Área de perguntas
    st.subheader("Faça uma pergunta sobre os dados")
    prompt = st.text_input("Digite sua pergunta:")
    
    if st.button('Analisar Pergunta'):
        if prompt:
            try:
                result = agente_analista.run(prompt)
                st.write(result)
                
                # Adicionar visualizações relevantes baseadas na pergunta
                if "média" in prompt.lower() or "distribuição" in prompt.lower():
                    st.subheader("Visualizações Relacionadas")
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        fig = create_plot('hist', col)
                        st.pyplot(fig)
                        plt.close()
                        
            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
        else:
            st.warning("Por favor, digite uma pergunta.")


# Sidebar para opções de visualização
if df is not None:  # Só mostra a sidebar se tiver dados carregados
    st.sidebar.title("Opções de Visualização")
    if st.sidebar.checkbox("Personalizar Gráficos"):
        plot_type = st.sidebar.selectbox(
            "Tipo de Gráfico",
            ['barra', 'dispersão', 'caixa', 'histograma']
        )
        x_col = st.sidebar.selectbox("Coluna X", df.columns)
        if plot_type != 'histograma':
            y_col = st.sidebar.selectbox("Coluna Y", df.columns)
        else:
            y_col = None
        
        if st.sidebar.button("Gerar Gráfico"):
            # Mapear nomes em português para tipos de gráfico em inglês
            plot_type_map = {
                'barra': 'bar',
                'dispersão': 'scatter',
                'caixa': 'box',
                'histograma': 'hist'
            }
            fig = create_plot(plot_type_map[plot_type], x_col, y_col)
            st.pyplot(fig)
            plt.close()

# Botões principais
if st.button('Mostrar Dados'):
    descritiva()

# Área de consulta sempre visível
consulta()

