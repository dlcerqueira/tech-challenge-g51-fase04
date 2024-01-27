import streamlit as st
import pandas as pd
import datetime
import joblib
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from utils import atualizando_dados_ipea

###### Configuração Inicial ######
dados = pd.read_csv(atualizando_dados_ipea())

def formata_numero(valor, prefixo = ''):
    if(isinstance(valor, str)):
        return datetime.datetime.strptime(valor, '%Y-%m-%d').strftime('%d/%m/%Y')
    return f'{prefixo} {valor:.2f}'

###### Modelos ######

# Previsão com Prophet
def prophet_prediction(periodo_previsao):
    # Carregando o modelo
    m = joblib.load('modelo/prophet.joblib')

    # Prevendo de acordo com o filtro
    future = m.make_future_dataframe(periods=periodo_previsao, freq="B")
    forecast = m.predict(future)
    forecast_resumo = forecast[["ds", "yhat"]].rename(columns=
                                                      {"ds": "Data", 
                                                       "yhat": "Preço - petróleo bruto - Brent (FOB)"})
    return m, forecast, forecast_resumo

# Previsão com Modelo 1
def modelo_1_prediction(periodo_previsao):
    st.text('Em construção...')


###### Gráficos e Métricas ######
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados['Data'], y=dados['Preço - petróleo bruto - Brent (FOB)'], name="Preço do Petróleo Brent"))
    fig.layout.update(title_text='Preço do Petróleo Brent (FOB)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
def prophet_plot_table(m, forecast, forecast_resumo, periodo_previsao):
    # Mostrando os últimos 5 dias de previsão e plotando o gráfico com a previsão e dados do IPEA
    st.subheader('Previsão')
    st.dataframe(forecast_resumo.tail())
        
    st.subheader(f'Gráfico de previsão em {periodo_previsao} dias')
    plot_prev_prophet = plot_plotly(m, forecast)
    st.plotly_chart(plot_prev_prophet)

###### Página dos Modelos de Previsão ######
    
#### Sidebar ####
    
st.sidebar.title('Parâmetros do Modelo')
with st.sidebar.expander('Período de Previsão', True):
    periodo = st.slider('Selecione o período de previsão:', 1, 365, 7)

with st.sidebar.expander('Modelo de Machine Learning', True):
    input_modelo = st.selectbox('Selecione o modelo que deseja utilizar:', ['Escolha um modelo','Modelo_1', 'Prophet'], 0)

rodar_modelo = st.sidebar.button(label="Rodar Modelo")

#### Página dos modelos de previsão do petróleo Brent ####
st.write('# :oil_drum: Análise de preços do Petróleo Brent')
st.header("Dados do Petróleo Brent", divider="gray")

coluna1, coluna2 = st.columns(2)
with coluna1:
    st.metric('Data da última atualização dos dados:', formata_numero(dados["Data"].max()))
with coluna2:
    st.metric('Preço da última atualização dos dados:', formata_numero(dados['Preço - petróleo bruto - Brent (FOB)'].iloc[-1], "U$"))

# Gráfico dos dados atuais  
plot_raw_data()

if(input_modelo == "Escolha um modelo"):
    st.info("Defina o modelo, os parâmetros e rode o modelo para exibição dos dados\n", icon="🚨")
if(input_modelo == "Prophet" and rodar_modelo):
    st.header("Prophet", divider="gray")
    m, forecast, forecast_resumo = prophet_prediction(periodo)

    aba1, aba2 = st.tabs(['Resultado', 'Métricas do modelo'])
    with aba1:
        prophet_plot_table(m, forecast, forecast_resumo, periodo)
    with aba2:
        st.write("Em construção")
if(input_modelo == "Modelo_1" and rodar_modelo):
    st.header("Modelo 1", divider="gray")
    modelo_1_prediction(periodo)
    
