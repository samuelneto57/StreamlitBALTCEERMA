import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_draws(df: pd.DataFrame,
               var_name: str,
               var_label: str):
    """
    Plota a KDE e os samples de cada chain do parâmetro 
    da distribuição.

    Args:
        df (pd.DataFrame): DataFrame contendo os draws do modelo.
        var_name (str): Nome da coluna do parâmetro a ser plotado.
        var_label (str): Rótulo do parâmetro para os eixos dos gráficos.
    
    Returns:
        fig (go.Figure): Figura do Plotly contendo os gráficos de 
        KDE e trace dos samples para cada chain.
    """

    # Filtra as chains utilizadas no fit
    chains = sorted(df['chain__'].unique())
    # Define as cores
    colors = px.colors.qualitative.Alphabet
    # Subplot 
    fig = make_subplots(
        rows=1, cols=2,
    )

    for idx, i in enumerate(chains):
        # Filtra os dados da chain atual
        df_chain = df[df['chain__'] == i]
        # Define a cor para a chain atual
        color = colors[idx % len(colors)]
        # Extrai os valores do parâmetro
        param_values = df_chain[var_name].values
        # Calcula a densidade usando KDE
        kde = gaussian_kde(param_values)

        # Plota a KDE
        fig.add_trace(
            go.Scatter(x=np.linspace(param_values.min(), param_values.max(), 200),
                       y=kde(np.linspace(param_values.min(), param_values.max(), 200)),
                       mode='lines',
                       line=dict(color=color),
                       name=f'Chain {int(i)}',
                       legendgroup=f'chain_{i}',
                       showlegend=True),
            row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=df_chain['iter__'], 
                       y=df_chain[var_name], 
                       mode='lines', 
                       line=dict(color=color),
                       name=f'Chain {int(i)}',
                       legendgroup=f'chain_{i}',
                       showlegend=False,),
            row=1, col=2
            )
    fig.update_xaxes(title_text=var_label, row=1, col=1)
    fig.update_yaxes(title_text=var_label, row=1, col=2)
    return fig


def plot_reliability(df_show: pd.DataFrame):
    """
    Plota a confiabilidade média com intervalo de confiança (5% - 95%).

    Args:
        df_show (pd.DataFrame): DataFrame do summary do fit que contém as informações
        de confiabilidade média e os percentis 5% e 95%.

    Returns:
        fig (go.Figure): Figura do Plotly contendo o gráfico da confiabilidade.
    """
    x = df_show.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=df_show['95%'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='95%'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=df_show['5%'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(120,120,120,0.2)',
        line=dict(width=0),
        name='Interval of 5%-95%'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=df_show['Mean'],
        mode='lines',
        line=dict(width=2, color='black'),
        name='Mean',
    ))

    fig.update_layout(
        title='Plot of mean reliability with confidence interval (5% - 95%)',
        yaxis_title='Reliability',
        template='plotly_white'
    )
    return fig