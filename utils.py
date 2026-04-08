import os
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import zipfile

def reset_run():
    """
    Reseta a variável de controle do run do modelo e os resultados
    armazenados na sessão. Deve ser chamada sempre que o usuário alterar
    algum input para garantir que o modelo seja reexecutado com os novos
    valores.
    """

    st.session_state.run_MCMC = False
    st.session_state.results = None

def generate_priors_code(selected_priors: dict):
    """
    Gera o código de definição das priors para os parâmetros do modelo
    a partir do dicionário de priors selecionados pelo usuário.

    Args:
        selected_priors (dict): Dicionário contendo as informações das priors
        selecionadas pelo usuário. A chave é o nome do parâmetro e o valor
        é outro dicionário com as chaves 'distribution' e 'values'.

    Returns:
        str: Código formatado para definição das priors no modelo Stan.
    """

    lines = []
    for param, info in selected_priors.items():
        dist = info['distribution'].lower()
        values = info['values']
        
        if dist == 'uniform':
            lines.append(f"{param} ~ uniform({values[0]}, {values[1]});")
        elif dist == 'normal':
            lines.append(f"{param} ~ normal({values[0]}, {values[1]});")
        elif dist == 'lognormal':
            lines.append(f"{param} ~ lognormal({values[0]}, {values[1]});")
        elif dist == 'gamma':
            lines.append(f"{param} ~ gamma({values[0]}, {values[1]});")
        else:
            raise ValueError(f"Distribution {dist} not implemented.")
    
    return "\n  ".join(lines)

def values_to_evaluate(generate_values: str, 
                       col, 
                       inputs_spec: list):
    """
    Função para entrada de valores para cálculo da confiabilidade.
    Permite ao usuário escolher entre gerar um único valor ou múltiplos 
    valores para cada variável de interesse, e também escolher o método 
    de entrada (manual ou upload de arquivo).

    Args:
        generate_values (str): Opção escolhida pelo usuário para gerar um único 
            valor ou múltliplos valores.
        col: Coluna do Streamlit onde os inputs serão renderizados.
        inputs_spec (list): Lista de strings com os nomes das variáveis de interesse.

    Returns:
        dict: Dicionário onde as chaves são os nomes das variáveis de interesse 
            e os valores são arrays numpy com os valores inseridos.

    """

    result = {}
    # Opção para gerar um único valor
    if generate_values == 'Single':
        for var in inputs_spec:
            value = col.number_input(
                var,
                value=0.0,
                format='%0.5f',
                key=f"{var}_single",
                on_change=reset_run,
            )
            result[var] = np.array([value])

    # Opção para gerar múltiplos valores
    elif generate_values == 'Multiples':
        input_method = col.radio(
            'Choose the input method:',
            ('Manual input', 'Drop file'),
            key='metric_radio_dynamic'
        )

        if input_method == 'Manual input':

            df_values = pd.DataFrame({var: [] for var in inputs_spec})

            df_to_generate = col.data_editor(
                df_values,
                num_rows="dynamic",
                on_change=reset_run,
            )

            if not df_to_generate.empty:
                for var in inputs_spec:
                    result[var] = df_to_generate[var].dropna().values

        else:
            uploaded_file = col.file_uploader(
                "Upload a XLSX file",
                type="xlsx",
                accept_multiple_files=False,
                on_change=reset_run,
            )

            if uploaded_file is not None:
                df_uploaded = pd.read_excel(uploaded_file)

                missing = [v for v in inputs_spec if v not in df_uploaded.columns]

                if missing:
                    col.error(f"Missing columns: {missing}")
                else:
                    for var in inputs_spec:
                        result[var] = df_uploaded[var].dropna().values

    return result

def delete_model(tmp_dir: str,
                 model_name: str):
    """
    Apaga os arquivos do modelo Stan gerados durante a execução 
    para evitar acúmulo de arquivos temporários.

    Args:
        tmp_dir (str): Diretório onde os arquivos temporários do modelo são salvos.
        model_name (str): Nome do modelo para identificar os arquivos a serem apagados.
    """
    
    for file in os.listdir(tmp_dir):
        if file.startswith(model_name):
            os.remove(os.path.join(tmp_dir, file))

def zip_file(df_summary: pd.DataFrame,
             df_draws: pd.DataFrame,):
    """
    Cria um arquivo zip contendo os arquivos CSV do resumo 
    dos parâmetros e das amostras do modelo.
    
    Args:
        df_summary (pd.DataFrame): DataFrame contendo o 
            resumo dos parâmetros do modelo.
        df_draws (pd.DataFrame): DataFrame contendo as 
            amostras dos parâmetros do modelo.
    Returns:
        BytesIO: Objeto em memória contendo o arquivo zip gerado.
    """

    buffer = BytesIO()
    
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("df_summary.csv", df_summary.to_csv(index=False))
        z.writestr("df_draws.csv", df_draws.to_csv(index=False))
    
    buffer.seek(0)

    return buffer