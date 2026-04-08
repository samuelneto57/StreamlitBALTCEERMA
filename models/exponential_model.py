import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

def description():
    """
    Describes the exponential life distribution.
    """

    st.write("""
    The exponential life distribution is one of the simplest and most widely used 
    models in reliability analysis. It is frequently applied in testing to assess 
    whether a system meets specified Mean Time Between Failure (MTBF) requirements. 
    This approach relies on the assumption of a constant failure (or repair) rate over 
    time, where the rate is defined as the inverse of the MTBF.
    """)

    st.write("""
    For complete failure and right censored data, the likelihood is written as:
    """)

    st.latex(r"""
    l(\text{data} \mid \lambda) =
    \prod_{i=1}^{N_f}
    \left[
    \lambda \exp(-\lambda t_i)
    \right]
    \times
    \prod_{j=1}^{N_c}
    \left[
    \exp(-\lambda t_j)
    \right]
    """)

    st.write("""
    The expressions for mean life, reliability and hazard rate are:
    """)

    st.markdown("**Mean life function:**")
    st.latex(r"""
    \mu = \frac{1}{\lambda}
    """)

    st.markdown("**Reliability function:**")
    st.latex(r"""
    R(t) = \exp(-\lambda t)
    """)

    st.markdown("**Hazard rate function:**")
    st.latex(r"""
    \lambda(t) = \frac{f(t)}{R(t)} = \lambda
    """)

def data_format():
    """
    Describes the required format for the input data.
    """

    st.info("""
    Upload an excel file that contains the following columns:
    * Type - 'F' for failure time, 'C' for right-censored time;
    * Time - the time values at failure and/or that did not result in failure;

    """)
    df_show = {
        'Type': ['F','F','F','F','F','F','F','F','F','C','C','C'],
        'Time': [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000],   
    }
    df_show = pd.DataFrame.from_dict(df_show)
    st.write(df_show)
    # Botão para download do exemplo de dados
    buffer = BytesIO()
    df_show.to_excel(buffer, index=False)

    st.download_button(
        "Download",
        data=buffer.getvalue(),
        file_name="data.xlsx"
    )

def model_data(df):
    """
    Extracts the necessary data from the input DataFrame.
    """
    if df.shape[1] != 2:
        st.error("The uploaded file must contain at least two columns: 'Type' and 'Time'.")
        return None
    else: 
        counts = df.iloc[:, 0].value_counts()
        # Number of failure samples
        Nf = counts.get("F", 0)
        # Number of censored samples
        Nc = counts.get("C", 0)

        # Failure times
        Tf = df.loc[df.iloc[:, 0] == "F", df.columns[1]].values
        # Censored times
        Tc = df.loc[df.iloc[:, 0] == "C", df.columns[1]].values

        data = {
        "N_f": Nf,
        "N_c": Nc,
        "Tf": Tf,
        "Tc": Tc,
        }
        return data

# Modelo template para exponencial life model
stan_template = """
data {
    int<lower=0> N_f; // number of failures
    int<lower=0>  N_c; // number of censored samples
    int<lower=1> nt; // number of points for calculation
    array[N_f] real<lower=0> Tf; // failure times
    array[N_c] real<lower=0> Tc; // censored times
    array[nt] real<lower=0> Time; // time points for calculation
}
parameters {
    real<lower=0, upper=1> lambda;
}
model {
    // prior
    {priors}

    // likelihood
    for (i in 1:N_f) {
        target += exponential_lpdf(Tf[i] | lambda);
    }
    for (i in 1:N_c) {
        target += exponential_lccdf(Tc[i] | lambda);
    }
}
generated quantities {
    array[nt] real R;

    for (i in 1:nt) {
        R[i] = exp(-lambda * Time[i]);
    }
}
"""

# Dicionário com as informações do modelo exponencial
exponential_info = {
    'n_params': 1,
    'params': ['λ'],
    'description': description,
    'params_model':{'λ': 'lambda'},
    'stan_template': stan_template,
    'data_format': data_format,
    'data': model_data,
    'input_gq': ['Time'],
}