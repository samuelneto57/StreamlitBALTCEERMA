import streamlit as st
import pandas as pd
from io import BytesIO

def description():
    """
    Describes the lognormal power life-stress model.
    """
    
    st.write("""
    In the case of power life-stress with lognormal model, the median
    of lognormal distribution is equal to the nominal value for life
    in the power life stress model as:
    """)

    st.latex(r"""
    \mu = e^{\mu_t} = L(S) = \frac{1}{a S^n}

             """)

    st.write("""
    By sustituting the log-linear form of the above equation as
    the mean of the lognormal pdf, the conditional power-lognormal
    given stress is obtained as follow:
    """)

    st.latex(r"""
    f(t, S) = \frac{1}{\sigma_t \, t \sqrt{2\pi}}
    \exp\left(
        -\frac{1}{2}
        \left(
            \frac{\ln(t) + \ln(a) + n \ln(S)}{\sigma_t}
        \right)^2
    \right)
    """)

    st.write("""
    For complete failure and right censored data, the likelihood is written as: 
    """)
    
    st.latex(r"""
    l(\text{data} \mid a, n, \sigma_t) =
    \prod_{i=1}^{N_c} 
    \left\{
        \frac{1}{\sigma_t t_i} 
        \, \phi\!\left(
            \frac{\ln(t_i) + \ln(a) + n \ln(S_i)}{\sigma_t}
        \right)
    \right\}
    \times
    \prod_{j=1}^{N_r} 
    \left\{
        1 - \Phi\!\left(
            \frac{\ln(t_j) + \ln(a) + n \ln(S_j)}{\sigma_t}
        \right)
    \right\}
    """)

    st.write("""
    The expressions for mean life, reliability and hazard rate are: 
    """)

    st.markdown("**Mean life function:**")
    st.latex(r"\mu = e^{\mu_t + \frac{1}{2}\sigma_t^2}")

    st.markdown("**Reliability function:**")
    st.latex(r"""
    R(t,S) = \int_{t}^{\infty} f(t,S) \, dt 
    = \int_{t}^{\infty} \frac{1}{\sigma_t \, t \sqrt{2\pi}} 
    \exp\left(
        -\tfrac{1}{2}
        \left(
            \frac{\ln(t) + \ln(a) + n \ln(S)}{\sigma_t}
        \right)^2
    \right) dt
    """)

    st.markdown("**Hazard rate function:**")
    st.latex(r"""
    \lambda(t,S) = \frac{f(t,S)}{R(t,S)} 
    = \frac{\frac{1}{\sigma_t \, t \sqrt{2\pi}}
    \exp\left(
        -\frac{1}{2}
        \left(
            \frac{\ln(t) + \ln(a) + n \ln(S)}{\sigma_t}
        \right)^2
    \right)
    }{\int_{t}^{\infty} f(t,S) \, dt 
    = \int_{t}^{\infty} \frac{1}{\sigma_t \, t \sqrt{2\pi}} 
    \exp\left(
        -\tfrac{1}{2}
        \left(
            \frac{\ln(t) + \ln(a) + n \ln(S)}{\sigma_t}
        \right)^2
    \right) dt
    }
    """)

def data_format():
    """
    Describes the required format for the input data.
    """
    st.info("""
    Upload an excel file that contains the following columns:
    * Type - 'F' for failure time, 'C' for right-censored time;
    * Time - the time values at failure and/or that did not result in failure;
    * Stress - stress level associated with this failure and/or that did not result in failure.

    """)
    df_show = {
        'Type': ['F','F','F','F','F','F','F','F','F','C','C','C'],
        'Time': [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000],
        'Stress':[340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210],   
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
    if df.shape[1] != 3:
        st.error("The uploaded file must contain at least three columns: 'Type', 'Time', and 'Stress'.")
        return None
    else: 
        counts = df.iloc[:, 0].value_counts()
        # Number of failure samples
        Nf = counts.get("F", 0)
        # Number of censored samples
        Nc = counts.get("C", 0)

        # Stress failure data
        Sf = df.loc[df.iloc[:, 0] == "F", df.columns[1]].values
        # Stress censored data
        Sc = df.loc[df.iloc[:, 0] == "C", df.columns[1]].values

        # Failure times
        Tf = df.loc[df.iloc[:, 0] == "F", df.columns[2]].values
        # Censored times
        Tc = df.loc[df.iloc[:, 0] == "C", df.columns[2]].values

        data = {
        "N_f": Nf,
        "N_c": Nc,
        "Sf": Sf,
        "Sc": Sc,
        "Tf": Tf,
        "Tc": Tc}

        return data

# Modelo template para lognormal power life-stress model
stan_template = """
data {
    int<lower=0> N_f; // number of failures
    int<lower=0> N_c; // number of censored samples
    int<lower=1> nt; // number of points for calculation
    array[N_f] real<lower=0> Sf; // stress levels for failure data
    array[N_c] real<lower=0> Sc; // stress levels for censored data
    array[N_f] real<lower=0> Tf; // failure times
    array[N_c] real<lower=0> Tc; // censored times
    array[nt] real<lower=0> Time; // time points for calculation
    array[nt] real Stress; // stress levels for calculation

}
parameters {
    real lna;
    real n;
    real <lower=0> sigma;
}
transformed parameters {
    array[N_f] real mu_f;
    array[N_c] real mu_c;

    for (i in 1:N_f) {
        mu_f[i] = (lna - n * log(Sf[i]));
    }
    for (i in 1:N_c) {
        mu_c[i] = (lna - n * log(Sc[i]));
    }
}
model {
    // priors
    {priors}

    // likelihood
    for (i in 1:N_f) {
        target += lognormal_lpdf(Tf[i] | mu_f[i], sigma);
    }
    for (i in 1:N_c) {
        target += lognormal_lccdf(Tc[i] | mu_c[i], sigma);
    }
}
generated quantities {
    array[nt] real R;
    real mu_gq;

    for (i in 1:nt) {
        mu_gq = lna - n * log(Stress[i]);
        R[i] = 1 - lognormal_cdf(Time[i] | mu_gq, sigma);
    }
}
"""

# Dicionário com as informações do modelo lognormal power
lognormal_power_info = {
    'n_params': 3,
    'params': ['ln(a)', 'n', 'σ'],
    'description': description,
    'params_model':{'ln(a)': 'lna', 'n': 'n', 'σ': 'sigma'},
    'stan_template': stan_template,
    'data_format': data_format,
    'data': model_data,
    'input_gq': ['Time', 'Stress'],
}