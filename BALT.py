import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import uuid

import distributions
from utils import *
from visualizations import *

import cmdstanpy
from cmdstanpy import CmdStanModel

from __init__ import __version__

image_ufpe = Image.open('./src/logo.png')
image_ceerma = Image.open('./src/favicon.png')

st.set_page_config(page_title="BALT",
                   page_icon=image_ceerma, layout="wide",
                   initial_sidebar_state="expanded")

version_info = f"Version {__version__}"
st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-80px">{version_info}</div>
        """,
        unsafe_allow_html=True,
)

st.sidebar.image(image_ufpe)

st.sidebar.title("🔄 BALT")

st.sidebar.caption("Bayesian Updating for Accelerated Life Testing")

with st.sidebar:

    text =\
    """
    This app is an easy-to-use interface built in Streamlit for Bayesian Updating for Accelerated Life Testing data using
    <a href="https://mc-stan.org/cmdstanpy/" target="_blank">cmdstanpy</a> Python library.
    """
    align = 'justify'

    st.markdown(
        f'<div style="text-align: {align};">{text}</div>',
        unsafe_allow_html=True
    )


st.title('Bayesian Updating for Accelerated Life Testing (BALT)')

st.write("""
    In this app, you can provide your data to update
    your knowledge about the probability distribution
    using Bayesian Updating, combining prior information
    with new evidence.
""")

with st.expander('Short Guide'):
    st.write("""
    In simple terms, Bayesian Updating is a technique applied in the analysis
    of accelerated life data that combines prior information, derived from
    probability distributions based on historical data or expert knowledge,
    with new evidence. This process results in a posterior distribution that
    can be used to quantify reliability, time to failure, or hazard rate.

    The mathematical expression for the posterior distribution for a vector of
    parameters of interest 𝜃 is given by:

    """)

    st.latex(r"""
                \pi_1(\theta \mid \text{Data}) = \frac{ \ell(\text{Data} \mid \theta)
                \cdot \pi_0(\theta) }{ \int \ell(\text{Data} \mid \theta) \cdot
                \pi_0(\theta) \, d\theta }
                """)

    st.latex(r"""
    \begin{array}{ll}
    \pi_1(\theta \mid \text{Data}): \text{Degree of belief on } \theta \text{, after observing the data;} \\
    \pi_0(\theta): \text{Degree of belief on } \theta \text{, before observing the data;} \\
    \ell(\text{Data} \mid \theta): \text{Expresses the likelihood
                of a particular value of } \theta \text{ producing the observed data;} \\
    \int \ell(\text{Data} \mid \theta) \cdot \pi_0(\theta) \, d\theta: \text{Normalizing factor.} \\
    \end{array}
    """)

    st.write("""
    In the case of accelerated life testes, the likelihood function is obtained by the
    product of the probability distribution that describes the life of the item under teste.
    For complete failure and right censored data, the likelihood is written as:
    """)

    st.latex(r"""
    l = \prod_{i=1}^{N_c} \left[ f(t_i; \boldsymbol{\theta}_M) \right]^{n_i} \cdot
        \prod_{j=1}^{N_r} \left[ 1 - F(t_j; \boldsymbol{\theta}_M) \right]^{n_j}
    """)

    st.latex(r"""
    \begin{array}{ll}
    l: & \text{Likelihood function combining observed and censored data;} \\
    f(t_i; \boldsymbol{\theta}_M): & \text{Probability density function (PDF) evaluated at failure time } t_i; \\
    F(t_j; \boldsymbol{\theta}_M): & \text{Cumulative distribution function (CDF) evaluated at censoring time } t_j; \\
    \boldsymbol{\theta}_M: & \text{Vector of model parameters;} \\
    N_c: & \text{Number of observed failures;} \\
    N_r: & \text{Number of right-censored observations;} \\
    n_i: & \text{Number of failures in the } i^\text{th} \text{ time-to-failure data point;} \\
    n_j: & \text{Number of right censored in the } j^\text{th} \text{ censored data point}. \\
    \end{array}
    """)

# Carrega a lista de distrbuições e modelos
distr = distributions.distributions
models = distributions.alt_models
# Define o modelo selecionado
model = st.selectbox(f'Select the appropriate life distribution',
                        list(models),
                        on_change=reset_run,)
# Descrição do modelo selecionado
with st.expander('Model Description'):
    models[model]['description']()
# Descrição do formato dos dados
with st.expander('Data Format'):
    models[model]['data_format']()
# Carrega os dados de entrada
header = st.checkbox("Does your data contain header?",
                     value=True,
                     on_change=reset_run,)
head = 0 if header else None

col_1, col_2 = st.columns(2)
uploaded_file = col_1.file_uploader("Upload a XLSX file",
                                    type="xlsx",
                                    accept_multiple_files=False,
                                    label_visibility="collapsed",
                                    key='table_1',
                                    on_change=reset_run,)

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=head)
    col_2.dataframe(df, width='stretch')
    # Processa o excel para extrair os dados do modelo
    data = models[model]['data'](df)

# Define a priori de cada parâmetro do modelo
selected_priors = {}
for i in range(models[model]['n_params']):
    cols = st.columns([3,1,1,1])
    parameter_name = models[model]['params'][i]

    distribution_name = cols[0].selectbox(f'Select prior distribution \
                                            for {parameter_name}:',
                                            list(distr),
                                            key=f"dist_{i}",
                                            on_change=reset_run,)

    var = []
    variables = distr[distribution_name]['variables']
    for j, variable in enumerate(variables):
        item = variables[variable]
        to_append = cols[j+1].number_input(variable,
                                            value=item[0],
                                            min_value=item[1],
                                            step=0.01,
                                            key=f'var_{j}_{i}',
                                            on_change=reset_run,)
        var.append(to_append)
    selected_priors[models[model]['params_model'][parameter_name]] = {
        'distribution': distr[distribution_name]['distribution'],
        'values': var
    }

# Configurações do MCMC
with st.expander('MCMC setup'):
    cols = st.columns([1,1])

    def number_or_none(col, label, min_value, default, step=1, is_float=False, format_str='%d'):

        use_value = col.checkbox(f"{label}", value=False, key=label+"_check")
        if use_value:
            if is_float:
                return col.number_input(f"{label}",
                                        min_value=min_value,
                                        value=default,
                                        step=step,
                                        format=format_str,
                                        key=label+"_value",
                                        on_change=reset_run,)
            else:
                return col.number_input(f"{label}",
                                        min_value=min_value,
                                        value=default,
                                        step=step,
                                        key=label+"_value",
                                        on_change=reset_run,)
        else:
            return None

    chains = number_or_none(cols[0], 'Number of chains', 1, 1)
    parallel_chains = number_or_none(cols[0], 'Number of processes to run in parallel', 0, 0)
    threads_per_chain = number_or_none(cols[0], 'Number of threads per chain', 0, 0)
    seed = number_or_none(cols[0], 'Random seed', 0, 0)

    iter_warmup = number_or_none(cols[1], 'Warmup iterations', 0, 0)
    iter_sampling = number_or_none(cols[1], 'Sampling iterations', 0, 0)
    max_treedepth = number_or_none(cols[1], 'Max tree depth', 0, 0)
    step_size = number_or_none(cols[1], 'Step size', 0.0, 0.0,
                               is_float=True, step=0.01, format_str='%0.5f')

    check_inits= st.checkbox("Do you want to specify how the sampler initializes parameters values?", value=False)
    inits = {}
    if check_inits:
        for i in range(models[model]['n_params']):
            cols = st.columns([3,1,1,1])
            parameter_name = models[model]['params'][i]

            parameter_value = cols[0].number_input(f'{parameter_name}',
                                                   value=0.0,
                                                   format='%0.5f')

            inits[f'{parameter_name}'] = parameter_value
    else:
        inits = None

# Define a saída de interesse a ser gerada
col_1, col_2 = st.columns(2)
generate_values = col_1.radio(
    'Choose the values to be estimated:',
    ('Single', 'Multiples'),
    key='metric_radio_2')

# Configurações para rodar o MCMC
if "run_MCMC" not in st.session_state:
    st.session_state.run_MCMC = False

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Run"):
    st.session_state.run_MCMC = True
    st.session_state.results = None

# Valores de entrada para gerar as saídas de interesse
input_values = values_to_evaluate(generate_values,
                                  col_2,
                                  models[model]['input_gq'])
# Condição para rodar o MCMC
if st.session_state.run_MCMC and st.session_state.results is None:
    if uploaded_file == None:
        st.error('No file uploaded to the app'); st.stop()

    else:
        if not input_values:
            st.error('No values to evaluate were provided'); st.stop()

        for key, values in input_values.items():
            if values is None or len(values) == 0:
                st.error(f"No values provided for '{key}'."); st.stop()

            if np.all(np.array(values) == 0):
                st.error(f"Value for '{key}' is invalid."); st.stop()

    run_msg = st.empty()
    run_msg.info("Running MCMC ...")

    # Adiciona os valores para a entrada do modelo stan
    for i, (key, values) in enumerate(input_values.items()):
        if i ==0:
            nt = len(values)
            data['nt'] = nt
        data[key] = values

    cmdstanpy.install_cmdstan(compiler=True)

    # Carrega o modelo stan
    stan_template = models[model]['stan_template']
    # Substitui a priori dos parâmetros
    priors_code = generate_priors_code(selected_priors)
    # Define o código final do modelo stan
    stan_code_final = stan_template.replace("{priors}", priors_code)
    # Salva o modelo
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    model_name = f"model_{uuid.uuid4().hex}"
    stan_file = os.path.join(tmp_dir, f"{model_name}.stan")
    with open(stan_file, "w") as f:
        f.write(stan_code_final)

    try:
        # Carrega o modelo stan
        model_stan = CmdStanModel(stan_file=stan_file)
        # Executa o modelo stan
        fit = model_stan.sample(data=data,
                                chains=chains,
                                parallel_chains=parallel_chains,
                                threads_per_chain=threads_per_chain,
                                seed=seed,
                                iter_warmup=iter_warmup,
                                iter_sampling=iter_sampling,
                                max_treedepth=max_treedepth,
                                step_size=step_size,
                                inits=inits,
                                show_progress=True)
        # Remove a msg
        run_msg.empty()
        # Salva os resultados na sessão
        st.session_state.results = {
            "df_summary": fit.summary(),
            "df_draws": fit.draws_pd(),
            "fit_diagnose": fit.diagnose(),
            "input_values": input_values,
            "nt": nt,
        }
    except Exception as e:
        # Remove a msg
        run_msg.empty()
        st.error(f"An error occurred while running the MCMC: {e}")
        st.session_state.run_MCMC = False
        st.session_state.results = None
        delete_model(tmp_dir, model_name)
    finally:
        # Remove a msg
        run_msg.empty()
        # Apaga os arquivos do modelo stan
        delete_model(tmp_dir, model_name)

# Exibe os resultados
if st.session_state.results is not None:

    df_summary = st.session_state.results["df_summary"]
    df_draws = st.session_state.results["df_draws"]
    fit_diagnose = st.session_state.results["fit_diagnose"]
    input_values = st.session_state.results["input_values"]
    nt = st.session_state.results["nt"]

    st.subheader("Parameter summary")
    st.divider()

    for param_label, param_name in models[model]['params_model'].items():
        st.write(f"**Results for parameter {param_label}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Mean", f"{df_summary.loc[param_name, 'Mean']:.4f}")
        col2.metric("StdDev", f"{df_summary.loc[param_name, 'StdDev']:.4f}")
        col3.metric("P5", f"{df_summary.loc[param_name, '5%']:.4f}")
        col4.metric("P95", f"{df_summary.loc[param_name, '95%']:.4f}")

        fig = plot_draws(df_draws, param_name, param_label)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    st.subheader("Reliability")
    st.divider()

    if generate_values == 'Single':
        for i, (key, values) in enumerate(input_values.items()):
            result_str = " | ".join(
                [f"{key} = {values[0]}" for key, values in input_values.items()]
                )
        st.write(f"**Results for: {result_str}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Mean", f"{df_summary.loc['R[1]']['Mean']:.4f}")
        col2.metric("StdDev", f"{df_summary.loc['R[1]']['StdDev']:.4f}")
        col3.metric("P5", f"{df_summary.loc['R[1]']['5%']:.4f}")
        col4.metric("P95", f"{df_summary.loc['R[1]']['95%']:.4f}")

        fig = plot_draws(df_draws, 'R[1]', 'Reliability')
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    if generate_values == 'Multiples':
        st.write(f"**Results of reliability following the order of the input values provided**")

        df_show = df_summary.loc['R[1]':f'R[{nt}]'][['Mean', 'StdDev', '5%', '95%']]

        for i, (key, values) in enumerate(input_values.items()):
            df_show.insert(loc=i, column=key, value=values)

        st.write(df_show)
        st.plotly_chart(plot_reliability(df_show), use_container_width=True)
        st.divider()

    st.subheader("MCMC diagnostics statistics")
    st.write("""
             MCMC diagnostics are tools that can be used to check whether the quality
             of a sample generated with an MCMC algorithm is sufficient to provide an accurate
             approximation of the target distribution.
             """)
    st.divider()

    st.subheader("1. Effective Sample Size (ESS)")
    with st.expander('ESS'):
        st.write("""
                 Is the effective sample size large enough to get a stable estimate of uncertainty?

                 Another technical difficulty posed by MCMC methods is that the draws will typically
                 be autocorrelated (or anticorrelated) within a chain. This increases (or reduces) the
                 uncertainty of the estimation of posterior quantities of interest.

                 The **ESS** is the (estimated) number of independent observations that our sample is
                 equivalent to.

                 * We want this number to be larger rather than smaller;
                 * The greater the correlation between observations, the smaller the ESS will be.
        """)
    df_ess = df_summary[['ESS_bulk', 'ESS_tail']]
    df_ess = df_ess.loc[list(models[model]['params_model'].values()) + [f'R[{i}]' for i in range(1, nt + 1)]]
    st.dataframe(df_ess, width='stretch')

    st.subheader("2. R-hat")
    with st.expander('R-hat'):
        st.write("""
                 Another way to measure the MCMC convergece is trougth the potential
                 reduction factor or **R-hat**.

                 The statistic **R-hat** measures the ratio of the average variance of
                 draws within each chain to the variance of the pooled draws across chains;
                 if all chains are at equilibrium, these will be the same and **R-hat** will
                 be one.

        """)

        st.latex(r"""
        \hat{R} = \sqrt{\frac{\widehat{\mathrm{var}}^+(\theta|y)}{W}}
                 """)

        st.latex(r"""
        \widehat{\mathrm{var}}^+(\theta|y) = \frac{N-1}{N} W + \frac{1}{N} B
                 """)

        st.latex(r"""
        \begin{array}{ll}
        \text{N}: \text{is the number of draws per chain;} \\
        \text{B} and \text{W}: \text{are between-andwithin-chain variances;} \\
        \widehat{\mathrm{var}}^+(\theta|y): \text{marginal posterior variance of the estimand.} \\
        \end{array}
        """)
    df_rhat = df_summary[['R_hat']]
    df_rhat = df_rhat.loc[list(models[model]['params_model'].values()) + [f'R[{i}]' for i in range(1, nt + 1)]]
    st.dataframe(df_rhat, width='stretch')
    st.divider()

    with st.expander('Fit diagnose'):
        st.write(fit_diagnose)
    st.divider()


    # Download dos resultados
    st.download_button(
        "Download results",
        data=zip_file(df_summary, df_draws),
        file_name="results.zip",
        mime="application/zip"
    )