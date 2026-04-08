import streamlit as st

from models.lognormal_power import lognormal_power_info
from models.exponential_model import exponential_info

# Descrição das distribuição e equações
normal_equations = [
    r'\mu = \text{Location parameter } (-\infty<\mu<\infty)',
    r'\sigma = \text{Scale parameter } (\sigma>0)',
    r'\text{Limits: } (-\infty<t<\infty) ',
    r'\text{PDF: } f(t) = \frac{1}{\sigma\sqrt{2\pi}} e^{\frac{1}{2}\left(\frac{t-\mu}{\sigma}\right)^2}=\frac{1}{\sigma}\phi\left[\frac{t-\mu}{\sigma}\right]',
    r'\text{Where } \phi\text{ is the standard normal PDF with }\mu=0\text{ and }\sigma=1',
    r'\text{CDF: } F(t) = \frac{1}{\sigma\sqrt{2\pi}}\int^t_{-\infty}e^{\left[-\frac{1}{2}\left(\frac{\theta-\mu}{\sigma}\right)^2\right]d\theta}=\Phi\left(\frac{t-\mu}{\sigma}\right)',
    r'\text{Where } \Phi\text{ is the standard normal CDF with }\mu=0\text{ and }\sigma=1',
    r'R(t) =  1-\Phi\left(\frac{t-\mu}{\sigma}\right)=\Phi\left(\frac{\mu-t}{\sigma}\right)',
    r'\text{HF: } h(t) = \frac{\phi\left[\frac{t-\mu}{\sigma}\right]}{\sigma\left(\Phi\left[\frac{\mu-t}{\sigma}\right]\right)}',
    r'\text{CHF: } H(t) = -ln\left[\Phi\left(\frac{\mu-t}{\sigma}\right)\right]',
]
lognormal_equations = [
    r'\alpha = \text{Scale parameter } (-\infty<\mu<\infty)',
    r'\beta = \text{Shape parameter } (\sigma>0)',
    r'\text{Limits: } (t \leq 0)',
    r'\text{PDF: } f(t) = \frac{1}{\sigma t\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{ln(t)-\mu}{\sigma}\right)^2}=\frac{1}{\sigma t}\phi\left[\frac{ln(t)-\mu}{\sigma}\right]',
    r'\text{Where } \phi \text{ is the standard normal PDF with }\mu=0\text{ and }\sigma=1 ',
    r'\text{CDF: } F(t) = \frac{1}{\sigma\sqrt{2\pi}}\int^t_{0}\frac{1}{\theta}e^{\left[-\frac{1}{2}\left(\frac{ln(\theta)-\mu}{\sigma}\right)^2\right]d\theta}=\Phi\left(\frac{ln(t)-\mu}{\sigma}\right)',
    r'\text{Where } \Phi \text{ is the standard normal CDF with }\mu=0\text{ and }\sigma=1',
    r'R(t) =  1-\Phi\left(\frac{ln(t)-\mu}{\sigma}\right)=\Phi\left(\frac{\mu-ln(t)}{\sigma}\right)',
    r'\text{HF: } h(t) = \frac{\phi\left[\frac{ln(t)-\mu}{\sigma}\right]}{\sigma\left(\Phi\left[\frac{\mu-ln(t)}{\sigma}\right]\right)}',
    r'\text{CHF: } H(t) = -ln\left[1-\Phi\left(\frac{ln(t)-\mu}{\sigma}\right)\right]',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]
gamma_equations = [
    r'\alpha = \text{Scale parameter } ( \alpha > 0)',
    r'\beta = \text{Shape parameter } ( \beta > 0)',
    r'\text{Limits: } ( t \leq 0 ) ',
    r'\text{PDF: } f(t) = \frac{t^{\beta-1}}{\Gamma(\beta)\alpha^\beta}e^{\frac{t}{\alpha}}',
    r'\text{Where } \Gamma(x)\text{ is the complete gamma function. }\Gamma(x)=\int^\infty_{0}t^{x-1}e^{-t} dt',
    r'\text{CDF: } F(t) = \frac{1}{\Gamma(\beta)}\gamma(\beta,\frac{t}{\alpha})',
    r'\text{Where } \gamma(x,y) \text{ is the lower incomplete gamma function. }\gamma(x,y)=\frac{1}{\Gamma(x)}\int^y_{0}t^{x-1}e^{-t}dt',
    r'R(t) =  \frac{\Gamma(\beta,\frac{t}{\alpha})}{\Gamma(\beta)}',
    r'\text{HF: } h(t) = \frac{t^{\beta-1}e^{-\frac{t}{\alpha}}}{\alpha^\beta\Gamma(\beta,\frac{t}{\alpha})}',
    r'\text{CHF: } H(t) = -ln\left[\frac{1}{\Gamma(\beta)}\Gamma(\beta,\frac{t}{\alpha})\right]',
    r'\text{When using a location parameter γ, } t = t_{real} - γ',
]

# Definição das distribuições e limites de cada parâmetro
uniform_info = {
    'n_params': 2,
    'variables': {'lower': [0.00, None],
                  'upper': [1.00, None]},
    'equations': [''],
    'distribution': 'uniform', 
}
normal_info = {
    'n_params': 2,
    'variables': {'μ': [0.00, None],
                  'σ': [1.00, 0.00]},
    'equations': normal_equations,
    'distribution': 'normal', 
}
lognormal_info = {
    'n_params': 2,
    'variables': {'μ': [0.00, None],
                  'σ': [1.00, 0.00]},
    'equations': lognormal_equations,
    'distribution': 'lognormal', 
}
gamma_info = {
    'n_params': 2,
    'variables': {'α': [0.00, 0.00],
                  'β': [1.00, 0.00]},
    'equations': gamma_equations,
    'distribution': 'gamma', 
}

# Dicionário de distribuições que podem ser utilizad
# como priori dos parâmetros dos modelos ALT
distributions = {
    'Uniform distribution': uniform_info, 
    'Normal distribution': normal_info,
    'Lognormal distribution': lognormal_info,
    'Gamma distribution': gamma_info,
}

# Definição dos modelos ALT
alt_models = {
    'Lognormal power': lognormal_power_info,
    'Exponential': exponential_info,
}