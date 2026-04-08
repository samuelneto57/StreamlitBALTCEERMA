# BAYESIAN UPDATING FOR ACCELERATED LIFE TESTING (BALT)

This project implements Bayesian Updating for reliability analysis using accelerated life testing models. The app is an easy-to-use interface built in Streamlit using [cmdstan.py](https://mc-stan.org/cmdstanpy/) Python library. The goal of this project is to use life testing data to update the parameters of life distribution to estimate the reliability.

## Installation and use
The code was tested in Python 3.9.23. To install the required packages simply run:

```
pip install -r requirements.txt
```

To run the App locally:

```
streamlit run BALT.py
```

## Contributing
New models can be implemented following the same structure as the models already established. Note that the distributions.py file must also be updated accordingly.
