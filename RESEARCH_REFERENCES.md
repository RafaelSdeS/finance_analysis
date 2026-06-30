# Research References: RL-Based Portfolio Allocation

This document lists academic papers, technical reports, and resources relevant to learning-based portfolio allocation strategies using reinforcement learning.

---

## Foundational Portfolio Theory

### Modern Portfolio Theory & Markowitz Optimization
- **Markowitz, H. (1952).** "Portfolio Selection." *Journal of Finance*, 7(1), 77–91.
  - Seminal work on risk-return tradeoff and diversification.
  - Baseline for all portfolio optimization.

- **Sharpe, W. F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." *Journal of Finance*, 19(3), 425–442.
  - CAPM; foundational for understanding systematic vs idiosyncratic risk.

- **Merton, R. C. (1969).** "Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case." *Review of Economics and Statistics*, 51(3), 247–257.
  - Dynamic portfolio theory; optimal control framework (precursor to RL).

### Factor Models & Multi-Factor Investing
- **Fama, E. F., & French, K. R. (1993).** "Common risk factors in the returns on stocks and bonds." *Journal of Financial Economics*, 33(1), 3–56.
  - 3-factor model (market, size, value); fundamental for feature engineering.

- **Carhart, M. M. (1997).** "On Persistence in Mutual Fund Performance." *Journal of Finance*, 52(1), 57–82.
  - 4-factor model; adds momentum factor.

---

## Reinforcement Learning Fundamentals

### Deep Q-Networks (DQN)
- **Mnih, V., et al. (2013).** "Playing Atari with Deep Reinforcement Learning." *arXiv:1312.5602*.
  - DQN algorithm; breakthrough in deep RL using experience replay.

- **Mnih, V., et al. (2015).** "Human-level control through deep reinforcement learning." *Nature*, 529(7587), 529–533.
  - Improved DQN with double Q-learning and dueling architectures.

### Policy Gradient Methods (PPO, A3C, TRPO)
- **Schulman, G., et al. (2015).** "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICML*, 2015.
  - GAE; efficient advantage estimation for policy gradients.

- **Schulman, G., et al. (2017).** "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
  - PPO; stable, practical algorithm widely used in finance applications.

- **Mnih, A., & Gregor, K. (2014).** "Neural Variational Inference and Learning." *ICML*, 2014.
  - Variational methods; alternative to policy gradients.

### Actor-Critic Methods
- **Konda, V. R., & Tsitsiklis, J. N. (2000).** "Actor-Critic Algorithms." *SIAM Journal on Control and Optimization*, 42(4), 1143–1166.
  - Foundational actor-critic theory; combines policy gradient + value function.

- **Fujimoto, S., et al. (2018).** "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*, 2018.
  - TD3 (Twin Delayed DDPG); practical improvements for continuous control.

---

## RL for Portfolio Optimization & Trading

### Seminal Works in RL-Based Trading
- **Moody, J., & Saffell, M. (2001).** "Learning to Trade via Direct Reinforcement." *IEEE Transactions on Neural Networks*, 12(4), 875–888.
  - Early application of RL to trading; reward function design is critical.

- **Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006).** "Reinforcement Learning for Optimized Trade Execution." *ICML*, 2006.
  - RL for execution strategy (not just allocation); addresses market impact.

### Modern Deep RL for Portfolio Management
- **Jiang, Z., Xu, D., & Liang, J. (2017).** "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem." *arXiv:1706.10059*.
  - One of the first end-to-end deep RL portfolio papers.
  - Uses DQN with discrete action space (select top assets).

- **Li, Y., et al. (2018).** "A Deep Reinforcement Learning Framework for High-Frequency Trading." *arXiv:1811.07522*.
  - Continuous action space; addresses position sizing + timing.

- **Théate, T., & Ernst, D. (2020).** "An Application of Deep Reinforcement Learning to Algorithmic Trading." *arXiv:2010.01893*.
  - Comprehensive framework; includes transaction costs, slippage, realistic market model.

- **Wang, Y., et al. (2019).** "A Multi-Agent Deep Reinforcement Learning Method for Impression-based Ad Recommendation." *SIGIR*, 2019.
  - Multi-asset allocation using multi-agent RL; relevant for sector rotation.

### Practical Portfolio Applications
- **Almahdi, S., & Yang, S. Y. (2017).** "An Adaptive Portfolio Trading System: A Risk-Return Portfolio Optimization using Recurrent Neural Network and Nonparametric Regression." *Computers & Operations Research*, 84, 186–203.
  - RNN for feature engineering; portfolio optimization with regime switching.

- **Chong, E., Han, C., & Park, F. C. (2017).** "Deep Learning Networks for Stock Market Analysis and Prediction: Methodology, Analysis and Results." *Journal of Computer Science and Technology*, 32(4), 835–853.
  - Deep learning features for trading; comparison with traditional ML.

---

## Risk Management & Constraints

### Volatility Control & Risk-Adjusted Returns
- **Clarke, R., De Silva, H., & Thorley, S. (2016).** "Fundamentals of Efficient Factor Investing." *Financial Analysts Journal*, 72(6), 1–20.
  - Risk parity; volatility targeting strategies.

- **Arnott, R. D., Beck, S. L., Kalesnik, V., & West, J. (2016).** "How Can 'Smart Beta' Go Horribly Wrong?" *Research Affiliates Publications*.
  - Practical pitfalls in factor-based investing; data mining bias, crowding.

### Drawdown Control
- **Grossman, S. J., & Zhou, Z. (1993).** "Optimal Investment Strategies for Controlling Drawdowns." *Mathematical Finance*, 3(3), 241–276.
  - Theory of drawdown-constrained investing; relevant to reward function design.

- **Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005).** "Portfolio Optimization with Drawdown Constraints." *Bernoulli*, 11(3), 541–564.
  - CVaR-based approach; practical constraint for risk management.

### Transaction Costs & Market Microstructure
- **Almgren, R., & Chriss, N. (2001).** "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5–39.
  - Execution cost model; linear + quadratic (market impact) terms.

- **Huberman, G., & Stanzl, W. (2005).** "Optimal Liquidity Trading." *Review of Finance*, 9(2), 165–200.
  - Inventory models; realistic trading costs.

---

## Market Regimes & Macro Features

### Interest Rates & Regime Switching
- **Hamilton, J. D. (1989).** "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 357–384.
  - Hidden Markov Models (HMM) for regime detection.
  - Applicable to SELIC/interest rate environments.

- **Guidolin, M., & Timmermann, A. (2007).** "Asset Allocation under Multivariate Regime Switching." *Journal of Economic Dynamics and Control*, 31(11), 3503–3544.
  - Regime-aware portfolio optimization; adapts to market states.

### Inflation & Real Returns
- **Brière, M., Signori, O., & Urevig, A. (2012).** "Inflation Hedging Portfolios in Different Economic Regimes." *Journal of Real Estate Finance and Economics*, 45(1), 127–152.
  - Inflation adjustment strategies; relevant to IPCA adjustments.

### Sector & Industry Factors
- **Fama, E. F., & French, K. R. (2012).** "Size, value, and momentum in international stock returns." *Journal of Financial Economics*, 105(3), 457–472.
  - Industry/sector factor model; applicable to Brazilian equities.

---

## Deep Learning for Feature Engineering

### Representation Learning
- **Bengio, Y., Courville, A., & Vincent, P. (2013).** "Representation Learning: A Review and New Perspectives." *PAMI*, 35(8), 1798–1828.
  - Foundational; autoencoders, RBMs for feature extraction.

### Attention & Transformers
- **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*, 2017.
  - Transformer architecture; applicable to time-series financial data.

- **Lin, K., Zhao, Z., Liu, Z., & Zhou, X. (2020).** "A2C: An Attention-based Cryptocurrency Trading Agent." *ACM SIGMOD Workshop DMDB*, 2020.
  - Attention for feature weighting in trading; sector/asset selection.

### LSTMs & RNNs for Time Series
- **Hochreiter, S., & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780.
  - LSTM; essential for sequential financial data.

- **Cho, K., et al. (2014).** "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP*, 2014.
  - GRU; alternative to LSTM; faster training.

---

## Backtesting & Performance Evaluation

### Walk-Forward & Anchored Analysis
- **Pring, M. J. (2002).** "Technical Analysis Explained: The Successful Investor's Guide to Understanding Bottom-Up and Top-Down Investing Techniques." 4th ed.
  - Classic reference on avoiding data mining bias in backtests.

- **Arnott, R. D., Beck, S. L., Kalesnik, V., & West, J. (2016).** "How Can 'Smart Beta' Go Horribly Wrong?" *Research Affiliates Publications*.
  - Practical backtest pitfalls; importance of walk-forward validation.

### Performance Metrics
- **Sharpe, W. F. (1994).** "The Sharpe Ratio." *Journal of Portfolio Management*, 21(1), 49–58.
  - Sharpe ratio; risk-adjusted return metric (return / volatility).

- **Calmar, T. R. (1991).** "Drawdown: A Rational Metric for Evaluating Investment Strategy." *Futures*, 20(5), 60–61.
  - Calmar ratio; return / max_drawdown (addresses tail risk).

- **Dowd, K. (1999).** "Measuring Market Risk." 2nd ed., Wiley.
  - VaR, CVaR; tail risk measurement.

---

## Applications in Brazilian Markets

### B3 / Bovespa Studies
- **Machado, M. A., et al. (2017).** "Machine Learning (ML) for the stock market: A systematic literature review." *Expert Systems with Applications*, 89, 106–119.
  - Survey of ML in stock markets; includes Brazilian context.

- **Fonseca, R. P., et al. (2019).** "Optimal Asset Allocation under Regime Switching in a Developing Market: The Case of Brazil." *Emerging Markets Review*, 39, 100598.
  - Regime switching applied to B3; SELIC as state variable.

### Dividend Strategies in Emerging Markets
- **Blakeslee, D., et al. (2016).** "Dividend Yield Strategies in Developed and Emerging Markets." *Journal of Portfolio Management*, 42(9), 29–42.
  - Dividend yield as factor; particularly relevant for Brazilian stocks (high dividend culture).

---

## Practical Implementation & Production

### MLOps for Trading Systems
- **Venzin, M., Immorlev, K., & Sliwinski, P. (2021).** "Machine Learning in Production: Challenges and Solutions." *arXiv:2104.06999*.
  - Production considerations; monitoring, retraining, drift detection.

- **Folkvord, E., et al. (2019).** "Operational Machine Learning." *O'Reilly Media*.
  - Data pipelines, feature stores, model governance for financial systems.

### Backtesting Frameworks
- **Wes McKinney & Pandas Development Team.** *Pandas: Powerful Data Structures for Data Analysis and Computing.*
  - Industry standard for data wrangling; essential for backtesting.

- **Lopez de Prado, M. (2018).** "Advances in Financial Machine Learning." *Wiley*.
  - Comprehensive guide; covers backtesting, feature engineering, risk management.

---

## Sentiment & Alternative Data

### NLP for Financial News
- **Gentzkow, M., Shapiro, J. M., & Taddy, M. (2019).** "Measuring Polarization in High-Dimensional Data." *NBER Working Paper w26504*.
  - Sentiment extraction; applicable to earnings calls, news.

- **Renault, T. (2017).** "Intraday Online Investor Sentiment and Return Predictability." *Economics Letters*, 151, 8–11.
  - Retail investor sentiment as predictive feature.

### Web Search & Attention Data
- **Da, Z., Engelberg, J., & Gao, P. (2015).** "The Sum of Small Things: A Theory of Marginal Measures." *Journal of Finance*, 70(2), 339–380.
  - Google Trends for investor attention; weak but measurable signal.

---

## Tutorials & Open-Source Resources

### Reinforcement Learning
- **Sutton, R. A., & Barto, A. G. (2018).** "Reinforcement Learning: An Introduction." 2nd ed., MIT Press.
  - Gold standard textbook; comprehensive RL theory.

- **Lillicrap, T. P., et al. (2015).** "Continuous Control with Deep Reinforcement Learning." *ICLR*, 2016.
  - DDPG algorithm; practical for continuous action spaces (portfolio weights).

### Python Libraries & Tools
- **OpenAI Gym:** https://gym.openai.com/
  - Standard RL environment interface; useful for portfolio simulation.

- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
  - Production-grade RL implementations (DQN, PPO, TD3, SAC).

- **Ray Tune:** https://docs.ray.io/en/latest/tune/
  - Hyperparameter tuning for RL agents; distributed training.

- **Backtrader:** https://www.backtrader.com/
  - Backtesting framework; simpler than Zipline for portfolio testing.

- **Zipline-Reloaded:** https://zipline.ml4trading.io/
  - Event-driven backtester; production-grade (used by Quantopian).

---

## Critical Papers on Pitfalls & Overfitting

### Data Mining Bias & P-Hacking
- **Arnott, R. D., Beck, S. L., Kalesnik, V., & West, J. (2016).** "How Can 'Smart Beta' Go Horribly Wrong?" *Research Affiliates Publications*.
  - Essential reading; shows how overfitting ruins real-world strategies.

- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** "… and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5–68.
  - Meta-analysis of factor discovery; many factors don't replicate out-of-sample.

- **Farmer, R. E., et al. (2012).** "The Econometrics of Rational Expectations." *Econometric Reviews*, 31(1), 1–24.
  - Importance of parameter stability; why many trading strategies fail in production.

### Out-of-Sample Testing
- **López de Prado, M. (2015).** "Building Diversified Portfolios that Outperform." *SSRN 2513547*.
  - Walk-forward optimization; critical for avoiding lookahead bias.

- **Prado, M. L. D. (2018).** "Advances in Financial Machine Learning." *Wiley*, Chapter 2.
  - "Backtesting under Realistic Conditions"; must-read for practitioners.

---

## Recent Trends (2020–2024)

### Transformers & Attention for Finance
- **Tsai, Y. S., et al. (2022).** "Transformer-based Deep Learning for Financial Time Series Prediction." *arXiv:2204.01623*.
  - Application of attention mechanisms to market prediction.

### Explainability & Interpretability
- **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why Should I Trust You? Explaining the Predictions of Any Classifier." *KDD*, 2016.
  - LIME; local interpretable model-agnostic explanations.
  - Important for understanding RL allocation decisions.

### Multi-Task & Meta-Learning
- **Finn, C., Abbeel, P., & Levine, S. (2019).** "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML*, 2019.
  - Meta-learning; adapting to new market regimes quickly.

### Causal Inference
- **Pearl, J. (2009).** "Causality: Models, Reasoning, and Inference." 2nd ed., Cambridge University Press.
  - Causal frameworks; avoiding spurious correlations in feature selection.

---

## Key Takeaways for This Project

1. **Reward Function Design:** Most critical. Moody & Saffell (2001) + Théate & Ernst (2020) show that naive reward functions fail. Your reward = return − λ₁·volatility − λ₂·drawdown − λ₃·turnover is well-grounded.

2. **Regime Awareness:** Guidolin & Timmermann (2007) + Fonseca et al. (2019) emphasize that market regimes matter. SELIC as a state variable is smart; incorporate Hamilton's HMM if possible.

3. **Lookahead Bias:** López de Prado (2018) is mandatory reading. Walk-forward testing (train 2005–2020, validate 2020–2023, test 2024–2026) is essential.

4. **Transaction Costs:** Almgren & Chriss (2001) + Huberman & Stanzl (2005) show realistic costs matter. 0.1% per trade is reasonable for Brazil; test sensitivity.

5. **Dividend Reinvestment:** Brazilian equities pay high dividends. Blakeslee et al. (2016) + your dividend features are well-motivated.

6. **Benchmark:** IBOV outperformance is achievable but requires discipline. Arnott et al. (2016) on smart beta pitfalls is required reading before launch.

---

## Recommended Reading Order (for Getting Started)

**Week 1: Foundations**
1. Markowitz (1952) — 30 min
2. Merton (1969) — 1 hour
3. Moody & Saffell (2001) — 1.5 hours

**Week 2: Deep RL**
1. Sutton & Barto (2018), Chapters 1–3 — 3 hours
2. Schulman et al. (2017) PPO — 1 hour
3. Mnih et al. (2015) DQN — 1 hour

**Week 3: Portfolio RL**
1. Jiang et al. (2017) — 1.5 hours
2. Théate & Ernst (2020) — 2 hours
3. Guidolin & Timmermann (2007) — 1.5 hours

**Week 4: Production & Pitfalls**
1. López de Prado (2018) Chapters 1, 2, 4 — 3 hours
2. Arnott et al. (2016) — 1 hour
3. Harvey et al. (2016) — 1.5 hours

---

## Where to Find Papers

- **arXiv.org** (free preprints): https://arxiv.org/ → CS.LG, q-fin sections
- **Scholar.google.com** (aggregator): https://scholar.google.com/
- **SSRN** (finance preprints): https://www.ssrn.com/
- **ResearchGate** (author requests): https://www.researchgate.net/
- **Your university library** (institutional access): SpringerLink, JSTOR, etc.

---

Last updated: 2026-06-30
