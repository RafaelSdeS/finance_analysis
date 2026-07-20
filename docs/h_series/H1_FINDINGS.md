# H1 Findings — Sector-Neutral Alpha Screen

**Verdict: PASS**

Gate: sector-neutralized rank IC, NW-HAC |t| >= 2, BH-FDR 10%, sign-consistent in >= 60% of sub-windows, at either k in {21, 63}. Raw (non-sector-neutralized) IC is diagnostic only and never gates (a raw-only-significant characteristic is a sector-timing bet, not stock-picking alpha).

## Full screen

        characteristic  k        variant  mean_ic  se_ic   tstat  n_obs  quintile_spread_mean  sign_consistency  pvalue
        earnings_yield 21            raw   0.0195 0.0150  1.3017    180                0.0026              0.50  0.1930
        earnings_yield 21 sector_neutral   0.0169 0.0188  0.9017    180                0.0024              0.25  0.3672
        book_to_market 21            raw  -0.0190 0.0170 -1.1173    180                0.0010              0.25  0.2639
        book_to_market 21 sector_neutral  -0.0476 0.0182 -2.6139    180               -0.0120              0.75  0.0090
                    pl 21            raw   0.0237 0.0160  1.4832    180                0.0019              0.50  0.1380
                    pl 21 sector_neutral   0.0522 0.0172  3.0381    180                0.0117              0.75  0.0024
                   pvp 21            raw   0.0287 0.0173  1.6550    180                0.0025              0.50  0.0979
                   pvp 21 sector_neutral   0.0593 0.0160  3.7151    180                0.0068              1.00  0.0002
             ev_ebitda 21            raw   0.0262 0.0136  1.9245    180                0.0023              0.75  0.0543
             ev_ebitda 21 sector_neutral   0.0353 0.0176  2.0090    180                0.0058              0.75  0.0445
                   roe 21            raw   0.0314 0.0148  2.1141    180                0.0022              0.50  0.0345
                   roe 21 sector_neutral   0.0331 0.0163  2.0285    180                0.0002              0.50  0.0425
            net_margin 21            raw   0.0382 0.0154  2.4872    180                0.0053              0.75  0.0129
            net_margin 21 sector_neutral   0.0467 0.0172  2.7170    180                0.0065              0.75  0.0066
          roe_trend_4q 21            raw   0.0008 0.0148  0.0560    170               -0.0072              0.50  0.9553
          roe_trend_4q 21 sector_neutral   0.0081 0.0173  0.4679    170               -0.0089              0.50  0.6399
           debt_equity 21            raw  -0.0057 0.0105 -0.5478    180               -0.0006              0.50  0.5838
           debt_equity 21 sector_neutral  -0.0071 0.0170 -0.4213    180                0.0011              0.50  0.6736
         div_yield_12m 21            raw   0.0566 0.0145  3.9147    180                0.0179              1.00  0.0001
         div_yield_12m 21 sector_neutral   0.0519 0.0186  2.7959    180                0.0061              0.75  0.0052
momentum_vs_market_12m 21            raw   0.0655 0.0190  3.4542    180                0.0114              1.00  0.0006
momentum_vs_market_12m 21 sector_neutral   0.0876 0.0216  4.0499    180                0.0127              1.00  0.0001
momentum_vs_sector_12m 21            raw   0.0393 0.0149  2.6333    180                0.0094              0.75  0.0085
momentum_vs_sector_12m 21 sector_neutral   0.0662 0.0183  3.6085    180                0.0074              1.00  0.0003
        volatility_60d 21            raw  -0.0391 0.0196 -1.9976    180               -0.0012              0.75  0.0458
        volatility_60d 21 sector_neutral  -0.0739 0.0228 -3.2372    180               -0.0041              1.00  0.0012
        turnover_ratio 21            raw  -0.0504 0.0152 -3.3213    180               -0.0068              0.75  0.0009
        turnover_ratio 21 sector_neutral  -0.0689 0.0200 -3.4524    180               -0.0059              0.75  0.0006
    amihud_illiquidity 21            raw   0.0045 0.0110  0.4124    180                0.0017              0.75  0.6800
    amihud_illiquidity 21 sector_neutral   0.0126 0.0167  0.7519    180               -0.0002              0.50  0.4521
           pl_zhist_5y 21            raw   0.0009 0.0130  0.0671    162                0.0009              0.50  0.9465
           pl_zhist_5y 21 sector_neutral   0.0019 0.0194  0.0981    161               -0.0043              0.50  0.9219
          roe_zhist_5y 21            raw   0.0249 0.0163  1.5303    162               -0.0016              0.75  0.1259
          roe_zhist_5y 21 sector_neutral   0.0043 0.0199  0.2168    161               -0.0066              0.25  0.8283
        earnings_yield 63            raw   0.0142 0.0217  0.6542    179                0.0023              0.25  0.5130
        earnings_yield 63 sector_neutral   0.0175 0.0303  0.5778    179                0.0026              0.25  0.5634
        book_to_market 63            raw  -0.0197 0.0252 -0.7836    179                0.0055              0.50  0.4333
        book_to_market 63 sector_neutral  -0.0651 0.0266 -2.4459    179               -0.0256              0.75  0.0144
                    pl 63            raw   0.0306 0.0220  1.3883    179                0.0009              0.50  0.1651
                    pl 63 sector_neutral   0.0871 0.0209  4.1723    179                0.0314              1.00  0.0000
                   pvp 63            raw   0.0339 0.0250  1.3593    179                0.0023              0.50  0.1741
                   pvp 63 sector_neutral   0.0869 0.0218  3.9934    179                0.0191              1.00  0.0001
             ev_ebitda 63            raw   0.0291 0.0199  1.4607    179                0.0008              0.50  0.1441
             ev_ebitda 63 sector_neutral   0.0429 0.0261  1.6410    179                0.0112              0.75  0.1008
                   roe 63            raw   0.0376 0.0188  1.9997    179                0.0004              0.50  0.0455
                   roe 63 sector_neutral   0.0551 0.0237  2.3243    179                0.0035              0.50  0.0201
            net_margin 63            raw   0.0469 0.0226  2.0774    179                0.0071              0.50  0.0378
            net_margin 63 sector_neutral   0.0518 0.0247  2.0939    179                0.0065              0.75  0.0363
          roe_trend_4q 63            raw  -0.0105 0.0196 -0.5383    169               -0.0168              0.75  0.5903
          roe_trend_4q 63 sector_neutral   0.0081 0.0224  0.3628    169               -0.0142              0.25  0.7168
           debt_equity 63            raw  -0.0110 0.0144 -0.7643    179                0.0012              0.75  0.4447
           debt_equity 63 sector_neutral  -0.0072 0.0210 -0.3455    179                0.0031              0.50  0.7297
         div_yield_12m 63            raw   0.0783 0.0208  3.7639    179                0.0271              1.00  0.0002
         div_yield_12m 63 sector_neutral   0.0783 0.0269  2.9135    179                0.0215              0.75  0.0036
momentum_vs_market_12m 63            raw   0.0979 0.0258  3.7931    179                0.0278              1.00  0.0001
momentum_vs_market_12m 63 sector_neutral   0.1022 0.0311  3.2859    179                0.0282              1.00  0.0010
momentum_vs_sector_12m 63            raw   0.0622 0.0223  2.7924    179                0.0257              0.75  0.0052
momentum_vs_sector_12m 63 sector_neutral   0.0757 0.0266  2.8427    179                0.0197              1.00  0.0045
        volatility_60d 63            raw  -0.0439 0.0277 -1.5872    179                0.0036              0.75  0.1125
        volatility_60d 63 sector_neutral  -0.0828 0.0343 -2.4177    179               -0.0087              0.75  0.0156
        turnover_ratio 63            raw  -0.0732 0.0198 -3.6971    179               -0.0194              0.75  0.0002
        turnover_ratio 63 sector_neutral  -0.1025 0.0283 -3.6231    179               -0.0232              0.75  0.0003
    amihud_illiquidity 63            raw   0.0280 0.0156  1.8010    179                0.0196              0.75  0.0717
    amihud_illiquidity 63 sector_neutral   0.0379 0.0196  1.9334    179                0.0141              0.75  0.0532
           pl_zhist_5y 63            raw   0.0085 0.0168  0.5039    161               -0.0027              0.50  0.6143
           pl_zhist_5y 63 sector_neutral   0.0107 0.0211  0.5069    160               -0.0064              0.75  0.6122
          roe_zhist_5y 63            raw   0.0348 0.0220  1.5823    161                0.0027              0.75  0.1136
          roe_zhist_5y 63 sector_neutral  -0.0009 0.0273 -0.0340    160               -0.0075              0.75  0.9729

## Gate results (sector-neutral variant)

        characteristic  k        variant  mean_ic  se_ic   tstat  n_obs  quintile_spread_mean  sign_consistency  pvalue  fdr_reject  passes_gate
        earnings_yield 21 sector_neutral   0.0169 0.0188  0.9017    180                0.0024              0.25  0.3672       False        False
        book_to_market 21 sector_neutral  -0.0476 0.0182 -2.6139    180               -0.0120              0.75  0.0090        True         True
                    pl 21 sector_neutral   0.0522 0.0172  3.0381    180                0.0117              0.75  0.0024        True         True
                   pvp 21 sector_neutral   0.0593 0.0160  3.7151    180                0.0068              1.00  0.0002        True         True
             ev_ebitda 21 sector_neutral   0.0353 0.0176  2.0090    180                0.0058              0.75  0.0445        True         True
                   roe 21 sector_neutral   0.0331 0.0163  2.0285    180                0.0002              0.50  0.0425        True        False
            net_margin 21 sector_neutral   0.0467 0.0172  2.7170    180                0.0065              0.75  0.0066        True         True
          roe_trend_4q 21 sector_neutral   0.0081 0.0173  0.4679    170               -0.0089              0.50  0.6399       False        False
           debt_equity 21 sector_neutral  -0.0071 0.0170 -0.4213    180                0.0011              0.50  0.6736       False        False
         div_yield_12m 21 sector_neutral   0.0519 0.0186  2.7959    180                0.0061              0.75  0.0052        True         True
momentum_vs_market_12m 21 sector_neutral   0.0876 0.0216  4.0499    180                0.0127              1.00  0.0001        True         True
momentum_vs_sector_12m 21 sector_neutral   0.0662 0.0183  3.6085    180                0.0074              1.00  0.0003        True         True
        volatility_60d 21 sector_neutral  -0.0739 0.0228 -3.2372    180               -0.0041              1.00  0.0012        True         True
        turnover_ratio 21 sector_neutral  -0.0689 0.0200 -3.4524    180               -0.0059              0.75  0.0006        True         True
    amihud_illiquidity 21 sector_neutral   0.0126 0.0167  0.7519    180               -0.0002              0.50  0.4521       False        False
           pl_zhist_5y 21 sector_neutral   0.0019 0.0194  0.0981    161               -0.0043              0.50  0.9219       False        False
          roe_zhist_5y 21 sector_neutral   0.0043 0.0199  0.2168    161               -0.0066              0.25  0.8283       False        False
        earnings_yield 63 sector_neutral   0.0175 0.0303  0.5778    179                0.0026              0.25  0.5634       False        False
        book_to_market 63 sector_neutral  -0.0651 0.0266 -2.4459    179               -0.0256              0.75  0.0144        True         True
                    pl 63 sector_neutral   0.0871 0.0209  4.1723    179                0.0314              1.00  0.0000        True         True
                   pvp 63 sector_neutral   0.0869 0.0218  3.9934    179                0.0191              1.00  0.0001        True         True
             ev_ebitda 63 sector_neutral   0.0429 0.0261  1.6410    179                0.0112              0.75  0.1008       False        False
                   roe 63 sector_neutral   0.0551 0.0237  2.3243    179                0.0035              0.50  0.0201        True        False
            net_margin 63 sector_neutral   0.0518 0.0247  2.0939    179                0.0065              0.75  0.0363        True         True
          roe_trend_4q 63 sector_neutral   0.0081 0.0224  0.3628    169               -0.0142              0.25  0.7168       False        False
           debt_equity 63 sector_neutral  -0.0072 0.0210 -0.3455    179                0.0031              0.50  0.7297       False        False
         div_yield_12m 63 sector_neutral   0.0783 0.0269  2.9135    179                0.0215              0.75  0.0036        True         True
momentum_vs_market_12m 63 sector_neutral   0.1022 0.0311  3.2859    179                0.0282              1.00  0.0010        True         True
momentum_vs_sector_12m 63 sector_neutral   0.0757 0.0266  2.8427    179                0.0197              1.00  0.0045        True         True
        volatility_60d 63 sector_neutral  -0.0828 0.0343 -2.4177    179               -0.0087              0.75  0.0156        True         True
        turnover_ratio 63 sector_neutral  -0.1025 0.0283 -3.6231    179               -0.0232              0.75  0.0003        True         True
    amihud_illiquidity 63 sector_neutral   0.0379 0.0196  1.9334    179                0.0141              0.75  0.0532        True        False
           pl_zhist_5y 63 sector_neutral   0.0107 0.0211  0.5069    160               -0.0064              0.75  0.6122       False        False
          roe_zhist_5y 63 sector_neutral  -0.0009 0.0273 -0.0340    160               -0.0075              0.75  0.9729       False        False