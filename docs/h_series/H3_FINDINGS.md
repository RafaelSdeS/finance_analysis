# H3 Findings — Fitted Portfolio Construction

**Verdict: FAIL**

Gate: blended IR exceeds anchor-alone IR with both cross-date and within-date permutation-null tests at p < 0.10 on both pre-2024 and post-2024 splits; OOS score-quintile monotonicity on both splits; bootstrap CI on the IR delta (blended vs. anchor-alone) excludes 0 on both splits; the anchor alone beats BOVA11 and CDI pre-2024.

## Gate checklist

{
  "cross_date_p_pre_below_0.10": false,
  "within_date_p_pre_below_0.10": true,
  "cross_date_p_post_below_0.10": false,
  "within_date_p_post_below_0.10": true,
  "quintile_monotone_pre2024": false,
  "quintile_monotone_post2024": true,
  "ir_delta_ci_excludes_zero_pre2024": true,
  "ir_delta_ci_excludes_zero_post2024": false,
  "anchor_alone_beats_bova11_pre2024": false,
  "anchor_alone_beats_cdi_pre2024": true
}

## Combination layer (Design §a): model class selected by walk-forward CV

{
  "class": "ridge",
  "hyperparams": {
    "alpha": 100.0
  },
  "mean_pre2024_oof_ic": 0.06968284928106845,
  "diagnostics": {
    "ic_by_class": {
      "ridge": 0.06968284928106845,
      "elasticnet": 0.06570555719710938,
      "gbm": 0.046337223223841324
    },
    "hyperparams_by_class": {
      "ridge": {
        "alpha": 100.0
      },
      "elasticnet": {
        "alpha": 0.01,
        "l1_ratio": 0.2
      },
      "gbm": {
        "max_depth": 3,
        "max_features": "sqrt",
        "min_samples_leaf": 753
      }
    },
    "train_oos_ic_gap_by_class": {
      "ridge": 0.0005706096649733955,
      "elasticnet": -0.009074814420627908,
      "gbm": 0.13514492111432938
    }
  }
}

## Anchor layer (Design §b): type selected by pre-2024 anchor-alone Sharpe/IR

{
  "winning_type": "risk_parity",
  "by_type": {
    "min_variance": {
      "pre2024": {
        "mean_monthly_active": -0.0038412447777153346,
        "se": 0.0040801909867085985,
        "tstat": -0.9414374940360287,
        "ir_annualized": -0.410823560059371,
        "n_obs": 62
      },
      "post2024": {
        "mean_monthly_active": 0.002256789730406343,
        "se": 0.004465163246508412,
        "tstat": 0.5054215503030189,
        "ir_annualized": 0.3194665533966228,
        "n_obs": 29
      }
    },
    "risk_parity": {
      "pre2024": {
        "mean_monthly_active": -0.0011983922862573526,
        "se": 0.0026701389647680943,
        "tstat": -0.448812703035265,
        "ir_annualized": -0.19585244227989132,
        "n_obs": 62
      },
      "post2024": {
        "mean_monthly_active": -0.0027476354954486066,
        "se": 0.0021267547483502713,
        "tstat": -1.2919381031498596,
        "ir_annualized": -0.8166074690871583,
        "n_obs": 29
      }
    }
  }
}

## Blend layer (Design §c): kappa = 0.8181, derived from the winning model's own pre-2024 OOF mean Spearman IC

{
  "pre2024": {
    "mean_monthly_active": 0.0014291999046902037,
    "se": 0.00312208892055659,
    "tstat": 0.4577704034244525,
    "ir_annualized": 0.19976139469271117,
    "n_obs": 62
  },
  "pre2024_ir_ci": [
    0.19976139469271117,
    -0.49428030457078237,
    1.3915079845723612
  ],
  "post2024": {
    "mean_monthly_active": -0.0008320752624494962,
    "se": 0.0022704660607508506,
    "tstat": -0.36647773637027026,
    "ir_annualized": -0.23164302999073572,
    "n_obs": 29
  },
  "post2024_ir_ci": [
    -0.23164302999073572,
    -2.1125364597436818,
    1.0464776571827357
  ]
}

## Bootstrap CI on the IR delta (blended vs. anchor-alone)

{
  "pre2024": [
    0.3956138369726025,
    0.0548039673953546,
    0.9003691468900868
  ],
  "post2024": [
    0.5849644390964226,
    -0.36610650838346526,
    2.1345996872284863
  ]
}

## Permutation-null tests (cross-date and within-date, both splits)

{
  "pre2024": {
    "observed_ir": 0.19976139469271117,
    "cross_date": {
      "null_mean": 0.0819125277061748,
      "p_value": 0.23383084577114427
    },
    "within_date": {
      "null_mean": -0.1812620972400515,
      "p_value": 0.009950248756218905
    }
  },
  "post2024": {
    "observed_ir": -0.23164302999073572,
    "cross_date": {
      "null_mean": -0.1708728402724506,
      "p_value": 0.6069651741293532
    },
    "within_date": {
      "null_mean": -1.4853666106011791,
      "p_value": 0.004975124378109453
    }
  }
}

## Quintile monotonicity

{
  "pre2024": {
    "monotone": false,
    "means": {
      "0": -0.014021563595407021,
      "1": -0.005898967214019705,
      "2": 0.004163295260397852,
      "3": 0.005098322278302021,
      "4": 0.003937051706070904
    }
  },
  "post2024": {
    "monotone": true,
    "means": {
      "0": -0.025350061157334486,
      "1": -0.004966964651629947,
      "2": -0.002730537693119108,
      "3": -0.00039555081933214666,
      "4": 0.002089760665329387
    }
  }
}
