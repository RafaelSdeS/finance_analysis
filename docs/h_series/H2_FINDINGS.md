# H2 Findings — Ridge Composite + Benchmark-Relative Construction A

**Verdict: FAIL**

Gate (evaluated on the capW-multiplicative primary variant): pre-2024 net IR > 0 with bootstrap CI excluding 0; OOS score-quintile monotonicity; direction replicates on the untouched 2024-2026 segment; survives 2x costs.

## Gate checklist

{
  "pre2024_ir_positive": true,
  "pre2024_ir_ci_excludes_zero": true,
  "quintile_monotone": false,
  "replicates_direction_post2024": true,
  "survives_2x_costs": true
}

## Chosen hyperparameters (pre-2024 selection only, sec 2)

{
  "k21_raw": 100.0,
  "k21_sector_neutral": 1000.0,
  "k63": 100.0
}
{
  "multiplicative": 2.0,
  "additive": 0.5
}

## Untilted cap-weight anchor (gamma=0 diagnostic baseline)

{
  "pre2024": {
    "mean_monthly_active": 0.006042233176666999,
    "se": 0.0027625692296992027,
    "tstat": 2.187178917259169,
    "ir_annualized": 0.9544389669706793,
    "n_obs": 62
  },
  "post2024": {
    "mean_monthly_active": 0.004543750394394249,
    "se": 0.0015778036964115831,
    "tstat": 2.8797944920069285,
    "ir_annualized": 1.8202587924880895,
    "n_obs": 29
  }
}

## Variant summary (pre-2024 stitched OOS / post-2024 confirmation segment)

{
  "capw_mult": {
    "gamma": 2.0,
    "pre2024": {
      "mean_monthly_active": 0.008175566076636206,
      "se": 0.0031945067636156067,
      "tstat": 2.559257713820877,
      "ir_annualized": 1.1168063432377622,
      "n_obs": 62
    },
    "pre2024_ir_ci": [
      1.1168063432377622,
      0.6262089355013253,
      2.316860243839698
    ],
    "post2024": {
      "mean_monthly_active": 0.005188459953072492,
      "se": 0.001632261783273599,
      "tstat": 3.178693519777645,
      "ir_annualized": 2.009186712475402,
      "n_obs": 29
    }
  },
  "ew_mult": {
    "gamma": 2.0,
    "pre2024": {
      "mean_monthly_active": 0.002006291213531324,
      "se": 0.0032871260119067135,
      "tstat": 0.6103481297230723,
      "ir_annualized": 0.26634311158932383,
      "n_obs": 62
    },
    "pre2024_ir_ci": [
      0.26634311158932383,
      -0.42335636340657307,
      1.4455513747881124
    ],
    "post2024": {
      "mean_monthly_active": -0.00041137912763612104,
      "se": 0.002486360216613912,
      "tstat": -0.16545435568316968,
      "ir_annualized": -0.10458029089355476,
      "n_obs": 29
    }
  },
  "capw_add": {
    "gamma": 0.5,
    "pre2024": {
      "mean_monthly_active": 0.007094184030718132,
      "se": 0.002960844957545278,
      "tstat": 2.39599983533067,
      "ir_annualized": 1.045564032118889,
      "n_obs": 62
    },
    "pre2024_ir_ci": [
      1.045564032118889,
      0.562603417020297,
      2.2696814524245403
    ],
    "post2024": {
      "mean_monthly_active": 0.005420400205405248,
      "se": 0.0017331980867606477,
      "tstat": 3.1273979857293703,
      "ir_annualized": 1.9767638617734153,
      "n_obs": 29
    }
  }
}

## Ablations

{
  "sector_neutral": {
    "pre2024": {
      "mean_monthly_active": 0.00664394004481829,
      "se": 0.0030101269635385798,
      "tstat": 2.2071959506345706,
      "ir_annualized": 0.9631739801448999,
      "n_obs": 62
    },
    "post2024": {
      "mean_monthly_active": 0.005174638175993331,
      "se": 0.0018004352431358155,
      "tstat": 2.8741040233030937,
      "ir_annualized": 1.8166619644087654,
      "n_obs": 29
    }
  },
  "size_beta_neutral": {
    "pre2024": {
      "mean_monthly_active": 0.008174302377259095,
      "se": 0.003280343596952744,
      "tstat": 2.4919043190635777,
      "ir_annualized": 1.0874147356253943,
      "n_obs": 62
    },
    "post2024": {
      "mean_monthly_active": 0.004296841137671397,
      "se": 0.0016425308761964304,
      "tstat": 2.6159880462165126,
      "ir_annualized": 1.6535121708809386,
      "n_obs": 29
    }
  },
  "single_characteristic": {
    "pre2024": {
      "mean_monthly_active": 0.008210269397086712,
      "se": 0.003252293995926344,
      "tstat": 2.524454863973083,
      "ir_annualized": 1.1016191101338557,
      "n_obs": 62
    },
    "post2024": {
      "mean_monthly_active": 0.004169317715935653,
      "se": 0.0018194640223778565,
      "tstat": 2.2915087435951462,
      "ir_annualized": 1.448415485955574,
      "n_obs": 29
    }
  },
  "cost_2x": {
    "pre2024": {
      "mean_monthly_active": 0.00811130001806327,
      "se": 0.0031967940051039275,
      "tstat": 2.537323332411458,
      "ir_annualized": 1.1072346396298192,
      "n_obs": 62
    },
    "post2024": {
      "mean_monthly_active": 0.0051366864539737775,
      "se": 0.0016328989576133407,
      "tstat": 3.1457466673146777,
      "ir_annualized": 1.9883616855344242,
      "n_obs": 29
    }
  },
  "quarterly_rebalance": {
    "pre2024": {
      "mean_monthly_active": 0.007700325047441636,
      "se": 0.003199098519844368,
      "tstat": 2.4070296677878638,
      "ir_annualized": 1.05037721946869,
      "n_obs": 62
    },
    "post2024": {
      "mean_monthly_active": 0.005060954950192929,
      "se": 0.001609434962522193,
      "tstat": 3.1445538763876217,
      "ir_annualized": 1.9876077469535294,
      "n_obs": 29
    }
  },
  "permutation_null": {
    "observed_ir": 1.1168063432377622,
    "null_mean": 1.2747886367158958,
    "null_std": 0.2176162949180762,
    "p_value": 0.7761194029850746
  }
}

## Quintile monotonicity

{
  "monotone": false,
  "means": {
    "0": -0.01745829881082322,
    "1": -0.005616224863631801,
    "2": 0.0020719077351736043,
    "3": 0.0034316416756489583,
    "4": 0.003376637570115612
  }
}

## Interpretation: the capW-mult IR is mostly an anchor effect, not composite skill

The untilted cap-weight anchor ALONE (gamma=0, no ridge tilt at all -- just holding the top-50 universe cap-weighted) already has a pre-2024 IR of 0.95 vs BOVA11, t=2.19 -- significant on its own. The tilted primary variant's IR (1.12) is barely above this baseline. This is a genuinely new, until-now-undiscovered structural fact: the top-50 cap-weighted BASKET itself outperforms BOVA11/IBOV over this window (a universe/index-composition effect, e.g. IBOV's broader and differently-weighted constituent set), independent of any stock selection -- something neither H0 (which only tested an equal-weight UCRP anchor) nor the R-series ever measured. **The date-permutation null confirms this directly**: shuffling the composite's score cross-sections across dates (same anchor, same construction, same real forward returns -- only WHICH score is real vs. borrowed changes) produces a null IR distribution centered at 1.27, ABOVE the observed real-signal IR (1.12) -- permutation p=0.78. A random, uncorrelated score fed through the identical cap-weight-anchor-plus-cap-loop machinery does just as well or better. **The composite's incremental contribution beyond the anchor is statistically indistinguishable from noise.** Combined with the quintile-monotonicity failure at the top end (Q4 does not clear Q3), the FAIL verdict is correct and this is why: what looked like a positive, significant, cost-surviving, replicating IR was mostly a beneficial anchor/universe choice, not evidence the H1 survivors combine into real stock-picking skill at the portfolio level.
