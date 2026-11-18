window.RippleRelativeData = {
  "source": {
    "url": "https://www.ricursive.com/ripple",
    "module": "https://framerusercontent.com/sites/65flFwVmwsqUH0wMjInREr/0bTELr9D72LknJERbSOpjth7QaW4SmOr_AbPkiFMKdI.Byjjw4Qb.mjs",
    "retrieved_date": "2026-11-05"
  },
  "referenceCompute": 1e+20,
  "method": {
    "optimizer": "L-BFGS-B",
    "objective": "Huber on raw loss residuals",
    "delta": 0.01,
    "fitPoints": "All reported points of each selected baseline",
    "definition": "Equivalent baseline compute at the comparison recipe's reported loss divided by its reported compute",
    "comparisonNote": "v1.5 is compared with v1.3 because v1.4 was rejected; the recipe ancestry is not explicitly documented."
  },
  "defaultBaseline": "v1.0",
  "defaultSeries": "Final (RI-Mix-14T)",
  "fits": {
    "Olmo 3": {
      "E": 1.0178691086971305,
      "K": 0.4212698329759266,
      "alpha": 0.27136434814979166,
      "rmse": 0.015730393603517582,
      "objective": 0.0006505943187153018,
      "computeRange": [
        4.35e+18,
        5.69e+21
      ],
      "lossRange": [
        1.14,
        2.01
      ],
      "maxAbsResidual": 0.024790941358840746,
      "linearTailCount": 4,
      "fitPointCount": 7
    },
    "v1.0": {
      "E": 0.6919949059260179,
      "K": 0.7406921422351113,
      "alpha": 0.15393094955981362,
      "rmse": 0.003077072971161167,
      "objective": 2.8405134209551828e-05,
      "computeRange": [
        4.35e+18,
        1.21e+21
      ],
      "lossRange": [
        1.199,
        1.89
      ],
      "maxAbsResidual": 0.004801161318126024,
      "linearTailCount": 0,
      "fitPointCount": 6
    },
    "v1.1": {
      "E": 0.7457470403039693,
      "K": 0.6770405092527968,
      "alpha": 0.16852963979654345,
      "rmse": 0.0027227994544445225,
      "objective": 1.853409217280847e-05,
      "computeRange": [
        4.28e+18,
        1.78e+21
      ],
      "lossRange": [
        1.161,
        1.898
      ],
      "maxAbsResidual": 0.004181470956197142,
      "linearTailCount": 0,
      "fitPointCount": 5
    },
    "v1.2": {
      "E": 0.7533956192597361,
      "K": 0.6595919008820459,
      "alpha": 0.17617445590881683,
      "rmse": 0.0016409715062280724,
      "objective": 6.731968710631073e-06,
      "computeRange": [
        4.28e+18,
        1.78e+21
      ],
      "lossRange": [
        1.149,
        1.903
      ],
      "maxAbsResidual": 0.002641418826247799,
      "linearTailCount": 0,
      "fitPointCount": 5
    },
    "v1.3": {
      "E": 0.7483224877934078,
      "K": 0.6556302556995687,
      "alpha": 0.17410000547011392,
      "rmse": 0.004113988102481714,
      "objective": 4.231224526840273e-05,
      "computeRange": [
        4.58e+18,
        2.02e+21
      ],
      "lossRange": [
        1.133,
        1.871
      ],
      "maxAbsResidual": 0.005910725845868381,
      "linearTailCount": 0,
      "fitPointCount": 5
    },
    "v1.5": {
      "E": 0.7303121525803535,
      "K": 0.6637327642854168,
      "alpha": 0.171618323736314,
      "rmse": 0.0007280654306334387,
      "objective": 1.3251981782086363e-06,
      "computeRange": [
        4.58e+18,
        2.02e+21
      ],
      "lossRange": [
        1.127,
        1.857
      ],
      "maxAbsResidual": 0.0012149196985484778,
      "linearTailCount": 0,
      "fitPointCount": 5
    },
    "v1.6": {
      "E": 0.72420316768128,
      "K": 0.6531452454437798,
      "alpha": 0.1659773480299028,
      "rmse": 0.0014464534470328995,
      "objective": 5.230568936083393e-06,
      "computeRange": [
        4.58e+18,
        2.02e+21
      ],
      "lossRange": [
        1.121,
        1.814
      ],
      "maxAbsResidual": 0.002501496946604176,
      "linearTailCount": 0,
      "fitPointCount": 5
    }
  },
  "series": [
    {
      "key": "v1.0",
      "name": "Ricursive v1.0",
      "color": "#c3cfe2",
      "marker": "circle",
      "points": [
        [
          4350000000000000000,
          1.89
        ],
        [
          16500000000000000000,
          1.674
        ],
        [
          36000000000000000000,
          1.56
        ],
        [
          69000000000000000000,
          1.475
        ],
        [
          196000000000000000000,
          1.355
        ],
        [
          1.21e+21,
          1.199
        ]
      ],
      "baselineKey": "Olmo 3",
      "comparisonLabel": "OLMo 3 \u2192 v1.0"
    },
    {
      "key": "v1.1",
      "name": "Ricursive v1.1",
      "color": "#a9b8cf",
      "marker": "circle",
      "points": [
        [
          4280000000000000000,
          1.898
        ],
        [
          39600000000000000000,
          1.533
        ],
        [
          138000000000000000000,
          1.391
        ],
        [
          426000000000000000000,
          1.277
        ],
        [
          1.78e+21,
          1.161
        ]
      ],
      "baselineKey": "v1.0",
      "comparisonLabel": "v1.0 \u2192 v1.1"
    },
    {
      "key": "v1.2",
      "name": "Ricursive v1.2",
      "color": "#8ea0bb",
      "marker": "circle",
      "points": [
        [
          4280000000000000000,
          1.903
        ],
        [
          39600000000000000000,
          1.528
        ],
        [
          138000000000000000000,
          1.377
        ],
        [
          426000000000000000000,
          1.267
        ],
        [
          1.78e+21,
          1.149
        ]
      ],
      "baselineKey": "v1.1",
      "comparisonLabel": "v1.1 \u2192 v1.2"
    },
    {
      "key": "v1.3",
      "name": "Ricursive v1.3",
      "color": "#7488a8",
      "marker": "circle",
      "points": [
        [
          4580000000000000000,
          1.871
        ],
        [
          44100000000000000000,
          1.499
        ],
        [
          152000000000000000000,
          1.36
        ],
        [
          481000000000000000000,
          1.253
        ],
        [
          2.02e+21,
          1.133
        ]
      ],
      "baselineKey": "v1.2",
      "comparisonLabel": "v1.2 \u2192 v1.3"
    },
    {
      "key": "v1.4",
      "name": "Ricursive v1.4",
      "color": "#5a7195",
      "marker": "circle",
      "points": [
        [
          4580000000000000000,
          1.835
        ],
        [
          44100000000000000000,
          1.481
        ],
        [
          152000000000000000000,
          1.349
        ],
        [
          481000000000000000000,
          1.232
        ],
        [
          2.02e+21,
          1.127
        ]
      ],
      "baselineKey": "v1.3",
      "comparisonLabel": "v1.3 \u2192 v1.4"
    },
    {
      "key": "v1.5",
      "name": "Ricursive v1.5",
      "color": "#405982",
      "marker": "circle",
      "points": [
        [
          4580000000000000000,
          1.857
        ],
        [
          44100000000000000000,
          1.494
        ],
        [
          152000000000000000000,
          1.349
        ],
        [
          481000000000000000000,
          1.236
        ],
        [
          2.02e+21,
          1.127
        ]
      ],
      "baselineKey": "v1.3",
      "comparisonLabel": "v1.3 \u2192 v1.5"
    },
    {
      "key": "v1.6",
      "name": "Ricursive v1.6",
      "color": "#25416e",
      "marker": "circle",
      "points": [
        [
          4580000000000000000,
          1.814
        ],
        [
          44100000000000000000,
          1.471
        ],
        [
          152000000000000000000,
          1.336
        ],
        [
          481000000000000000000,
          1.226
        ],
        [
          2.02e+21,
          1.121
        ]
      ],
      "baselineKey": "v1.5",
      "comparisonLabel": "v1.5 \u2192 v1.6"
    },
    {
      "key": "Final (RI-Mix-14T)",
      "name": "Final",
      "color": "#0b2a5b",
      "marker": "star",
      "points": [
        [
          4580000000000000000,
          1.57
        ],
        [
          44100000000000000000,
          1.309
        ],
        [
          152000000000000000000,
          1.196
        ],
        [
          481000000000000000000,
          1.103
        ],
        [
          2.02e+21,
          1.013
        ],
        [
          8.33e+21,
          0.927
        ]
      ],
      "baselineKey": "v1.6",
      "comparisonLabel": "v1.6 \u2192 Final Ripple"
    }
  ]
};
