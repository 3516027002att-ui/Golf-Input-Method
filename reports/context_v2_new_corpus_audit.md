# Context reranker v2 data audit

本报告用于判断数据是否有资格进入上下文选词模型训练。

## 总览

- kept samples: 146259
- train: 116572
- val: 15686
- test: 14001

## 读取与修复/丢弃计数

### train
- kept: 116572
- physical_lines: 116572
- read: 116572

### val
- kept: 15686
- physical_lines: 15686
- read: 15686

### test
- kept: 14001
- physical_lines: 14001
- read: 14001

## Split: train

- samples: 116572
- avg candidates: 7.50
- median candidates: 8.00
- min/max candidates: 5 / 10

### Baselines

- candidates[0]: {"top1": 0.1581769207013691, "top3": 0.4700957348248293, "top5": 0.7815513159249219}
- static-rank: {"usable": 1.0, "top1": 0.1581769207013691, "top3": 0.4700957348248293, "top5": 0.7815513159249219}
- frequency: {"usable": 1.0, "top1": 0.1581769207013691, "top3": 0.4700957348248293, "top5": 0.7815513159249219}
- random: {"top1": 0.14084605522086222, "top3": 0.4225381656625866, "top5": 0.704230276104311}
- candidate recall buckets: {"recall@10": 1.0, "recall@30": 1.0, "recall@100": 1.0}

### Candidate generator audit

```json
{
  "target_position_distribution": [
    [
      0,
      18439
    ],
    [
      3,
      18226
    ],
    [
      2,
      18194
    ],
    [
      1,
      18167
    ],
    [
      4,
      18081
    ],
    [
      5,
      14513
    ],
    [
      6,
      10952
    ]
  ],
  "target_first_rate": 0.1581769207013691,
  "candidate_count_under_3_rate": 0.0,
  "target_match_type_distribution": [
    [
      "exact_target",
      116572
    ]
  ],
  "negative_match_type_distribution": [
    [
      "heuristic",
      758165
    ]
  ],
  "negative_source_distribution": [
    [
      "heuristic",
      758165
    ]
  ],
  "hard_negative_candidate_ratio": 1.0,
  "samples_without_hard_negative_rate": 0.0,
  "target_unique_known_match_rate": 0.0,
  "high_freq_conflict_sample_rate": 1.0
}
```

### Distributions

- target_index: [(0, 18439), (3, 18226), (2, 18194), (1, 18167), (4, 18081), (5, 14513), (6, 10952)]
- candidate_count: [(6, 19551), (8, 19536), (9, 19510), (10, 19412), (7, 19309), (5, 19254)]
- source_prefix: [('ultra_fineweb', 66754), ('lccc', 49818)]
- domain: [('web_text', 66754), ('chat', 49818)]
- license: [('apache2', 66754), ('research_only', 49818)]

## Split: val

- samples: 15686
- avg candidates: 7.00
- median candidates: 7.00
- min/max candidates: 5 / 8

### Baselines

- candidates[0]: {"top1": 0.15427769985974754, "top3": 0.44574780058651026, "top5": 0.7366441412724722}
- static-rank: {"usable": 1.0, "top1": 0.15427769985974754, "top3": 0.44574780058651026, "top5": 0.7366441412724722}
- frequency: {"usable": 1.0, "top1": 0.003187555782226189, "top3": 0.004207573632538569, "top5": 0.180989417314803}
- random: {"top1": 0.1475391310419361, "top3": 0.4426173931258083, "top5": 0.7376956552096805}
- candidate recall buckets: {"recall@10": 1.0, "recall@30": 1.0, "recall@100": 1.0}

### Candidate generator audit

```json
{
  "target_position_distribution": [
    [
      0,
      2420
    ],
    [
      3,
      2329
    ],
    [
      2,
      2324
    ],
    [
      1,
      2248
    ],
    [
      4,
      2234
    ],
    [
      5,
      1798
    ],
    [
      6,
      1318
    ],
    [
      7,
      1015
    ]
  ],
  "target_first_rate": 0.15427769985974754,
  "candidate_count_under_3_rate": 0.0,
  "target_match_type_distribution": [
    [
      "exact_pinyin",
      15686
    ]
  ],
  "negative_match_type_distribution": [
    [
      "exact_pinyin",
      51147
    ],
    [
      "exact_short",
      42573
    ],
    [
      "fuzzy_pinyin",
      333
    ]
  ],
  "negative_source_distribution": [
    [
      "rime_lmdg",
      91334
    ],
    [
      "cc_cedict",
      2386
    ],
    [
      "pycorrector",
      333
    ]
  ],
  "hard_negative_candidate_ratio": 1.0,
  "samples_without_hard_negative_rate": 0.0,
  "target_unique_known_match_rate": 0.0,
  "high_freq_conflict_sample_rate": 0.9968124442177738
}
```

### Distributions

- target_index: [(0, 2420), (3, 2329), (2, 2324), (1, 2248), (4, 2234), (5, 1798), (6, 1318), (7, 1015)]
- candidate_count: [(8, 7835), (5, 2657), (7, 2610), (6, 2584)]
- source_prefix: [('ultra_fineweb', 9250), ('lccc', 6436)]
- domain: [('web_text', 9250), ('chat', 6436)]
- license: [('apache2', 9250), ('research_only', 6436)]

## Split: test

- samples: 14001
- avg candidates: 6.98
- median candidates: 7.00
- min/max candidates: 5 / 8

### Baselines

- candidates[0]: {"top1": 0.15006070994928933, "top3": 0.4460395686022427, "top5": 0.7403756874508963}
- static-rank: {"usable": 1.0, "top1": 0.15006070994928933, "top3": 0.4460395686022427, "top5": 0.7403756874508963}
- frequency: {"usable": 1.0, "top1": 0.004928219412899079, "top3": 0.0064281122776944505, "top5": 0.18734376115991716}
- random: {"top1": 0.14792871937718735, "top3": 0.443786158131562, "top5": 0.7396435968859367}
- candidate recall buckets: {"recall@10": 1.0, "recall@30": 1.0, "recall@100": 1.0}

### Candidate generator audit

```json
{
  "target_position_distribution": [
    [
      1,
      2119
    ],
    [
      0,
      2101
    ],
    [
      4,
      2063
    ],
    [
      3,
      2058
    ],
    [
      2,
      2025
    ],
    [
      5,
      1545
    ],
    [
      6,
      1214
    ],
    [
      7,
      876
    ]
  ],
  "target_first_rate": 0.15006070994928933,
  "candidate_count_under_3_rate": 0.0,
  "target_match_type_distribution": [
    [
      "exact_pinyin",
      14001
    ]
  ],
  "negative_match_type_distribution": [
    [
      "exact_pinyin",
      46221
    ],
    [
      "exact_short",
      37074
    ],
    [
      "fuzzy_pinyin",
      435
    ]
  ],
  "negative_source_distribution": [
    [
      "rime_lmdg",
      81161
    ],
    [
      "cc_cedict",
      2134
    ],
    [
      "pycorrector",
      435
    ]
  ],
  "hard_negative_candidate_ratio": 1.0,
  "samples_without_hard_negative_rate": 0.0,
  "target_unique_known_match_rate": 0.0,
  "high_freq_conflict_sample_rate": 0.9950717805871009
}
```

### Distributions

- target_index: [(1, 2119), (0, 2101), (4, 2063), (3, 2058), (2, 2025), (5, 1545), (6, 1214), (7, 876)]
- candidate_count: [(8, 6910), (5, 2447), (7, 2352), (6, 2292)]
- source_prefix: [('ultra_fineweb', 7808), ('lccc', 6193)]
- domain: [('web_text', 7808), ('chat', 6193)]
- license: [('apache2', 7808), ('research_only', 6193)]

## Split leakage checks

- source_doc_key crossing splits: 0
- exact_sample crossing splits: 0
- no_context_sample crossing splits: 16
  examples: [{"signature": "2b53d740d1c473e73a0d2ba0d7235edd84af8e44", "splits": ["test", "val"], "sample_ids": ["7f48f94809b1ed34", "d7b56b8647d1da94"]}, {"signature": "a19db432eb0c563e40ba39c09be7a154cd5b1a98", "splits": ["test", "val"], "sample_ids": ["63d0822c73efbe0a", "04c098a39fb91fac"]}, {"signature": "5a87066b613db594dbf65a4e3eeda3d008931312", "splits": ["test", "val"], "sample_ids": ["266a858293ffcfbf", "bb8193164abe515f"]}, {"signature": "9339419482004efc8fea829fb582ead3978e50b9", "splits": ["test", "val"], "sample_ids": ["438be7af496d3039", "374aecec9e2c8b79"]}, {"signature": "8f6a9bffe9b62e4b837670bbca3ac8a53974a76b", "splits": ["test", "val"], "sample_ids": ["9e2a0375d91b583f", "16024cefe2db71bf"]}, {"signature": "9ca4780f0711bdd2261dbde55b3bfa0a078cdf27", "splits": ["test", "val"], "sample_ids": ["0d4d05522c7267db", "172599cff617efa4"]}, {"signature": "c8fa2f8607cd991488f5b5fa0cbdd8de1cf6dc67", "splits": ["test", "val"], "sample_ids": ["812aa5e47e70b20f", "bfc39c7e0fbf5ead"]}, {"signature": "09faa91da793921c1a723585ab8f67d0136b23c3", "splits": ["test", "val"], "sample_ids": ["18cd386e3e1a00fc", "cd5f3090a11cb566"]}, {"signature": "094c6685e6690c9215f07fc500510324f71166bf", "splits": ["test", "val"], "sample_ids": ["fe6baee159b02af7", "4640969b8aa66519"]}, {"signature": "15704591cbecb23ecc823db7589520d5b7c4742e", "splits": ["test", "val"], "sample_ids": ["1f9d7ea971f6a19d", "cd0b4906c489ed34"]}]
- raw_input_target crossing splits: 7625
  examples: [{"signature": "450c21469ee58ebdcd2757b9fdf1f426bbec6798", "splits": ["test", "train", "val"], "sample_ids": ["2f18a0907b72ef8a", "3191f4d4c71eb503", "2eb5371de99b25dc"]}, {"signature": "8d3a71257c9f0a8d5366407b05681e85ea2a5546", "splits": ["test", "train", "val"], "sample_ids": ["87c5e250d82a34a2", "dca0643231dbdfbd", "ddefdcc07d4f72e3"]}, {"signature": "ebe19da58659e37b4d00de000d92403afe2de472", "splits": ["test", "train", "val"], "sample_ids": ["5e5ab4bab5efe104", "0c178a2e412dc52f", "ecf2ced4a0519b25"]}, {"signature": "beaa29f37e6073d11dfe17f492afed2e379ba33a", "splits": ["train", "val"], "sample_ids": ["04fc1454a0231892", "ebc4a4b1e66abe15", "c48bcb89be90dffc"]}, {"signature": "35c53dd6bec77c300bc5cee0e4313171cd38eac5", "splits": ["test", "train", "val"], "sample_ids": ["ab39ab6a4fd18434", "fa439ada36acf42b", "c4cb0c91606a03f4"]}, {"signature": "8ff3ec60867c3da614767b3e84388d4a7fa04b7d", "splits": ["test", "train", "val"], "sample_ids": ["4e5dea6de17413a0", "46f488ab2854363b", "bb67d2d71a2b0d3a"]}, {"signature": "a6c2e6ee049f8cb9120e23c3546522fb80101b9f", "splits": ["test", "train", "val"], "sample_ids": ["2f9100beb61b3f3e", "ebdbef2406ff99c4", "cbb9bfb2a82ad666"]}, {"signature": "110e5142500c1b4c9c9fd4aaefc808d0564d3742", "splits": ["test", "train", "val"], "sample_ids": ["e37232fa9851c99f", "c2aac9691a1ba164", "292cdc67ccf7d2b4"]}, {"signature": "2fe4e63c0f434194ac5baf60df2d844360976399", "splits": ["test", "train", "val"], "sample_ids": ["b534555bbefb64c0", "c50f3fa39735b245", "7f04344c4a0346c2"]}, {"signature": "3bcb7fe844edb8d60a66d7a818d3f1843398422c", "splits": ["test", "train", "val"], "sample_ids": ["89cde72d0cd2a74e", "9ebe35a74b19166e", "53d06003ed253fe4"]}]
- context_suffix32_raw_input_target crossing splits: 229
  examples: [{"signature": "0b027ebbd6bf3a31f05de116ca2edba1b9bbb018", "splits": ["test", "train"], "sample_ids": ["9dbff34c8636f4de", "47d65fcd866e0d03", "4316441c0beb5791"]}, {"signature": "9aa7bddac92a2d5ef3046cd2398386f36542ad52", "splits": ["test", "train"], "sample_ids": ["f96e931d257a71be", "ccc32b8baec5dd5e", "b4e9dd010731dfe3"]}, {"signature": "8a4f762c1129d2ffd9bd18f117f77d5b0f588d53", "splits": ["test", "train"], "sample_ids": ["778b38ba3498a295", "6ed55d53359658dd", "0b6d4059e375bf26"]}, {"signature": "b7a3c940c53fec88c0abb6364c78ce9bfd48474c", "splits": ["test", "train"], "sample_ids": ["e3b3a09eca4b8eb3", "dc2869912b6aef52", "9f0b11aba0b51f14"]}, {"signature": "16998f9daea408643f523080d63899d89c61dd5d", "splits": ["train", "val"], "sample_ids": ["25bd04e9d67f0ad4", "cb8e98e1d84b1daf"]}, {"signature": "72b86e1b8cc4a013a0aa8edb9e33f93506cc69d9", "splits": ["test", "train"], "sample_ids": ["f166a19343caa0ea", "9708e62237da5158", "911aa6ce33dfe61e"]}, {"signature": "a615da4968382321880648d88c4589b439ca45e5", "splits": ["train", "val"], "sample_ids": ["95d3b7c1cff6983e", "57bd30dfbb42148b"]}, {"signature": "d8e83da4d417dfb41e57df45f2f487de3ff9c0b8", "splits": ["test", "train"], "sample_ids": ["f1316cd9bd1dbf6a", "ed9f1c6a85a84697", "b48e33a8820d0eda"]}, {"signature": "ee17b6b3e2fb0c9cd263ff90fc33c98ace186a5f", "splits": ["train", "val"], "sample_ids": ["7f1ee36239d57c6c", "7c345b5f0e44e34e"]}, {"signature": "80a8481e4d843362a4cd48f2f8175258fb688ee8", "splits": ["test", "train"], "sample_ids": ["2d68e4ca29435133", "4bbb6202ba772403"]}]
- candidates_set_target crossing splits: 2747
  examples: [{"signature": "c4c620be1b4566616745c03c8a55db3dde507074", "splits": ["test", "val"], "sample_ids": ["586d06b94fcabf98", "758f41d4254e6ef9", "a3ad872b4744c7c5"]}, {"signature": "02f59780f51905d6bb2d8ad88c59834cc97536ad", "splits": ["test", "val"], "sample_ids": ["7c0f4e1762b2e9d5", "ea736cf414bc401d", "61c4e3c9bcee0350"]}, {"signature": "ad1a3f9d5503eb0abb353ddc0dd64b3074d4bd48", "splits": ["test", "val"], "sample_ids": ["bd1f7c886f5a5a66", "bc13d18790f02bb2", "45698d5ddc23e318"]}, {"signature": "af38ba0fe3bdbafb073b864f505aa3f94ee6b9e4", "splits": ["test", "val"], "sample_ids": ["8c44ac6206c6ed20", "182834a3c93328be", "ca2e05a01a2e7111"]}, {"signature": "ed831914bab0766bae76975e3c67fd1cf1d41aa8", "splits": ["test", "val"], "sample_ids": ["3f0ec86a9ead5f58", "b1a78c24048652b4", "d1d9622b78f2d1ca"]}, {"signature": "a574bb55d0c773689752e93118494be7d9f8ba49", "splits": ["test", "val"], "sample_ids": ["f2ee2bcc1e636830", "6642b289046c15c3", "47603da903a774af"]}, {"signature": "1d51a5c3544bd5b1bcfc0e963984bc7c323cc67f", "splits": ["test", "val"], "sample_ids": ["dbf055da9dfcfe17", "2233660d06f4ff95"]}, {"signature": "0d412206d5dcad0b2d491a4814dd41b194d37fa7", "splits": ["test", "val"], "sample_ids": ["1da0c167b7b9b6d9", "3ae3b201df993f82", "f767f7991d845f89"]}, {"signature": "9a62aba46c497cf2a1cd6298649246857ed23dde", "splits": ["test", "val"], "sample_ids": ["b75c687088b42142", "c499fa10e454a795", "b5a34adb669137f2"]}, {"signature": "4beb33d414bcf07e1b6da4fc14b2cf1a5829813e", "splits": ["test", "val"], "sample_ids": ["7a004533e882d3f2", "63e672b55650e8d6", "23e95d85698b0f0f"]}]
- context_window_target crossing splits: 229
  examples: [{"signature": "bc4b5a015c33a0b8b590aba403cc1d79e2c124d6", "splits": ["test", "train"], "sample_ids": ["9dbff34c8636f4de", "47d65fcd866e0d03", "4316441c0beb5791"]}, {"signature": "0d24cf6149e2ca9039563856a1140ea0e77ba5da", "splits": ["test", "train"], "sample_ids": ["f96e931d257a71be", "ccc32b8baec5dd5e", "b4e9dd010731dfe3"]}, {"signature": "aa801c07dd8d56482de7ef951fd710a7b7ba9da3", "splits": ["test", "train"], "sample_ids": ["778b38ba3498a295", "6ed55d53359658dd", "0b6d4059e375bf26"]}, {"signature": "4662a02152f1c5a1654ee383d2ab1214e869cc41", "splits": ["test", "train"], "sample_ids": ["e3b3a09eca4b8eb3", "dc2869912b6aef52", "9f0b11aba0b51f14"]}, {"signature": "057ebb806250d4667c2cb2af2508977426e32fc8", "splits": ["train", "val"], "sample_ids": ["25bd04e9d67f0ad4", "cb8e98e1d84b1daf"]}, {"signature": "2d22c5ceb8d6ae32d5f84dd9b842b1fb7e3ee1b1", "splits": ["test", "train"], "sample_ids": ["f166a19343caa0ea", "9708e62237da5158", "911aa6ce33dfe61e"]}, {"signature": "d03882313cb099fcfc2f09f2bbbd2b409bd6d889", "splits": ["train", "val"], "sample_ids": ["95d3b7c1cff6983e", "57bd30dfbb42148b"]}, {"signature": "42a27865fa95102e1863765ee90bb141230dd7cc", "splits": ["test", "train"], "sample_ids": ["f1316cd9bd1dbf6a", "ed9f1c6a85a84697", "b48e33a8820d0eda"]}, {"signature": "eb827df2928dd498be5b549a2893e65e3c331749", "splits": ["train", "val"], "sample_ids": ["7f1ee36239d57c6c", "7c345b5f0e44e34e"]}, {"signature": "d437efcf084485882aeb0bc349fbefae27e72cb8", "splits": ["test", "train"], "sample_ids": ["2d68e4ca29435133", "4bbb6202ba772403"]}]
- near_context_window crossing splits: 236
  examples: [{"signature": "f3a3cd8e62f4a3778ecc2b3a4e00db82bc825a99", "splits": ["test", "train"], "sample_ids": ["9dbff34c8636f4de", "47d65fcd866e0d03", "4316441c0beb5791"]}, {"signature": "b6f156a4c34607ad28768c96914021a1d0545d57", "splits": ["test", "train"], "sample_ids": ["f96e931d257a71be", "ccc32b8baec5dd5e", "b4e9dd010731dfe3"]}, {"signature": "b2e6d6002adb59b789d94c963196d93549109544", "splits": ["test", "train"], "sample_ids": ["778b38ba3498a295", "6ed55d53359658dd", "0b6d4059e375bf26"]}, {"signature": "28eb11cadcfa7c1c8fcb63046f226f4f8da16b7e", "splits": ["test", "train"], "sample_ids": ["e3b3a09eca4b8eb3", "dc2869912b6aef52", "9f0b11aba0b51f14"]}, {"signature": "83af6211afd293908772d51f79481b669b8b39f1", "splits": ["train", "val"], "sample_ids": ["25bd04e9d67f0ad4", "cb8e98e1d84b1daf"]}, {"signature": "6b54b17132be92febe01810d32062335474c91ce", "splits": ["test", "train"], "sample_ids": ["f166a19343caa0ea", "9708e62237da5158", "911aa6ce33dfe61e"]}, {"signature": "56155854ff606989d9b11dec70f38f2f90f673d4", "splits": ["train", "val"], "sample_ids": ["95d3b7c1cff6983e", "57bd30dfbb42148b"]}, {"signature": "d58f03e0146451b27f0d7004746d54810463e7c6", "splits": ["test", "train"], "sample_ids": ["f1316cd9bd1dbf6a", "ed9f1c6a85a84697", "b48e33a8820d0eda"]}, {"signature": "1bb19c5d25ecfbc2efaa184d909f1a4028f4637d", "splits": ["train", "val"], "sample_ids": ["7f1ee36239d57c6c", "7c345b5f0e44e34e"]}, {"signature": "804274f07cee17b160666b95533aca77d991abb9", "splits": ["test", "train"], "sample_ids": ["2d68e4ca29435133", "4bbb6202ba772403"]}]

## 结论建议

- 先确认 original-rank / frequency baseline 是否已经过高。
- 再训练纯上下文 cross-encoder。
- 如果纯上下文模型没有明显超过 baseline，应优先修数据，而不是堆模型。
- 如果无上下文 ablation 接近正常模型，应视为任务定义或数据切分存在问题。

## Train-to-eval memorization baselines

```json
{
  "val": {
    "raw_input_to_target": {
      "train_keys": 25877.0,
      "eval_samples": 15686.0,
      "covered": 13197.0,
      "coverage": 0.8413234731607803,
      "hit_on_covered": 0.785329999242252,
      "top1_all": 0.6607165625398445
    },
    "context_suffix32_raw_input_to_target": {
      "train_keys": 114985.0,
      "eval_samples": 15686.0,
      "covered": 118.0,
      "coverage": 0.007522631646053806,
      "hit_on_covered": 0.9661016949152542,
      "top1_all": 0.007267627183475711
    },
    "candidates_set_to_target": {
      "train_keys": 116572.0,
      "eval_samples": 15686.0,
      "covered": 0.0,
      "coverage": 0.0,
      "hit_on_covered": 0.0,
      "top1_all": 0.0
    }
  },
  "test": {
    "raw_input_to_target": {
      "train_keys": 25877.0,
      "eval_samples": 14001.0,
      "covered": 11791.0,
      "coverage": 0.8421541318477251,
      "hit_on_covered": 0.7841574081926893,
      "top1_all": 0.6603814013284766
    },
    "context_suffix32_raw_input_to_target": {
      "train_keys": 114985.0,
      "eval_samples": 14001.0,
      "covered": 115.0,
      "coverage": 0.008213699021498464,
      "hit_on_covered": 0.991304347826087,
      "top1_all": 0.008142275551746304
    },
    "candidates_set_to_target": {
      "train_keys": 116572.0,
      "eval_samples": 14001.0,
      "covered": 0.0,
      "coverage": 0.0,
      "hit_on_covered": 0.0,
      "top1_all": 0.0
    }
  }
}
```

## Red-line findings

- train-val: raw_input+target overlap 0.7961 > 0.30
- train-test: raw_input+target overlap 0.8002 > 0.30
