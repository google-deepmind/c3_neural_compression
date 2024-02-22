# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dict containing results and meta-info for codecs on the CLIC2020 dataset."""

import immutabledict

RESULTS = immutabledict.immutabledict({
    'C3': immutabledict.immutabledict({
        'bpp': (
            0.061329951827845924,
            0.09172627638752867,
            0.1044846795408464,
            0.13844043170896972,
            0.15884458128272033,
            0.1886409727356783,
            0.23819740470953105,
            0.3481186962709194,
            0.39229455372182337,
            0.49699320480590914,
            0.5581800268917549,
            0.6436851537082253,
            0.7813889435151729,
            1.0634540579226888,
        ),
        'psnr': (
            29.19010436825636,
            30.561563584862686,
            31.03043723687893,
            32.03827918448099,
            32.53981864743116,
            33.179984534659035,
            34.10305776828673,
            35.70015777029642,
            36.241154647454984,
            37.32775478828244,
            37.85520395418493,
            38.517472895180305,
            39.45305717282179,
            40.98722505447908,
        ),
        'meta': immutabledict.immutabledict({
            'source': 'Our experiments',
            'reference': 'https://arxiv.org/abs/2312.02753',
            'type': 'neural-field',
            'data': 'single',
        }),
    }),
    'C3 (Adaptive)': immutabledict.immutabledict({
        'bpp': (
            0.0537973679829298,
            0.08524668893617827,
            0.09885586530151891,
            0.13353873225973872,
            0.15376036841331459,
            0.18444221521296153,
            0.23472510559893237,
            0.34506475289420385,
            0.3886571886335931,
            0.4945863474433015,
            0.5539876204438325,
            0.6406401773778404,
            0.7767906178061555,
            1.0594375075363531,
        ),
        'psnr': (
            29.108123779296875,
            30.52411139883646,
            31.013311339587702,
            32.039528730438974,
            32.54025808194788,
            33.17999630439572,
            34.11839085090451,
            35.71286406168124,
            36.236318355653346,
            37.32584855614639,
            37.85083826576791,
            38.522516111048255,
            39.45337286228087,
            40.99067976416611,
        ),
        'meta': immutabledict.immutabledict({
            'source': 'Our experiments',
            'reference': 'https://arxiv.org/abs/2312.02753',
            'type': 'neural-field',
            'data': 'single',
        }),
    }),
    'COOL-CHICv2': immutabledict.immutabledict({
        'bpp': (
            0.05617740145537679,
            0.16940205168571099,
            0.39484573211758134,
            0.63885202794137,
            1.2008880400357773,
        ),
        'psnr': (
            28.252848169823015,
            31.761616862374357,
            35.14136717075522,
            37.50671367416232,
            40.9451591220996,
        ),
        'psnr_of_mean_mse': (
            27.335227966308594,
            30.96285057067871,
            34.612213134765625,
            37.13885498046875,
            40.76046371459961,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/Orange-OpenSource/Cool-Chic/tree/main/results/clic20-pro-valid'
                ' (accessed 31/08/23)'
            ),
            'reference': 'https://arxiv.org/abs/2307.12706',
            'type': 'neural-field',
            'data': 'single',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 2300,
                'max': 2300,
                'source': (
                    'Obtained from paper, using the main model. Numbers were'
                    ' calculated using the fvcore library.'
                ),
            }),
        }),
    }),
    'CST': immutabledict.immutabledict({
        'bpp': (
            0.077656171,
            0.13373046,
            0.226810018,
            0.341805144,
            0.507908136,
            0.624,
        ),
        'psnr': (
            29.91564565,
            31.62525228,
            33.48489175,
            35.15752778,
            36.61327623,
            37.431,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/RDdata/data_CLIC_Proposed_optimized_by_MSE_PSNR.dat'
                ' (accessed 08/09/23)'
            ),
            'reference': 'https://arxiv.org/abs/2001.01568',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 260_286,
                'max': 583_058,
                'source': (
                    'Calculated using the CompressAI version of this model,'
                    ' with the fvcore library.'
                ),
            }),
        }),
    }),
    'BPG': immutabledict.immutabledict({
        'bpp': (
            0.103564713062667,
            0.170360743528309,
            0.268199464988708,
            0.407668363237427,
            0.597277598101488,
            0.760399946168634,
        ),
        'psnr': (
            29.9515956506731,
            31.4871618726295,
            33.0915804299117,
            34.7457636096469,
            36.4276668709393,
            37.5768207279668,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'Provided by Wei Jiang (https://jiangweibeta.github.io/).'
            ),
            'reference': 'BPG. BPG version b0.9.8',
            'type': 'classical',
        }),
    }),
    'VTM': immutabledict.immutabledict({
        'bpp': (
            0.0342348894214245,
            0.0479473243024482,
            0.0926004626984371,
            0.1272971827916389,
            0.1723905215563916,
            0.2286193155703299,
            0.2997856082010903,
            0.3875806077629035,
            0.4967919211746751,
            0.6348070671147779,
            0.8063403489780799,
        ),
        'psnr': (
            27.9705622242058567,
            28.8967498073586810,
            30.8124872629909987,
            31.8272356017229328,
            32.8972760150943557,
            33.9262030080800727,
            34.9725092902167844,
            36.0125398883515402,
            37.0613097130305107,
            38.1335157646361083,
            39.2055415081168945,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'Provided by Wei Jiang (https://jiangweibeta.github.io/).'
            ),
            'reference': (
                'https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM, VTM-17.0'
                ' Intra.'
            ),
            'type': 'classical',
        }),
    }),
    'MLIC': immutabledict.immutabledict({
        'bpp': (0.085, 0.1377, 0.2078, 0.3121, 0.4373, 0.6094),
        'psnr': (31.0988, 32.5423, 33.9524, 35.383, 36.7549, 38.1868),
        'meta': immutabledict.immutabledict({
            'source': 'Obtained from paper authors',
            'reference': 'https://arxiv.org/abs/2211.07273',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 446_750,
                'max': 446_750,
                'source': (
                    'Obtained from paper authors, who calculated the'
                    ' numbers using the DeepSpeed library.'
                ),
            }),
        }),
    }),
    'MLIC+': immutabledict.immutabledict({
        'bpp': (0.0829, 0.1327, 0.2009, 0.302, 0.4176, 0.5850),
        'psnr': (31.1, 32.5593, 33.9739, 35.4409, 36.7843, 38.1206),
        'meta': immutabledict.immutabledict({
            'source': 'Obtained from paper authors',
            'reference': 'https://arxiv.org/abs/2211.07273',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 555_340,
                'max': 555_340,
                'source': (
                    'Obtained from paper authors, who calculated the'
                    ' numbers using the DeepSpeed library.'
                ),
            }),
        }),
    }),
    'STF': immutabledict.immutabledict({
        'bpp': (0.092, 0.144, 0.223, 0.320, 0.483, 0.661),
        'psnr': (30.88, 32.24, 33.70, 35.27, 36.90, 38.42),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/Googolxx/STF/blob/main/results/stf_mse_CLIC%20.json'
                ' (accessed 16/10/23)'
            ),
            'reference': 'https://arxiv.org/abs/2203.08450',
            'type': 'autoencoder',
            'data': 'multi',
        }),
    }),
    'WYH': immutabledict.immutabledict({
        'bpp': (
            0.1415,
            0.2131,
            0.2888,
            0.5616,
            0.7777,
            0.8900,
        ),
        'psnr': (
            32.3069,
            33.6667,
            34.8301,
            37.6477,
            39.2067,
            39.9138,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/Dezhao-Wang/Neural-Syntax-Code/blob/main/rd_points.dat'
                ' (accessed 16/10/23)'
            ),
            'reference': 'https://arxiv.org/abs/2203.04963',
            'type': 'autoencoder',
            'data': 'multi',
        }),
    }),
})
