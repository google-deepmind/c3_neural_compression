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

"""Dict containing results and meta-info for codecs on the Kodak dataset."""

import immutabledict

RESULTS = immutabledict.immutabledict({
    'C3': immutabledict.immutabledict({
        'bpp': (
            0.09261394,
            0.1339127,
            0.15318637,
            0.2025493,
            0.23175092,
            0.27606731,
            0.35064952,
            0.51440305,
            0.57850416,
            0.72855291,
            0.8066705,
            0.91400582,
            1.07998864,
            1.40934119,
        ),
        'psnr': (
            27.06491995,
            28.45003923,
            28.9589448,
            30.03212317,
            30.59409277,
            31.33577712,
            32.39495262,
            34.36792223,
            35.0161012,
            36.38384724,
            37.03668944,
            37.84491142,
            38.98360984,
            40.84745836,
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
            0.07928384902576606,
            0.12279809592291713,
            0.14090672228485346,
            0.19467594816039005,
            0.22516465404381356,
            0.2717756511022647,
            0.34901742078363895,
            0.5145895009239515,
            0.5783201344311237,
            0.7276818454265594,
            0.806981734931469,
            0.9142339217166106,
            1.0806963220238686,
            1.407525323331356,
        ),
        'psnr': (
            27.032467524210613,
            28.376633564631145,
            28.838989973068237,
            29.946287473042805,
            30.48921473821004,
            31.26695728302002,
            32.39169502258301,
            34.3686675230662,
            35.026323318481445,
            36.396727561950684,
            37.062685330708824,
            37.881666819254555,
            39.02162663141886,
            40.883299032847084,
        ),
        'meta': immutabledict.immutabledict({
            'source': 'Our experiments',
            'reference': 'https://arxiv.org/abs/2312.02753',
            'type': 'neural-field',
            'data': 'single',
        }),
    }),
    'COIN': immutabledict.immutabledict({
        'bpp': (0.0368, 0.07361, 0.1604, 0.3074, 0.6109, 1.209),
        'psnr': (22.408, 23.205, 24.6410833, 26.412125, 28.408, 30.644),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/EmilienDupont/coin/blob/main/results.json'
                ' (accessed 10/08/23)'
            ),
            'reference': 'https://arxiv.org/abs/2103.03123',
            'type': 'neural-field',
            'data': 'single',
        }),
    }),
    'COIN++': immutabledict.immutabledict({
        'bpp': (
            0.06656413525342941,
            0.1324574202299118,
            0.2659042477607727,
            0.3982515782117843,
            0.5365909337997437,
        ),
        'psnr': (
            23.68178367614746,
            24.926698684692383,
            26.142541885375977,
            27.031984329223633,
            27.798105239868164,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/EmilienDupont/coinpp/blob/main/results/kodak/coinpp.json'
                ' (accessed 10/08/23)'
            ),
            'reference': 'https://arxiv.org/abs/2201.12904',
            'type': 'neural-field',
            'data': 'multi',
        }),
    }),
    'COOL-CHIC': immutabledict.immutabledict({
        'bpp': (
            0.07709166666666667,
            0.12764166666666665,
            0.14934166666666668,
            0.21052916666666666,
            0.24633750000000001,
            0.30772499999999997,
            0.4031208333333333,
            0.6132583333333333,
            0.7201291666666666,
            0.9081083333333333,
            1.0143791666666666,
            1.1614041666666666,
            1.3463083333333332,
            1.7372041666666664,
        ),
        'psnr': (
            25.699479166666674,
            27.141000000000002,
            27.591166666666666,
            28.690179166666667,
            29.217024999999996,
            30.095520833333335,
            31.258812499999994,
            33.31259166666667,
            34.240945833333335,
            35.804429166666665,
            36.55010416666667,
            37.5628625,
            38.766983333333336,
            40.870895833333336,
        ),
        'psnr_of_mean_mse': (
            25.26493263244629,
            26.728336334228516,
            27.196231842041016,
            28.354198455810547,
            28.91071128845215,
            29.794952392578125,
            31.016834259033203,
            33.130767822265625,
            34.090877532958984,
            35.69075012207031,
            36.437191009521484,
            37.47454833984375,
            38.69655227661133,
            40.82229995727539,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/Orange-OpenSource/Cool-Chic/tree/main/results/kodak'
                ' (accessed 20/03/23)'
            ),
            'reference': 'https://arxiv.org/abs/2212.05458',
            'type': 'neural-field',
            'data': 'single',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 800,
                'max': 800,
                'source': (
                    'Obtained from paper with correction for bicubic'
                    ' upsampling. Paper reports 680 MACs/pixel but ignores'
                    ' bicubic upsampling which costs approximately 120'
                    ' MACs/pixel, hence giving a total of 800 MACs/pixel. Note'
                    ' that bicubic upsampling costs 20 MACs/pixel and is'
                    ' applied to 6 grids, for a total of 6 * 20 = 120'
                    ' MACs/pixel.'
                ),
            }),
        }),
    }),
    'COOL-CHICv2': immutabledict.immutabledict({
        'bpp': (
            0.09670342339409717,
            0.26352606879340273,
            0.599063449435764,
            0.92889404296875,
            1.6057874891493054,
        ),
        'psnr': (
            26.393401128565245,
            30.059626917963936,
            33.92421188208954,
            36.81516599773773,
            40.96041189847251,
        ),
        'psnr_of_mean_mse': (
            25.90398597717285,
            29.674453735351562,
            33.75265121459961,
            36.69748306274414,
            40.92167663574219,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/Orange-OpenSource/Cool-Chic/tree/main/results/kodak'
                ' (accessed 10/08/23)'
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
    'RECOMBINER': immutabledict.immutabledict({
        'bpp': (
            0.07401529947917,
            0.12967936197917,
            0.17793782552083,
            0.31616210938,
            0.48844401041667,
            0.97241210938,
        ),
        'psnr': (
            26.15832332370,
            27.65276268810,
            28.59435645580,
            30.43926351810,
            31.95339836860,
            34.54002524860,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'Raw numbers used in the paper provided in personal '
                'communication with paper authors'
            ),
            'reference': 'https://arxiv.org/abs/2309.17182v1',
            'type': 'neural-field',
            'data': 'multi',
        }),
    }),
    'JPEG': immutabledict.immutabledict({
        'bpp': (
            0.22115325927734372,
            0.32661437988281244,
            0.42312622070312506,
            0.5083855523003472,
            0.5878660413953993,
            0.6601265801323783,
            0.728946261935764,
            0.786026848687066,
            0.8497187296549479,
            0.9060007731119791,
            0.9643800523546006,
            1.0372119479709203,
            1.1273964775933163,
            1.23984612358941,
            1.3688269721137154,
            1.5718070136176217,
            1.8588163587782118,
            2.350199381510416,
            3.4013400607638893,
        ),
        'psnr': (
            23.779894921045457,
            26.57723034342358,
            28.042246379237767,
            29.04180914810682,
            29.78473021842612,
            30.378313496658652,
            30.903164761925012,
            31.307827225129476,
            31.70484530103775,
            32.05865273799422,
            32.39929573599792,
            32.789915599755076,
            33.234742421489976,
            33.792594320368266,
            34.39429509119509,
            35.239001425077355,
            36.32886455167303,
            37.9121247026983,
            40.556657112988766,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/jpeg.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'JPEG. Pillow version 7.1.2',
            'type': 'classical',
        }),
    }),
    'JPEG2000': immutabledict.immutabledict({
        'bpp': (
            2.3973227606879344,
            1.198240492078993,
            0.7982796563042536,
            0.5988286336263021,
            0.47928958468967026,
            0.3986248440212674,
            0.3418426513671875,
            0.2985627916124132,
            0.2657267252604167,
            0.239408704969618,
            0.21756574842664925,
            0.19917382134331593,
            0.1844355265299479,
            0.17067379421657983,
            0.1594840155707465,
            0.14888424343532988,
            0.1407623291015625,
            0.1324556138780382,
            0.12579854329427081,
        ),
        'psnr': (
            39.792000990389106,
            35.680719121855866,
            33.4899612266554,
            32.08974009483378,
            31.07457563807384,
            30.30076938092785,
            29.696116673252778,
            29.191735537599026,
            28.774226514669053,
            28.416590303662094,
            28.099224033143784,
            27.796635846897356,
            27.554469623734875,
            27.316993364478737,
            27.11825064017241,
            26.922846006615767,
            26.757424438699193,
            26.586234996125643,
            26.440382666052425,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/jpeg2000.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'JPEG2000. ffmpeg version 3.4.6-0ubuntu0.18.04.1',
            'type': 'classical',
        }),
    }),
    'BPG': immutabledict.immutabledict({
        'bpp': (
            0.06776767306857638,
            0.1610234578450521,
            0.35155402289496523,
            0.6846966213650174,
            1.1996654934353295,
            1.9297120836046002,
            3.0578757392035594,
            4.703142801920573,
        ),
        'psnr': (
            26.19512806390954,
            28.68154930675666,
            31.5946565907943,
            34.857473813062924,
            38.259886899666974,
            41.47887283561703,
            44.70311031066196,
            47.63251604714922,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/bpg_444_x265_ycbcr.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'BPG. BPG version b0.9.8',
            'type': 'classical',
        }),
    }),
    'BPG420': immutabledict.immutabledict({
        'bpp': (
            0.023778,
            0.028261,
            0.034131,
            0.041103,
            0.049042,
            0.058963,
            0.070547,
            0.083888,
            0.100428,
            0.117291,
            0.138370,
            0.161282,
            0.189494,
            0.219051,
            0.256016,
            0.294495,
            0.338370,
            0.385684,
            0.440687,
            0.500803,
            0.569223,
            0.642459,
            0.711308,
            0.795128,
            0.884535,
            0.977693,
            1.083310,
            1.193119,
            1.302428,
            1.425981,
            1.558065,
            1.699156,
            1.862072,
            2.036448,
            2.211966,
            2.412092,
            2.613790,
            2.838744,
            3.068833,
            3.303337,
            3.539684,
            3.787450,
            4.045572,
            4.317412,
            4.674708,
            4.988948,
            5.355372,
            5.604246,
            5.786789,
            5.974362,
            6.206643,
            6.442166,
        ),
        'psnr': (
            23.791201,
            24.170052,
            24.583564,
            25.022156,
            25.445649,
            25.910184,
            26.350063,
            26.822649,
            27.306609,
            27.732251,
            28.237878,
            28.676496,
            29.215355,
            29.671082,
            30.254687,
            30.730227,
            31.313155,
            31.792825,
            32.397932,
            33.028258,
            33.673742,
            34.307052,
            34.755805,
            35.401802,
            36.017190,
            36.620787,
            37.241529,
            37.858404,
            38.418848,
            38.988973,
            39.540628,
            40.065401,
            40.618115,
            41.155617,
            41.638891,
            42.137440,
            42.670036,
            43.122145,
            43.538615,
            43.918761,
            44.257987,
            44.565300,
            44.851237,
            45.123028,
            45.475263,
            45.777371,
            46.138732,
            46.341547,
            46.442053,
            46.512323,
            46.555544,
            46.571582,
        ),
        'meta': immutabledict.immutabledict({
            'source': 'Run by us using internal libraries',
            'reference': 'BPG (4:2:0)',
            'type': 'classical',
        }),
    }),
    'BPG444': immutabledict.immutabledict({
        'bpp': (
            0.023857,
            0.028540,
            0.034282,
            0.041183,
            0.049301,
            0.058861,
            0.070444,
            0.084175,
            0.100255,
            0.119211,
            0.140076,
            0.165529,
            0.193939,
            0.226882,
            0.264902,
            0.308004,
            0.353821,
            0.406663,
            0.465174,
            0.528513,
            0.602615,
            0.681227,
            0.763150,
            0.855122,
            0.954195,
            1.058729,
            1.178031,
            1.302895,
            1.427853,
            1.570484,
            1.724310,
            1.890798,
            2.085680,
            2.296926,
            2.513738,
            2.762745,
            3.021654,
            3.311613,
            3.616902,
            3.943210,
            4.282488,
            4.657569,
            5.069882,
            5.543335,
            6.176737,
            6.804908,
            7.545595,
            8.045927,
            8.453601,
            8.895050,
            9.349519,
            9.810592,
        ),
        'psnr': (
            23.787652,
            24.169736,
            24.583862,
            25.005823,
            25.441511,
            25.892665,
            26.341951,
            26.815071,
            27.298468,
            27.803390,
            28.299163,
            28.836366,
            29.370955,
            29.927924,
            30.519758,
            31.114314,
            31.707365,
            32.333445,
            32.953625,
            33.589638,
            34.260077,
            34.921675,
            35.560822,
            36.239184,
            36.892823,
            37.539114,
            38.225035,
            38.892220,
            39.508856,
            40.150529,
            40.780304,
            41.391531,
            42.054650,
            42.720115,
            43.337300,
            43.993691,
            44.653016,
            45.301270,
            45.915667,
            46.518668,
            47.080322,
            47.624560,
            48.164119,
            48.722029,
            49.453080,
            50.169809,
            51.141528,
            51.834165,
            52.318361,
            52.695935,
            52.937088,
            53.030342,
        ),
        'meta': immutabledict.immutabledict({
            'source': 'Run by us using internal libraries',
            'reference': 'BPG (4:4:4)',
            'type': 'classical',
        }),
    }),
    'VTM': immutabledict.immutabledict({
        'bpp': (
            0.04824490017361111,
            0.11249542236328125,
            0.24581654866536465,
            0.4905454847547744,
            0.8748101128472224,
            1.43084971110026,
            2.3246070014105906,
            3.6326090494791665,
        ),
        'psnr': (
            26.14484539430146,
            28.493021754424603,
            31.199873720014093,
            34.26153257289846,
            37.419793158067485,
            40.42444421698711,
            43.505766973634486,
            46.59176008512777,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/vtm.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM',
            'type': 'classical',
        }),
    }),
    'SPY': immutabledict.immutabledict({
        'bpp': (
            0.08315870496961805,
            0.1546198527018229,
            0.24893612331814235,
            0.7844628228081597,
        ),
        'psnr': (
            25.56004644305473,
            27.019205940785646,
            28.28015802285809,
            32.13873826006784,
        ),
        'meta': immutabledict.immutabledict({
            'source': 'Obtained from paper authors',
            'reference': 'https://arxiv.org/abs/2112.04267',
            'type': 'neural-field',
            'data': 'multi',
        }),
    }),
    'MSCN': immutabledict.immutabledict({
        'bpp': (0.068714, 0.13155, 0.26615, 0.36841, 0.50215),
        'psnr': (25.414, 26.813, 28.905, 29.666, 30.055),
        'meta': immutabledict.immutabledict({
            'source': 'Obtained from paper authors',
            'reference': 'https://arxiv.org/abs/2205.08957',
            'type': 'neural-field',
            'data': 'multi',
        }),
    }),
    'VC-INR': immutabledict.immutabledict({
        'bpp': (0.08, 0.14, 0.48, 1.09, 1.54, 2.17, 3.09, 3.74, 5.56),
        'psnr': (26.86, 28.33, 32.07, 34.78, 36.59, 38.57, 41.26, 42.12, 42.24),
        'meta': immutabledict.immutabledict({
            'source': 'Obtained from paper authors',
            'reference': 'https://arxiv.org/abs/2301.09479',
            'type': 'neural-field',
            'data': 'multi',
        }),
    }),
    'BMS': immutabledict.immutabledict({
        'bpp': (
            0.13129340277777776,
            0.20889282226562503,
            0.3198581271701389,
            0.47835625542534727,
            0.6686876085069443,
            0.9388258192274304,
            1.2591722276475694,
            1.6602240668402775,
        ),
        'psnr': (
            27.581536752297392,
            29.196703405493214,
            30.972162072759534,
            32.83818257445048,
            34.52626403063645,
            36.74334835426406,
            38.58384824012354,
            40.556865931529494,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/compressai-bmshj2018-hyperprior_mse_cuda.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'https://arxiv.org/abs/1802.01436',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 43_542,
                'max': 43_542,
                'source': (
                    'Calculated using the CompressAI version of this model,'
                    ' with the fvcore library.'
                ),
            }),
        }),
    }),
    'MBT': immutabledict.immutabledict({
        'bpp': (
            0.1105312771267361,
            0.18697102864583334,
            0.28769599066840285,
            0.4323391384548611,
            0.6387668185763888,
            0.8853047688802084,
            1.2003648546006944,
            1.5873446994357636,
        ),
        'psnr': (
            28.08847023321361,
            29.64549843593544,
            31.362275162681673,
            33.0859466233318,
            35.093758380265,
            36.98882029536827,
            38.93350092275383,
            40.63906745832768,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/compressai-mbt2018_mse_cuda.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'https://arxiv.org/abs/1809.02736',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 107_930,
                'max': 139_638,
                'source': (
                    'Calculated using the CompressAI version of this model,'
                    ' with the fvcore library.'
                ),
            }),
        }),
    }),
    'CST': immutabledict.immutabledict({
        'bpp': (
            0.11959499782986115,
            0.18379720052083337,
            0.2709520128038195,
            0.4173753526475694,
            0.5944688585069444,
            0.805735270182292,
        ),
        'psnr': (
            28.58235164600031,
            29.969272906591517,
            31.342783743928436,
            33.390308674052186,
            35.11886905407247,
            36.7040956894179,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/compressai-cheng2020-anchor_mse_cuda.json'
                ' (accessed 11/08/23)'
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
    'ELIC': immutabledict.immutabledict({
        'bpp': (
            0.1235,
            0.1965,
            0.3337,
            0.4908,
            0.7041,
            0.8573,
        ),
        'psnr': (
            29.113,
            30.698,
            32.784,
            34.577,
            36.488,
            37.624,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/InterDigitalInc/CompressAI/blob/master/results/image/kodak/paper-elic2022_mse.json'
                ' (accessed 11/08/23)'
            ),
            'reference': 'https://arxiv.org/abs/2203.10886',
            'type': 'autoencoder',
            'data': 'multi',
        }),
    }),
    'MLIC': immutabledict.immutabledict({
        'bpp': (0.1162, 0.1896, 0.2908, 0.4375, 0.6136, 0.8415),
        'psnr': (29.2291, 30.7639, 32.3709, 34.1024, 35.7542, 37.4026),
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
        'bpp': (
            0.1124,
            0.1823,
            0.2815,
            0.427,
            0.5912,
            0.8103,
        ),
        'psnr': (
            29.1983,
            30.7816,
            32.4008,
            34.1389,
            35.7631,
            37.3736,
        ),
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
        'bpp': (
            0.124,
            0.191,
            0.298,
            0.441,
            0.651,
            0.903,
        ),
        'psnr': (
            29.14,
            30.50,
            32.15,
            33.97,
            35.82,
            37.72,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'https://github.com/Googolxx/STF/blob/main/results/stf_mse_Kodak.json'
                ' (accessed 16/10/23)'
            ),
            'reference': 'https://arxiv.org/abs/2203.08450',
            'type': 'autoencoder',
            'data': 'multi',
        }),
    }),
    'WYH': immutabledict.immutabledict({
        'bpp': (
            0.1868,
            0.2875,
            0.3950,
            0.7532,
            1.0548,
            1.1941,
        ),
        'psnr': (
            30.2688,
            31.8869,
            33.2138,
            36.5635,
            38.4772,
            39.2847,
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
    'EVC-S': immutabledict.immutabledict({
        'bpp': (
            0.3411951944851169,
            0.39075820548032414,
            0.43293287375974243,
            0.4702865665669307,
            0.5070591231117618,
            0.5401484216557382,
            0.5819165673854325,
            0.6399156033722273,
            0.6927335231044863,
            0.7406213337941195,
            0.7947925097992852,
            0.8616040815955328,
            0.9194094054948754,
            0.993123675618651,
        ),
        'psnr': (
            32.422170811437525,
            33.0034943826859,
            33.503416550813476,
            33.92287666600465,
            34.306292572692286,
            34.608280237682436,
            34.977981631678105,
            35.45866054031423,
            35.86853928362471,
            36.223855200511004,
            36.55823702757395,
            36.935050644394224,
            37.22571243001841,
            37.59973483486224,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'Obtained from Figure 17b in paper by using the PlotDigitizer'
                ' tool.'
            ),
            'reference': 'https://arxiv.org/abs/2302.05071',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 35_338,
                'max': 35_338,
                'source': (
                    'Obtained from Table 3 in paper. Numbers were obtained'
                    ' using the ptflops library.'
                ),
            }),
        }),
    }),
    'EVC-M': immutabledict.immutabledict({
        'bpp': (
            0.3338450485189665,
            0.377539401766145,
            0.420761752321362,
            0.45744154533135684,
            0.4941049683057923,
            0.5355866384131979,
            0.5800885800813045,
            0.6282628663932265,
            0.668074792873578,
            0.7165028147366699,
            0.7755440763207436,
            0.840835963149231,
            0.9037050847149342,
            0.9807943105031969,
        ),
        'psnr': (
            32.49552064154934,
            33.02212035491334,
            33.548424843933674,
            33.976902720894905,
            34.360640690502905,
            34.76066967616573,
            35.165449089903866,
            35.57675027777927,
            35.89285503411183,
            36.2633347468317,
            36.671039565429766,
            37.04527667888716,
            37.3852677684797,
            37.80633819827267,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'Obtained from Figure 17b in paper by using the PlotDigitizer'
                ' tool.'
            ),
            'reference': 'https://arxiv.org/abs/2302.05071',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 123_774,
                'max': 123_774,
                'source': (
                    'Obtained from Table 3 in paper. Numbers were obtained'
                    ' using the ptflops library.'
                ),
            }),
        }),
    }),
    'EVC-L': immutabledict.immutabledict({
        'bpp': (
            0.3275807815782533,
            0.3714442918595451,
            0.4145193120947279,
            0.4498103804214375,
            0.4864356066462344,
            0.5271778968142092,
            0.5728148276144311,
            0.6203479542002783,
            0.6608692488882018,
            0.7093300108224124,
            0.7679756632138017,
            0.8330983930081759,
            0.8945815182298535,
            0.9628527515301437,
        ),
        'psnr': (
            32.47638473636466,
            33.000059044868735,
            33.527195529766665,
            33.93715478880724,
            34.3401360179067,
            34.737749531666836,
            35.153318053237,
            35.56867186619359,
            35.891244719510034,
            36.266984793262445,
            36.67187156130736,
            37.05034916988282,
            37.39042077520545,
            37.76414795570561,
        ),
        'meta': immutabledict.immutabledict({
            'source': (
                'Obtained from Figure 17b in paper by using the PlotDigitizer'
                ' tool.'
            ),
            'reference': 'https://arxiv.org/abs/2302.05071',
            'type': 'autoencoder',
            'data': 'multi',
            'macs_per_pixel': immutabledict.immutabledict({
                'min': 257_941,
                'max': 257_941,
                'source': (
                    'Obtained from Table 3 in paper. Numbers were obtained'
                    ' using the ptflops library.'
                ),
            }),
        }),
    }),
})
