mnist_k_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
mnist_silhoette_scores = [
    0.08700278,
    0.05368753,
    0.06047742,
    0.06651142,
    0.0648853,
    0.06617383,
    0.073137105,
    0.056582894,
    0.058420964,
    0.057745416,
    0.05804279,
    0.059710518,
    0.060894962,
    0.061626505,
    0.064400226,
    0.06443312,
    0.06611478,
    0.06665654,
    0.06825592,
    0.05843336,
    0.059491906,
    0.060270775,
    0.06023429,
]

census_k_list = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
]
census_silhoette_scores = [
    0.16805345324006696,
    0.14804284201210302,
    0.16153600903228232,
    0.12582749097241536,
    0.1404755181562333,
    0.14331866454460604,
    0.14298071700706588,
    0.15827324015222413,
    0.1570605475778061,
    0.171355727311151,
    0.17006392660169467,
    0.17459225674714293,
    0.18099942490089463,
    0.17301218727107448,
    0.17779465341789322,
    0.16744055245214126,
    0.18002107799038713,
    0.1829350649232376,
    0.18931188127656134,
    0.1909065915693278,
    0.19664458626104786,
    0.1926283293980731,
    0.17017474517265985,
    0.1805771863669699,
    0.18967401496751188,
    0.16715917824859453,
    0.18523241182237066,
    0.19087356766334287,
]

census_bic_scores = {
    "spherical": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "bic_score": [
            21538278.120315816,
            20307695.89070165,
            19385636.359287634,
            19009428.69573563,
            17900481.095538992,
            17639921.429058954,
            16937527.176488444,
            16662092.977392962,
            16513882.962307882,
            16268050.53189757,
            16244086.537071513,
            16449840.378258932,
            15585161.549497483,
            15868571.279470019,
            15431655.518897312,
            15389200.75968362,
            15275011.762757933,
            15313578.532434914,
            15141610.075983247,
            14956466.07771694,
            14908317.527969072,
            14879343.509869684,
            15017926.22624891,
            14530282.902877139,
            13205815.676551769,
            14376632.730029924,
            14359128.482040701,
            14324729.490912287,
        ],
    },
    "tied": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "bic_score": [
            14912781.144728402,
            13249548.949512515,
            13927933.256212326,
            13645369.10839662,
            12824295.88514214,
            12441717.002603326,
            12874128.041720066,
            12139945.5136565,
            12652206.254223997,
            12693517.96547867,
            12267672.180122029,
            9872465.317058358,
            9591848.840556135,
            9583123.774253158,
            11477931.45992479,
            9796820.569593992,
            12270876.232415672,
            9144695.001596728,
            8711240.371955914,
            9112988.336651066,
            11499214.42922043,
            8485281.798147954,
            10662944.79921041,
            8456363.92577275,
            8483924.546171779,
            8693109.05717742,
            8326823.102289145,
            8188781.541915285,
        ],
    },
    "diag": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "bic_score": [
            6287364.85311645,
            -4716396.046487747,
            -6215072.085250653,
            -9855996.44955867,
            -14236624.313547578,
            -18031462.557793524,
            -21102883.73581618,
            -25132686.398493424,
            -24620902.50059988,
            -25474004.103181545,
            -24428985.20592104,
            -29884021.425879974,
            -27458543.986095775,
            -29803169.681355648,
            -30912205.813235056,
            -31439399.100309335,
            -32201470.00903037,
            -31753974.89125455,
            -33084493.24220692,
            -32901204.393431395,
            -33782741.2913745,
            -34299104.2248893,
            -37004455.88710795,
            -38416774.83606888,
            -38131092.30498729,
            -36172638.417212985,
            -36583675.64999416,
            -37540996.287510365,
        ],
    },
    "full": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "bic_score": [
            5033424.640636925,
            -2978701.3201025855,
            -10637638.356308578,
            -12517183.836359093,
            -14556029.025846578,
            -18479333.794341076,
            -22797697.418693718,
            -29376104.09369224,
            -27002710.95487399,
            -28718815.330475464,
            -28483742.683217563,
            -31563689.22793595,
            -27467394.31980107,
            -32805248.82774587,
            -34515987.83553406,
            -31977097.471155316,
            -34980634.36451822,
            -33054661.315785848,
            -35344368.291531876,
            -37271153.16058794,
            -36137601.438195504,
            -36433048.338004135,
            -38255317.825321645,
            -37575485.823921636,
            -37868229.79690218,
            -37470274.188010104,
            -38823499.954852484,
            -37301861.09920445,
        ],
    },
}


mnist_bic_scores = {
    "spherical": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "bic_score": [
            2553674.28468122,
            817990.232310261,
            -1953683.0793431902,
            -3463731.2060984042,
            -4809663.0917533925,
            -5852817.095577123,
            -7523862.88304406,
            -8444519.111222602,
            -9289448.294687292,
            -8770158.799812362,
            -10571916.92059275,
            -10795546.462770067,
            -11659435.02311634,
            -12408274.44414811,
            -12665684.90860642,
            -13035638.377519174,
            -13462452.072589524,
            -13857813.095330205,
            -14652049.979338687,
            -15141856.50301904,
            -15433021.530157005,
            -15663127.816424184,
            -16008080.428518707,
        ],
    },
    "tied": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "bic_score": [
            -182839752.0501885,
            -182916775.51655427,
            -182952891.74602297,
            -183008534.61350954,
            -183168383.04958028,
            -183206052.3453542,
            -183257645.5642003,
            -183412299.94395974,
            -183415545.59211457,
            -183522506.79217675,
            -183551247.70050004,
            -183648806.33612236,
            -183759362.20052862,
            -183788198.2196085,
            -183830320.13845646,
            -183925217.29110143,
            -184038943.849445,
            -184076948.2714147,
            -184148969.60570988,
            -184161991.36146125,
            -184257037.25060046,
            -184357450.59491283,
            -184584809.33067408,
        ],
    },
    "diag": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "bic_score": [
            -170110042.14075634,
            -200731710.31600752,
            -221977478.81292906,
            -243700571.78172365,
            -251660647.25392628,
            -253527707.19691744,
            -260546133.42946965,
            -265115066.92865026,
            -267939629.683693,
            -273111498.1100204,
            -274535324.27789915,
            -280998902.6568829,
            -282384014.7565043,
            -285496048.10555184,
            -285476547.59597677,
            -289283909.3010946,
            -292359217.1459337,
            -293071158.31403226,
            -297051738.1490723,
            -298510003.1974969,
            -299553408.6207938,
            -300381543.5697304,
            -302823467.5185548,
        ],
    },
    "full": {
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "bic_score": [
            -221220206.7053896,
            -242024995.34370816,
            -261661266.0085028,
            -268035790.13541687,
            -271192168.39373523,
            -272158713.59449166,
            -272349448.8268123,
            -274844067.90255064,
            -274652454.2594179,
            -273238619.13353235,
            -274614514.6884744,
            -273024357.25318885,
            -271817913.08974886,
            -270125565.13719493,
            -269676733.0195531,
            -268087024.15933892,
            -265687075.94768572,
            -264309602.37118956,
            -262416803.71099585,
            -259508865.58418605,
            -256615268.2812066,
            -253431710.26143792,
            -250836492.46889246,
        ],
    },
}
mnist_bic_time = 59698.92

MNIST_Reduced_PCA_85_0_Explained_Variance_silhoette_scores = {
    "K_list": [2, 3, 4, 5, 6, 7, 8, 9],
    "scores": [
        0.09911737908239855,
        0.06566759570722783,
        0.07485218102739928,
        0.08387190584522514,
        0.08360488497832427,
        0.08544383301700807,
        0.09355636455959626,
        0.07812731295804991,
    ],
}

MNIST_Reduced_PCA_85_0_Explained_Variance_silhoette_scores = {
    "K_list": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "scores": [
        0.09911737908239855,
        0.06566759570722783,
        0.07485218102739928,
        0.08387190584522514,
        0.08360488497832427,
        0.08544383301700807,
        0.09355636455959626,
        0.07812731295804991,
        0.08042267809630366,
        0.07978485630392529,
        0.08125016635072169,
        0.08301583728200529,
        0.08429841313990873,
    ],
}

MNIST_Reduced_PCA_85_0_EV_silhoette_scores = {"bic_scores": {"spherical": {"k": [2], "bic_score": [41346206.33036383]}, "tied": {"k": [2], "bic_score": [40319400.70123316]}, "diag": {"k": [2], "bic_score": [33220561.652728666]}, "full": {"k": [2], "bic_score": [16978028.92642781]}}}

MNIST_Reduced_PCA_85_0_EV_silhoette_scores = {"bic_scores": {"spherical": {"k": [2, 3, 4], "bic_score": [41346206.28478672, 40124579.914083436, 39349032.598058306]}, "tied": {"k": [2, 3, 4], "bic_score": [40349719.14586762, 40245708.13399168, 40100814.29743443]}, "diag": {"k": [2, 3, 4], "bic_score": [33220561.191994917, 32050252.402799323, 31506077.681052495]}, "full": {"k": [2, 3, 4], "bic_score": [19912574.213650197, 3991676.6224731384, -19535.00450892828]}}}

MNIST_Reduced_PCA_85_0_Explained_Variance_silhoette_scores = {"K_list": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "scores": [0.1388249185123579, 0.052287247586715824, 0.04713270649004603, 0.015417468938327183, 0.011517979504096656, 0.009946392874577579, 0.01655940305368793, 0.028839665301981163, 0.027687046754248816, 0.026596721289909333, 0.026868144633256135, 0.027039372864302587, 0.029931257360568904]}

MNIST_Reduced_PCA_85_0_EV_silhoette_scores = {"bic_scores": {"spherical": {"k": [2, 3, 4], "bic_score": [8795414.369386306, 8591970.293489708, 8464905.066782594]}, "tied": {"k": [2, 3, 4], "bic_score": [7195510.054151426, 7191751.144365712, 7148051.571187843]}, "diag": {"k": [2, 3, 4], "bic_score": [7042861.524382301, 6979728.582503037, 6940592.727750813]}, "full": {"k": [2, 3, 4], "bic_score": [6214124.868251806, 5404560.7055528015, 4999679.714240114]}}}

MNIST_silhoette_scores = {"K_list": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "scores": [0.1275710418442769, 0.04132621386552958, 0.03634962816045757, 0.003366575383101436, -0.0038583189430360016, -0.0061858013418560065, 0.0007308949917344371, 0.012207658805940616, 0.008827114686201353, 0.006238956961084217, 0.006516865544169581, 0.0062301861364063675, 0.009029305057984974]}

Census_Reduced_PCA_85.0%_Explained_Variance_silhoette_scores = {"K_list": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "scores": [0.1987750654381868, 0.18661861112369527, 0.1483431114313846, 0.18251678128610263, 0.18210999508024442, 0.1923890146862687, 0.19768770113330125, 0.19798094128015647, 0.21737155410449865, 0.2139588195386288, 0.2310622765167408, 0.2304803371529165, 0.21806074775167247]}
