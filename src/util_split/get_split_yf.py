import os
from collections import defaultdict

# FCVID_TO_UNIFIED 字典请保留原有定义（此处省略）
FCVID_TO_UNIFIED = {
    # (1) accordionPerformance -> 26: 演奏管乐
    1: 26,
    # (2) americanFootballAmateur -> 999: 不在统一大类
    2: 999,
    # (3) americanFootballProfessional -> 999: 不在统一大类
    3: 999,
    # (4) amusementPark -> 7: 户外休闲娱乐
    4: 7,
    # (5) archery -> 2: 射箭
    5: 2,
    # (6) armWrestling -> 11: 格斗与武术
    6: 11,
    # (7) assemblingABike -> 3: 自行车运动
    7: 3,
    # (8) assemblingAComputer -> 999: 不在统一大类
    8: 999,
    # (9) babyCrawling -> 999: 不在统一大类
    9: 999,
    # (10) babyEatingSnack -> 999: 不在统一大类
    10: 999,
    # (11) badminton -> 30: 球拍运动
    11: 30,
    # (12) barbellWorkout -> 14: 力量训练与健身
    12: 14,
    # (13) baseball -> 999: 不在统一大类
    13: 999,
    # (14) basketballAmateur -> 41: 篮球
    14: 41,
    # (15) basketballProfessional -> 41: 篮球
    15: 41,
    # (16) beach -> 999: 不在统一大类
    16: 999,
    # (17) beatbox -> 27: 演奏打击乐
    17: 27,
    # (18) bee -> 999: 不在统一大类
    18: 999,
    # (19) bikeTricks -> 3: 自行车运动
    19: 3,
    # (20) biking -> 3: 自行车运动
    20: 3,
    # (21) billiard -> 42: 台球与桌游
    21: 42,
    # (22) bird -> 999: 不在统一大类
    22: 999,
    # (23) birthday -> 18: 节日与庆典
    23: 18,
    # (24) boating -> 10: 水上划行运动
    24: 10,
    # (25) bowling -> 31: 保龄球
    25: 31,
    # (26) boxing -> 12: 拳击
    26: 12,
    # (27) bridge -> 999: 不在统一大类
    27: 999,
    # (28) brushingTeeth -> 6: 口腔与面部清洁
    28: 6,
    # (29) bumperCars -> 7: 户外休闲娱乐
    29: 7,
    # (30) bungeeJumping -> 8: 极限空中运动
    30: 8,
    # (31) butterfly -> 36: 游泳
    31: 36,
    # (32) camel -> 9: 骑乘动物与竞技
    32: 9,
    # (33) carAccidents -> 999: 不在统一大类
    33: 999,
    # (34) carExhibition -> 999: 不在统一大类
    34: 999,
    # (35) carRacing -> 999: 不在统一大类
    35: 999,
    # (36) carWashing -> 15: 清洁与家务
    36: 15,
    # (37) cardManipulation -> 42: 台球与桌游
    37: 42,
    # (38) cat -> 16: 宠物护理与遛狗
    38: 16,
    # (39) cathedralExterior -> 999: 不在统一大类
    39: 999,
    # (40) celloPerformance -> 25: 演奏弦乐
    40: 25,
    # (41) chamberMusic -> 25: 演奏弦乐
    41: 25,
    # (42) changingTires -> 999: 不在统一大类
    42: 999,
    # (43) chorus -> 13: 队列与团队表演
    43: 13,
    # (44) classroom -> 999: 不在统一大类
    44: 999,
    # (45) cleaningAppliance -> 15: 清洁与家务
    45: 15,
    # (46) cleaningCarpet -> 15: 清洁与家务
    46: 15,
    # (47) cleaningFloor -> 15: 清洁与家务
    47: 15,
    # (48) cleaningWindows -> 15: 清洁与家务
    48: 15,
    # (49) cow -> 999: 不在统一大类
    49: 999,
    # (50) debate -> 999: 不在统一大类
    50: 999,
    # (51) decoratingChristmasTree -> 18: 节日与庆典
    51: 18,
    # (52) deliciousFood -> 21: 烹饪与食物制作
    52: 21,
    # (53) desert -> 999: 不在统一大类
    53: 999,
    # (54) diningAtRestaurant -> 21: 烹饪与食物制作
    54: 21,
    # (55) dinnerAtHome -> 21: 烹饪与食物制作
    55: 21,
    # (56) diving -> 24: 跳水
    56: 24,
    # (57) dog -> 16: 宠物护理与遛狗
    57: 16,
    # (58) doingGraffiti -> 43: 绘画与书写
    58: 43,
    # (59) dolphin -> 999: 不在统一大类
    59: 999,
    # (60) dumbbellWorkout -> 14: 力量训练与健身
    60: 14,
    # (61) egyptianPyramids -> 999: 不在统一大类
    61: 999,
    # (62) eiffelTower -> 999: 不在统一大类
    62: 999,
    # (63) elephant -> 999: 不在统一大类
    63: 999,
    # (64) eyeMakeup -> 1: 化妆与美容护理
    64: 1,
    # (65) faceMassage -> 1: 化妆与美容护理
    65: 1,
    # (66) fashionShow -> 999: 不在统一大类
    66: 999,
    # (67) fencing -> 20: 击剑
    67: 20,
    # (68) fireFighting -> 999: 不在统一大类
    68: 999,
    # (69) fireworksShow -> 18: 节日与庆典
    69: 18,
    # (70) fishing -> 999: 不在统一大类
    70: 999,
    # (71) flutePerformance -> 26: 演奏管乐
    71: 26,
    # (72) flyingKites -> 7: 户外休闲娱乐
    72: 7,
    # (73) forest -> 999: 不在统一大类
    73: 999,
    # (74) fruitTreePruning -> 999: 不在统一大类
    74: 999,
    # (75) giraffe -> 999: 不在统一大类
    75: 999,
    # (76) golfing -> 999: 不在统一大类
    76: 999,
    # (77) gorilla -> 999: 不在统一大类
    77: 999,
    # (78) graduation -> 999: 不在统一大类
    78: 999,
    # (79) groupBanquet -> 21: 烹饪与食物制作
    79: 21,
    # (80) groupDance -> 4: 舞蹈
    80: 4,
    # (81) guitarPerformance -> 25: 演奏弦乐
    81: 25,
    # (82) hairCutting -> 5: 头发护理
    82: 5,
    # (83) hairstyleDesign -> 5: 头发护理
    83: 5,
    # (84) hamster -> 999: 不在统一大类
    84: 999,
    # (85) harmonicaPerformance -> 26: 演奏管乐
    85: 26,
    # (86) hiking -> 999: 不在统一大类
    86: 999,
    # (87) horseRiding -> 9: 骑乘动物与竞技
    87: 9,
    # (88) housePlants -> 999: 不在统一大类
    88: 999,
    # (89) hulaHoop -> 38: 体操运动
    89: 38,
    # (90) iceSkating -> 17: 冰雪运动
    90: 17,
    # (91) insideAirplane -> 999: 不在统一大类
    91: 999,
    # (92) insideBus -> 999: 不在统一大类
    92: 999,
    # (93) insideTheOrientalPearlTVTower -> 999: 不在统一大类
    93: 999,
    # (94) kickingShuttlecock -> 30: 球拍运动
    94: 30,
    # (95) kidsMakingFaces -> 999: 不在统一大类
    95: 999,
    # (96) kidsPlayingWithBlocks -> 999: 不在统一大类
    96: 999,
    # (97) kindergarten -> 999: 不在统一大类
    97: 999,
    # (98) kiteSurfing -> 34: 水上冲浪运动
    98: 34,
    # (99) knitting -> 39: 编织与手工
    99: 39,
    # (100) laptop -> 999: 不在统一大类
    100: 999,
    # (101) lightning -> 999: 不在统一大类
    101: 999,
    # (102) makingBookmark -> 39: 编织与手工
    102: 39,
    # (103) makingBracelets -> 39: 编织与手工
    103: 39,
    # (104) makingCake -> 21: 烹饪与食物制作
    104: 21,
    # (105) makingCeramicCraft -> 39: 编织与手工
    105: 39,
    # (106) makingChineseDumplings -> 21: 烹饪与食物制作
    106: 21,
    # (107) makingCoffee -> 22: 饮品制作
    107: 22,
    # (108) makingCookies -> 21: 烹饪与食物制作
    108: 21,
    # (109) makingEarrings -> 39: 编织与手工
    109: 39,
    # (110) makingEggTarts -> 21: 烹饪与食物制作
    110: 21,
    # (111) makingFestivalCards -> 39: 编织与手工
    111: 39,
    # (112) makingFrenchFries -> 21: 烹饪与食物制作
    112: 21,
    # (113) makingHotdog -> 21: 烹饪与食物制作
    113: 21,
    # (114) makingIcecream -> 21: 烹饪与食物制作
    114: 21,
    # (115) makingJuice -> 22: 饮品制作
    115: 22,
    # (116) makingMilkTea -> 22: 饮品制作
    116: 22,
    # (117) makingMixedDrinks -> 22: 饮品制作
    117: 22,
    # (118) makingPaperFlowers -> 39: 编织与手工
    118: 39,
    # (119) makingPaperPlane -> 39: 编织与手工
    119: 39,
    # (120) makingPencilCases -> 39: 编织与手工
    120: 39,
    # (121) makingPhoneCases -> 39: 编织与手工
    121: 39,
    # (122) makingPhotoFrame -> 39: 编织与手工
    122: 39,
    # (123) makingPizza -> 21: 烹饪与食物制作
    123: 21,
    # (124) makingRings -> 39: 编织与手工
    124: 39,
    # (125) makingSalad -> 21: 烹饪与食物制作
    125: 21,
    # (126) makingSandwich -> 21: 烹饪与食物制作
    126: 21,
    # (127) makingShorts -> 39: 编织与手工
    127: 39,
    # (128) makingSnowman -> 17: 冰雪运动
    128: 17,
    # (129) makingSushi -> 21: 烹饪与食物制作
    129: 21,
    # (130) makingTea -> 22: 饮品制作
    130: 22,
    # (131) makingWallet -> 39: 编织与手工
    131: 39,
    # (132) marathon -> 999: 不在统一大类
    132: 999,
    # (133) marchingBand -> 13: 队列与团队表演
    133: 13,
    # (134) marriageProposal -> 18: 节日与庆典
    134: 18,
    # (135) mountain -> 999: 不在统一大类
    135: 999,
    # (136) mowing -> 15: 清洁与家务
    136: 15,
    # (137) nailArtDesign -> 1: 化妆与美容护理
    137: 1,
    # (138) outsideAirplane -> 999: 不在统一大类
    138: 999,
    # (139) outsideBus -> 999: 不在统一大类
    139: 999,
    # (140) outsideTheOrientalPearlTVTower -> 999: 不在统一大类
    140: 999,
    # (141) painting -> 43: 绘画与书写
    141: 43,
    # (142) panda -> 999: 不在统一大类
    142: 999,
    # (143) paperCutting -> 39: 编织与手工
    143: 39,
    # (144) parade -> 13: 队列与团队表演
    144: 13,
    # (145) parkingCars -> 999: 不在统一大类
    145: 999,
    # (146) parkour -> 38: 体操运动
    146: 38,
    # (147) penSpinning -> 7: 户外休闲娱乐
    147: 7,
    # (148) pianoPerformance -> 28: 演奏键盘乐
    148: 28,
    # (149) picnic -> 7: 户外休闲娱乐
    149: 7,
    # (150) pitchingATent -> 7: 户外休闲娱乐
    150: 7,
    # (151) playground -> 7: 户外休闲娱乐
    151: 7,
    # (152) playingChess -> 42: 台球与桌游
    152: 42,
    # (153) playingFrisbeeWithDog -> 19: 飞盘运动
    153: 19,
    # (154) playingFrisbeeWithPeople -> 19: 飞盘运动
    154: 19,
    # (155) playingMahjong -> 42: 台球与桌游
    155: 42,
    # (156) playingWithNunChucks -> 11: 格斗与武术
    156: 11,
    # (157) playingWithRemoteControlledAircraft -> 999: 不在统一大类
    157: 999,
    # (158) playingWithRemoteControlledCars -> 999: 不在统一大类
    158: 999,
    # (159) publicSpeech -> 999: 不在统一大类
    159: 999,
    # (160) pullUps -> 14: 力量训练与健身
    160: 14,
    # (161) punchingBagWorkout -> 12: 拳击
    161: 12,
    # (162) pushUps -> 14: 力量训练与健身
    162: 14,
    # (163) rabbit -> 999: 不在统一大类
    163: 999,
    # (164) rafting -> 10: 水上划行运动
    164: 10,
    # (165) repairingMusicalInstruments -> 999: 不在统一大类
    165: 999,
    # (166) rhythmicGymnastics -> 38: 体操运动
    166: 38,
    # (167) river -> 10: 水上划行运动
    167: 10,
    # (168) roastingTurkey -> 21: 烹饪与食物制作
    168: 21,
    # (169) rockBandPerformance -> 27: 演奏打击乐
    169: 27,
    # (170) rockClimbing -> 32: 攀岩与爬绳
    170: 32,
    # (171) rollerSkating -> 40: 滑板运动
    171: 40,
    # (172) ropeSkipping -> 33: 跳绳
    172: 33,
    # (173) rowing -> 10: 水上划行运动
    173: 10,
    # (174) saxophonePerformance -> 26: 演奏管乐
    174: 26,
    # (175) sculpting -> 43: 绘画与书写
    175: 43,
    # (176) shavingBeard -> 35: 剃须与身体护理
    176: 35,
    # (177) shooting -> 999: 不在统一大类
    177: 999,
    # (178) shovelingSnow -> 17: 冰雪运动
    178: 17,
    # (179) showingFashionableHandbags -> 999: 不在统一大类
    179: 999,
    # (180) showingFashionableHighHeels -> 999: 不在统一大类
    180: 999,
    # (181) singingInKtv -> 13: 队列与团队表演
    181: 13,
    # (182) singingOnStage -> 13: 队列与团队表演
    182: 13,
    # (183) singleLensReflexCamera -> 999: 不在统一大类
    183: 999,
    # (184) sitUps -> 14: 力量训练与健身
    184: 14,
    # (185) skateboarding -> 40: 滑板运动
    185: 40,
    # (186) skiing -> 17: 冰雪运动
    186: 17,
    # (187) skydiving -> 8: 极限空中运动
    187: 8,
    # (188) smartphone -> 999: 不在统一大类
    188: 999,
    # (189) snake -> 999: 不在统一大类
    189: 999,
    # (190) snowballFight -> 17: 冰雪运动
    190: 17,
    # (191) soccerAmateur -> 29: 足球
    191: 29,
    # (192) soccerProfessional -> 29: 足球
    192: 29,
    # (193) socialDance -> 4: 舞蹈
    193: 4,
    # (194) solarEclipse -> 999: 不在统一大类
    194: 999,
    # (195) soloDance -> 4: 舞蹈
    195: 4,
    # (196) solvingMagicCube -> 42: 台球与桌游
    196: 42,
    # (197) sportsTrack -> 999: 不在统一大类
    197: 999,
    # (198) sprayPainting -> 43: 绘画与书写
    198: 43,
    # (199) streetFighting -> 11: 格斗与武术
    199: 11,
    # (200) sumoWrestling -> 11: 格斗与武术
    200: 11,
    # (201) sunset -> 999: 不在统一大类
    201: 999,
    # (202) surfing -> 34: 水上冲浪运动
    202: 34,
    # (203) swimmingAmateur -> 36: 游泳
    203: 36,
    # (204) swimmingProfessional -> 36: 游泳
    204: 36,
    # (205) symphonyOrchestraPerformance -> 25: 演奏弦乐
    205: 25,
    # (206) tableTennis -> 23: 乒乓球
    206: 23,
    # (207) tabletPC -> 999: 不在统一大类
    207: 999,
    # (208) taekwondo -> 11: 格斗与武术
    208: 11,
    # (209) taiChiChuan -> 37: 太极拳
    209: 37,
    # (210) tailgateParty -> 999: 不在统一大类
    210: 999,
    # (211) tajMahal -> 999: 不在统一大类
    211: 999,
    # (212) tattooing -> 35: 剃须与身体护理
    212: 35,
    # (213) temple -> 999: 不在统一大类
    213: 999,
    # (214) tennis -> 30: 球拍运动
    214: 30,
    # (215) theGreatWall -> 999: 不在统一大类
    215: 999,
    # (216) theStatueOfLiberty -> 999: 不在统一大类
    216: 999,
    # (217) tornado -> 999: 不在统一大类
    217: 999,
    # (218) townHallMeeting -> 999: 不在统一大类
    218: 999,
    # (219) toyFigures -> 999: 不在统一大类
    219: 999,
    # (220) train -> 999: 不在统一大类
    220: 999,
    # (221) treadmill -> 14: 力量训练与健身
    221: 14,
    # (222) trumpetPerformance -> 26: 演奏管乐
    222: 26,
    # (223) turtle -> 999: 不在统一大类
    223: 999,
    # (224) tyingATie -> 999: 不在统一大类
    224: 999,
    # (225) violinPerformance -> 25: 演奏弦乐
    225: 25,
    # (226) volcanoEruption -> 999: 不在统一大类
    226: 999,
    # (227) walkingWithDog -> 16: 宠物护理与遛狗
    227: 16,
    # (228) washingAnInfant -> 15: 清洁与家务
    228: 15,
    # (229) washingDishes -> 15: 清洁与家务
    229: 15,
    # (230) waterfall -> 999: 不在统一大类
    230: 999,
    # (231) wearLipstick -> 1: 化妆与美容护理
    231: 1,
    # (232) weddingCeremony -> 18: 节日与庆典
    232: 18,
    # (233) weddingDance -> 4: 舞蹈
    233: 4,
    # (234) weddingReception -> 18: 节日与庆典
    234: 18,
    # (235) wheelchairBasketball -> 41: 篮球
    235: 41,
    # (236) wheelchairRace -> 999: 不在统一大类
    236: 999,
    # (237) wheelchairTennis -> 30: 球拍运动
    237: 30,
    # (238) yoga -> 38: 体操运动
    238: 38,
    # (239) yoyoTricks -> 7: 户外休闲娱乐
    239: 7,
}


# 统一大类 ID -> 名称（用于统计信息展示）
UNIFIED_LABEL_NAMES = {
    1: "化妆与美容护理",
    2: "射箭",
    3: "自行车运动",
    4: "舞蹈",
    5: "头发护理",
    6: "口腔与面部清洁",
    7: "户外休闲娱乐",
    8: "极限空中运动",
    9: "骑乘动物与竞技",
    10: "水上划行运动",
    11: "格斗与武术",
    12: "拳击",
    13: "队列与团队表演",
    14: "力量训练与健身",
    15: "清洁与家务",
    16: "宠物护理与遛狗",
    17: "冰雪运动",
    18: "节日与庆典",
    19: "飞盘运动",
    20: "击剑",
    21: "烹饪与食物制作",
    22: "饮品制作",
    23: "乒乓球",
    24: "跳水",
    25: "演奏弦乐",
    26: "演奏管乐",
    27: "演奏打击乐",
    28: "演奏键盘乐",
    29: "足球",
    30: "球拍运动",
    31: "保龄球",
    32: "攀岩与爬绳",
    33: "跳绳",
    34: "水上冲浪运动",
    35: "剃须与身体护理",
    36: "游泳",
    37: "太极拳",
    38: "体操运动",
    39: "编织与手工",
    40: "滑板运动",
    41: "篮球",
    42: "台球与桌游",
    43: "绘画与书写",
}


def get_unified_label_fcvid(fcvid_id: int) -> int:
    """
    输入 FCVID 原编号（1-indexed），返回统一大类编号。
    不在统一大类中的返回 999。
    """
    return FCVID_TO_UNIFIED.get(fcvid_id, 999)


def read_data(file_path: str) -> list[str]:
    """从 txt 文件中读取所有行"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def remap_and_save(data: list[str], output_file: str) -> dict:
    """
    将数据中的原始类标（parts[1]，1-indexed）重映射到统一大类，
    排除 999 类，写入 output_file，并返回统计信息字典。

    与 ActivityNet 版本的区别：
      - 类标字段位于 parts[1]（第二列），而非 parts[2]
      - FCVID 类标从 1 开始编号

    统计信息结构：
    {
        "total_input":    int,
        "total_kept":     int,
        "total_excluded": int,
        "parse_error":    int,
        "per_unified":    {unified_id: count, ...},
    }
    """
    stats = {
        "total_input": len(data),
        "total_kept": 0,
        "total_excluded": 0,
        "parse_error": 0,
        "per_unified": defaultdict(int),
    }

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    kept_lines = []

    for line in data:
        parts = line.split(',')
        if len(parts) <= 1:
            stats["parse_error"] += 1
            stats["total_excluded"] += 1
            continue

        try:
            orig_label = int(parts[1])
        except ValueError:
            print(f"  [警告] 无法解析类标，跳过行: {line}")
            stats["parse_error"] += 1
            stats["total_excluded"] += 1
            continue

        unified_label = get_unified_label_fcvid(orig_label)

        if unified_label == 999:
            stats["total_excluded"] += 1
            continue

        # 替换第二个字段（index=1）为统一大类编号
        parts[1] = str(unified_label)
        kept_lines.append(','.join(parts))

        stats["total_kept"] += 1
        stats["per_unified"][unified_label] += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in kept_lines:
            f.write(line + '\n')

    return stats


def print_stats(file_label: str, stats: dict) -> None:
    """格式化打印统计信息"""
    print(f"\n{'=' * 60}")
    print(f"  数据集: {file_label}")
    print(f"{'=' * 60}")
    print(f"  输入总行数   : {stats['total_input']}")
    print(f"  保留行数     : {stats['total_kept']}")
    print(f"  排除行数     : {stats['total_excluded']}")
    print(f"  解析失败行数 : {stats['parse_error']}")
    print(f"\n  各统一大类样本数（共 {len(stats['per_unified'])} 个大类）:")
    print(f"  {'大类ID':<8} {'大类名称':<20} {'样本数':>8}")
    print(f"  {'-' * 40}")
    for uid in sorted(stats["per_unified"].keys()):
        name = UNIFIED_LABEL_NAMES.get(uid, "未知")
        count = stats["per_unified"][uid]
        print(f"  {uid:<8} {name:<20} {count:>8}")
    print(f"{'=' * 60}\n")


def main():
    # ── 输入 / 输出路径配置 ──────────────────────────────────────
    task_list = [
        {
            "input":  "../Jsonf/fcv_train.txt",
            "output": "../Jsonf_remapped/fcv_train.txt",
            "label":  "fcv_train",
        },
        {
            "input":  "../Jsonf/fcv_test.txt",
            "output": "../Jsonf_remapped/fcv_test.txt",
            "label":  "fcv_test",
        },
        {
            "input":  "../Jsonf/fcv_val.txt",
            "output": "../Jsonf_remapped/fcv_val.txt",
            "label":  "fcv_val",
        },
    ]
    # ─────────────────────────────────────────────────────────────

    for task in task_list:
        input_file  = task["input"]
        output_file = task["output"]
        label       = task["label"]

        print(f"\n处理: {input_file}  →  {output_file}")

        if not os.path.exists(input_file):
            print(f"  [错误] 输入文件不存在，跳过: {input_file}")
            continue

        data  = read_data(input_file)
        stats = remap_and_save(data, output_file)
        print_stats(label, stats)

    print("全部处理完成。")


if __name__ == "__main__":
    main()
