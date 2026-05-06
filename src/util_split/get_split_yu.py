import os
from collections import defaultdict

# UCF101 原编号(1-indexed) -> 统一大类编号映射
# 不在任何统一大类中的类别映射为 999

UCF101_TO_UNIFIED = {
    # (1) ApplyEyeMakeup -> 1: 化妆与美容护理
    1: 1,
    # (2) ApplyLipstick -> 1: 化妆与美容护理
    2: 1,
    # (3) Archery -> 2: 射箭
    3: 2,
    # (4) BabyCrawling -> 999: 不在统一大类
    4: 999,
    # (5) BalanceBeam -> 38: 体操运动
    5: 38,
    # (6) BandMarching -> 13: 队列与团队表演
    6: 13,
    # (7) BaseballPitch -> 999: 不在统一大类
    7: 999,
    # (8) Basketball -> 41: 篮球
    8: 41,
    # (9) BasketballDunk -> 41: 篮球
    9: 41,
    # (10) BenchPress -> 14: 力量训练与健身
    10: 14,
    # (11) Biking -> 3: 自行车运动
    11: 3,
    # (12) Billiards -> 42: 台球与桌游
    12: 42,
    # (13) BlowDryHair -> 5: 头发护理
    13: 5,
    # (14) BlowingCandles -> 18: 节日与庆典
    14: 18,
    # (15) BodyWeightSquats -> 14: 力量训练与健身
    15: 14,
    # (16) Bowling -> 31: 保龄球
    16: 31,
    # (17) BoxingPunchingBag -> 12: 拳击
    17: 12,
    # (18) BoxingSpeedBag -> 12: 拳击
    18: 12,
    # (19) BreastStroke -> 36: 游泳
    19: 36,
    # (20) BrushingTeeth -> 6: 口腔与面部清洁
    20: 6,
    # (21) CleanAndJerk -> 14: 力量训练与健身
    21: 14,
    # (22) CliffDiving -> 24: 跳水
    22: 24,
    # (23) CricketBowling -> 999: 不在统一大类
    23: 999,
    # (24) CricketShot -> 999: 不在统一大类
    24: 999,
    # (25) CuttingInKitchen -> 21: 烹饪与食物制作
    25: 21,
    # (26) Diving -> 24: 跳水
    26: 24,
    # (27) Drumming -> 27: 演奏打击乐
    27: 27,
    # (28) Fencing -> 20: 击剑
    28: 20,
    # (29) FieldHockeyPenalty -> 999: 不在统一大类
    29: 999,
    # (30) FloorGymnastics -> 38: 体操运动
    30: 38,
    # (31) FrisbeeCatch -> 19: 飞盘运动
    31: 19,
    # (32) FrontCrawl -> 36: 游泳
    32: 36,
    # (33) GolfSwing -> 999: 不在统一大类
    33: 999,
    # (34) Haircut -> 5: 头发护理
    34: 5,
    # (35) Hammering -> 999: 不在统一大类
    35: 999,
    # (36) HammerThrow -> 999: 不在统一大类
    36: 999,
    # (37) HandstandPushups -> 14: 力量训练与健身
    37: 14,
    # (38) HandstandWalking -> 14: 力量训练与健身
    38: 14,
    # (39) HeadMassage -> 999: 不在统一大类
    39: 999,
    # (40) HighJump -> 999: 不在统一大类
    40: 999,
    # (41) HorseRace -> 9: 骑乘动物与竞技
    41: 9,
    # (42) HorseRiding -> 9: 骑乘动物与竞技
    42: 9,
    # (43) HulaHoop -> 999: 不在统一大类
    43: 999,
    # (44) IceDancing -> 4: 舞蹈
    44: 4,
    # (45) JavelinThrow -> 999: 不在统一大类
    45: 999,
    # (46) JugglingBalls -> 7: 户外休闲娱乐
    46: 7,
    # (47) JumpingJack -> 14: 力量训练与健身
    47: 14,
    # (48) JumpRope -> 33: 跳绳
    48: 33,
    # (49) Kayaking -> 10: 水上划行运动
    49: 10,
    # (50) Knitting -> 39: 编织与手工
    50: 39,
    # (51) LongJump -> 999: 不在统一大类
    51: 999,
    # (52) Lunges -> 14: 力量训练与健身
    52: 14,
    # (53) MilitaryParade -> 13: 队列与团队表演
    53: 13,
    # (54) Mixing -> 22: 饮品制作
    54: 22,
    # (55) MoppingFloor -> 15: 清洁与家务
    55: 15,
    # (56) Nunchucks -> 11: 格斗与武术
    56: 11,
    # (57) ParallelBars -> 38: 体操运动
    57: 38,
    # (58) PizzaTossing -> 21: 烹饪与食物制作
    58: 21,
    # (59) PlayingCello -> 25: 演奏弦乐
    59: 25,
    # (60) PlayingDaf -> 26: 演奏管乐
    60: 26,
    # (61) PlayingDhol -> 26: 演奏管乐
    61: 26,
    # (62) PlayingFlute -> 26: 演奏管乐
    62: 26,
    # (63) PlayingGuitar -> 25: 演奏弦乐
    63: 25,
    # (64) PlayingPiano -> 28: 演奏键盘乐
    64: 28,
    # (65) PlayingSitar -> 25: 演奏弦乐
    65: 25,
    # (66) PlayingTabla -> 27: 演奏打击乐
    66: 27,
    # (67) PlayingViolin -> 25: 演奏弦乐
    67: 25,
    # (68) PoleVault -> 999: 不在统一大类
    68: 999,
    # (69) PommelHorse -> 38: 体操运动
    69: 38,
    # (70) PullUps -> 14: 力量训练与健身
    70: 14,
    # (71) Punch -> 11: 格斗与武术
    71: 11,
    # (72) PushUps -> 14: 力量训练与健身
    72: 14,
    # (73) Rafting -> 10: 水上划行运动
    73: 10,
    # (74) RockClimbingIndoor -> 32: 攀岩与爬绳
    74: 32,
    # (75) RopeClimbing -> 32: 攀岩与爬绳
    75: 32,
    # (76) Rowing -> 10: 水上划行运动
    76: 10,
    # (77) SalsaSpin -> 4: 舞蹈
    77: 4,
    # (78) ShavingBeard -> 35: 剃须与身体护理
    78: 35,
    # (79) Shotput -> 999: 不在统一大类
    79: 999,
    # (80) SkateBoarding -> 40: 滑板运动
    80: 40,
    # (81) Skiing -> 17: 冰雪运动
    81: 17,
    # (82) Skijet -> 999: 不在统一大类
    82: 999,
    # (83) SkyDiving -> 8: 极限空中运动
    83: 8,
    # (84) SoccerJuggling -> 29: 足球
    84: 29,
    # (85) SoccerPenalty -> 29: 足球
    85: 29,
    # (86) StillRings -> 38: 体操运动
    86: 38,
    # (87) SumoWrestling -> 11: 格斗与武术
    87: 11,
    # (88) Surfing -> 34: 水上冲浪运动
    88: 34,
    # (89) Swing -> 7: 户外休闲娱乐
    89: 7,
    # (90) TableTennisShot -> 23: 乒乓球
    90: 23,
    # (91) TaiChi -> 37: 太极拳
    91: 37,
    # (92) TennisSwing -> 30: 球拍运动
    92: 30,
    # (93) ThrowDiscus -> 999: 不在统一大类
    93: 999,
    # (94) TrampolineJumping -> 38: 体操运动
    94: 38,
    # (95) Typing -> 999: 不在统一大类
    95: 999,
    # (96) UnevenBars -> 38: 体操运动
    96: 38,
    # (97) VolleyballSpiking -> 999: 不在统一大类
    97: 999,
    # (98) WalkingWithDog -> 16: 宠物护理与遛狗
    98: 16,
    # (99) WallPushups -> 14: 力量训练与健身
    99: 14,
    # (100) WritingOnBoard -> 43: 绘画与书写
    100: 43,
    # (101) YoYo -> 7: 户外休闲娱乐
    101: 7,
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
    return UCF101_TO_UNIFIED.get(fcvid_id, 999)


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
            "input":  "../Jsonu/train.txt",
            "output": "../Jsonu_remapped/train.txt",
            "label":  "train",
        },
        {
            "input":  "../Jsonu/test.txt",
            "output": "../Jsonu_remapped/test.txt",
            "label":  "test",
        },
        {
            "input":  "../Jsonu/val.txt",
            "output": "../Jsonu_remapped/val.txt",
            "label":  "val",
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
