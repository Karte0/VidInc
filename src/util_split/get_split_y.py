import os
from collections import defaultdict

ACTIVITYNET_TO_UNIFIED = {
    # 0: Applying sunscreen -> 1: 化妆与美容护理
    0: 1,
    # 1: Archery -> 2: 射箭
    1: 2,
    # 2: Arm wrestling -> 11: 格斗与武术
    2: 11,
    # 3: Assembling bicycle -> 3: 自行车运动
    3: 3,
    # 4: BMX -> 3: 自行车运动
    4: 3,
    # 5: Baking cookies -> 21: 烹饪与食物制作
    5: 21,
    # 6: Ballet -> 4: 舞蹈
    6: 4,
    # 7: Bathing dog -> 16: 宠物护理与遛狗
    7: 16,
    # 8: Baton twirling -> 13: 队列与团队表演
    8: 13,
    # 9: Beach soccer -> 29: 足球
    9: 29,
    # 10: Beer pong -> 999: 不在统一大类
    10: 999,
    # 11: Belly dance -> 4: 舞蹈
    11: 4,
    # 12: Blow-drying hair -> 5: 头发护理
    12: 5,
    # 13: Blowing leaves -> 999: 不在统一大类
    13: 999,
    # 14: Braiding hair -> 5: 头发护理
    14: 5,
    # 15: Breakdancing -> 4: 舞蹈
    15: 4,
    # 16: Brushing hair -> 5: 头发护理
    16: 5,
    # 17: Brushing teeth -> 6: 口腔与面部清洁
    17: 6,
    # 18: Building sandcastles -> 7: 户外休闲娱乐
    18: 7,
    # 19: Bullfighting -> 999: 不在统一大类
    19: 999,
    # 20: Bungee jumping -> 8: 极限空中运动
    20: 8,
    # 21: Calf roping -> 9: 骑乘动物与竞技
    21: 9,
    # 22: Camel ride -> 9: 骑乘动物与竞技
    22: 9,
    # 23: Canoeing -> 10: 水上划行运动
    23: 10,
    # 24: Capoeira -> 11: 格斗与武术
    24: 11,
    # 25: Carving jack-o-lanterns -> 18: 节日与庆典
    25: 18,
    # 26: Changing car wheel -> 999: 不在统一大类
    26: 999,
    # 27: Cheerleading -> 13: 队列与团队表演
    27: 13,
    # 28: Chopping wood -> 999: 不在统一大类
    28: 999,
    # 29: Clean and jerk -> 14: 力量训练与健身
    29: 14,
    # 30: Cleaning shoes -> 15: 清洁与家务
    30: 15,
    # 31: Cleaning sink -> 15: 清洁与家务
    31: 15,
    # 32: Cleaning windows -> 15: 清洁与家务
    32: 15,
    # 33: Clipping cat claws -> 16: 宠物护理与遛狗
    33: 16,
    # 34: Cricket -> 999: 不在统一大类
    34: 999,
    # 35: Croquet -> 999: 不在统一大类
    35: 999,
    # 36: Cumbia -> 4: 舞蹈
    36: 4,
    # 37: Curling -> 17: 冰雪运动
    37: 17,
    # 38: Cutting the grass -> 999: 不在统一大类
    38: 999,
    # 39: Decorating the Christmas tree -> 18: 节日与庆典
    39: 18,
    # 40: Disc dog -> 19: 飞盘运动
    40: 19,
    # 41: Discus throw -> 999: 不在统一大类
    41: 999,
    # 42: Dodgeball -> 999: 不在统一大类
    42: 999,
    # 43: Doing a powerbomb -> 11: 格斗与武术
    43: 11,
    # 44: Doing crunches -> 14: 力量训练与健身
    44: 14,
    # 45: Doing fencing -> 20: 击剑
    45: 20,
    # 46: Doing karate -> 11: 格斗与武术
    46: 11,
    # 47: Doing kickboxing -> 12: 拳击
    47: 12,
    # 48: Doing motocross -> 999: 不在统一大类
    48: 999,
    # 49: Doing nails -> 1: 化妆与美容护理
    49: 1,
    # 50: Doing step aerobics -> 14: 力量训练与健身
    50: 14,
    # 51: Drinking beer -> 999: 不在统一大类
    51: 999,
    # 52: Drinking coffee -> 999: 不在统一大类
    52: 999,
    # 53: Drum corps -> 13: 队列与团队表演
    53: 13,
    # 54: Elliptical trainer -> 14: 力量训练与健身
    54: 14,
    # 55: Fixing bicycle -> 3: 自行车运动
    55: 3,
    # 56: Fixing the roof -> 999: 不在统一大类
    56: 999,
    # 57: Fun sliding down -> 7: 户外休闲娱乐
    57: 7,
    # 58: Futsal -> 29: 足球
    58: 29,
    # 59: Gargling mouthwash -> 6: 口腔与面部清洁
    59: 6,
    # 60: Getting a haircut -> 5: 头发护理
    60: 5,
    # 61: Getting a piercing -> 35: 剃须与身体护理
    61: 35,
    # 62: Getting a tattoo -> 35: 剃须与身体护理
    62: 35,
    # 63: Grooming dog -> 16: 宠物护理与遛狗
    63: 16,
    # 64: Grooming horse -> 16: 宠物护理与遛狗
    64: 16,
    # 65: Hammer throw -> 999: 不在统一大类
    65: 999,
    # 66: Hand car wash -> 999: 不在统一大类
    66: 999,
    # 67: Hand washing clothes -> 15: 清洁与家务
    67: 15,
    # 68: Hanging wallpaper -> 999: 不在统一大类
    68: 999,
    # 69: Having an ice cream -> 999: 不在统一大类
    69: 999,
    # 70: High jump -> 999: 不在统一大类
    70: 999,
    # 71: Hitting a pinata -> 7: 户外休闲娱乐
    71: 7,
    # 72: Hopscotch -> 7: 户外休闲娱乐
    72: 7,
    # 73: Horseback riding -> 9: 骑乘动物与竞技
    73: 9,
    # 74: Hula hoop -> 999: 不在统一大类
    74: 999,
    # 75: Hurling -> 999: 不在统一大类
    75: 999,
    # 76: Ice fishing -> 17: 冰雪运动
    76: 17,
    # 77: Installing carpet -> 999: 不在统一大类
    77: 999,
    # 78: Ironing clothes -> 15: 清洁与家务
    78: 15,
    # 79: Javelin throw -> 999: 不在统一大类
    79: 999,
    # 80: Kayaking -> 10: 水上划行运动
    80: 10,
    # 81: Kite flying -> 7: 户外休闲娱乐
    81: 7,
    # 82: Kneeling -> 999: 不在统一大类
    82: 999,
    # 83: Knitting -> 39: 编织与手工
    83: 39,
    # 84: Laying tile -> 999: 不在统一大类
    84: 999,
    # 85: Layup drill in basketball -> 41: 篮球
    85: 41,
    # 86: Long jump -> 999: 不在统一大类
    86: 999,
    # 87: Longboarding -> 40: 滑板运动
    87: 40,
    # 88: Making a cake -> 21: 烹饪与食物制作
    88: 21,
    # 89: Making a lemonade -> 22: 饮品制作
    89: 22,
    # 90: Making a sandwich -> 21: 烹饪与食物制作
    90: 21,
    # 91: Making an omelette -> 21: 烹饪与食物制作
    91: 21,
    # 92: Mixing drinks -> 22: 饮品制作
    92: 22,
    # 93: Mooping floor -> 15: 清洁与家务
    93: 15,
    # 94: Mowing the lawn -> 999: 不在统一大类
    94: 999,
    # 95: Paintball -> 999: 不在统一大类
    95: 999,
    # 96: Painting -> 43: 绘画与书写
    96: 43,
    # 97: Painting fence -> 43: 绘画与书写
    97: 43,
    # 98: Painting furniture -> 43: 绘画与书写
    98: 43,
    # 99: Peeling potatoes -> 21: 烹饪与食物制作
    99: 21,
    # 100: Ping-pong -> 23: 乒乓球
    100: 23,
    # 101: Plastering -> 999: 不在统一大类
    101: 999,
    # 102: Plataform diving -> 24: 跳水
    102: 24,
    # 103: Playing accordion -> 26: 演奏管乐
    103: 26,
    # 104: Playing badminton -> 30: 球拍运动
    104: 30,
    # 105: Playing bagpipes -> 26: 演奏管乐
    105: 26,
    # 106: Playing beach volleyball -> 999: 不在统一大类
    106: 999,
    # 107: Playing blackjack -> 42: 台球与桌游
    107: 42,
    # 108: Playing congas -> 27: 演奏打击乐
    108: 27,
    # 109: Playing drums -> 27: 演奏打击乐
    109: 27,
    # 110: Playing field hockey -> 999: 不在统一大类
    110: 999,
    # 111: Playing flauta -> 26: 演奏管乐
    111: 26,
    # 112: Playing guitarra -> 25: 演奏弦乐
    112: 25,
    # 113: Playing harmonica -> 26: 演奏管乐
    113: 26,
    # 114: Playing ice hockey -> 999: 不在统一大类
    114: 999,
    # 115: Playing kickball -> 29: 足球
    115: 29,
    # 116: Playing lacrosse -> 999: 不在统一大类
    116: 999,
    # 117: Playing piano -> 28: 演奏键盘乐
    117: 28,
    # 118: Playing polo -> 9: 骑乘动物与竞技
    118: 9,
    # 119: Playing pool -> 42: 台球与桌游
    119: 42,
    # 120: Playing racquetball -> 30: 球拍运动
    120: 30,
    # 121: Playing rubik cube -> 42: 台球与桌游
    121: 42,
    # 122: Playing saxophone -> 26: 演奏管乐
    122: 26,
    # 123: Playing squash -> 30: 球拍运动
    123: 30,
    # 124: Playing ten pins -> 31: 保龄球
    124: 31,
    # 125: Playing violin -> 25: 演奏弦乐
    125: 25,
    # 126: Playing water polo -> 999: 不在统一大类
    126: 999,
    # 127: Pole vault -> 999: 不在统一大类
    127: 999,
    # 128: Polishing forniture -> 15: 清洁与家务
    128: 15,
    # 129: Polishing shoes -> 15: 清洁与家务
    129: 15,
    # 130: Powerbocking -> 8: 极限空中运动
    130: 8,
    # 131: Preparing pasta -> 21: 烹饪与食物制作
    131: 21,
    # 132: Preparing salad -> 21: 烹饪与食物制作
    132: 21,
    # 133: Putting in contact lenses -> 1: 化妆与美容护理
    133: 1,
    # 134: Putting on makeup -> 1: 化妆与美容护理
    134: 1,
    # 135: Putting on shoes -> 999: 不在统一大类
    135: 999,
    # 136: Rafting -> 10: 水上划行运动
    136: 10,
    # 137: Raking leaves -> 999: 不在统一大类
    137: 999,
    # 138: Removing curlers -> 5: 头发护理
    138: 5,
    # 139: Removing ice from car -> 999: 不在统一大类
    139: 999,
    # 140: Riding bumper cars -> 7: 户外休闲娱乐
    140: 7,
    # 141: River tubing -> 10: 水上划行运动
    141: 10,
    # 142: Rock climbing -> 32: 攀岩与爬绳
    142: 32,
    # 143: Rock-paper-scissors -> 42: 台球与桌游
    143: 42,
    # 144: Rollerblading -> 999: 不在统一大类
    144: 999,
    # 145: Roof shingle removal -> 999: 不在统一大类
    145: 999,
    # 146: Rope skipping -> 33: 跳绳
    146: 33,
    # 147: Running a marathon -> 999: 不在统一大类
    147: 999,
    # 148: Sailing -> 999: 不在统一大类
    148: 999,
    # 149: Scuba diving -> 999: 不在统一大类
    149: 999,
    # 150: Sharpening knives -> 21: 烹饪与食物制作
    150: 21,
    # 151: Shaving -> 35: 剃须与身体护理
    151: 35,
    # 152: Shaving legs -> 35: 剃须与身体护理
    152: 35,
    # 153: Shot put -> 999: 不在统一大类
    153: 999,
    # 154: Shoveling snow -> 17: 冰雪运动
    154: 17,
    # 155: Shuffleboard -> 999: 不在统一大类
    155: 999,
    # 156: Skateboarding -> 40: 滑板运动
    156: 40,
    # 157: Skiing -> 17: 冰雪运动
    157: 17,
    # 158: Slacklining -> 38: 体操运动
    158: 38,
    # 159: Smoking a cigarette -> 999: 不在统一大类
    159: 999,
    # 160: Smoking hookah -> 999: 不在统一大类
    160: 999,
    # 161: Snatch -> 14: 力量训练与健身
    161: 14,
    # 162: Snow tubing -> 17: 冰雪运动
    162: 17,
    # 163: Snowboarding -> 17: 冰雪运动
    163: 17,
    # 164: Spinning -> 999: 不在统一大类
    164: 999,
    # 165: Spread mulch -> 999: 不在统一大类
    165: 999,
    # 166: Springboard diving -> 24: 跳水
    166: 24,
    # 167: Starting a campfire -> 7: 户外休闲娱乐
    167: 7,
    # 168: Sumo -> 11: 格斗与武术
    168: 11,
    # 169: Surfing -> 34: 水上冲浪运动
    169: 34,
    # 170: Swimming -> 36: 游泳
    170: 36,
    # 171: Swinging at the playground -> 7: 户外休闲娱乐
    171: 7,
    # 172: Table soccer -> 999: 不在统一大类
    172: 999,
    # 173: Tai chi -> 37: 太极拳
    173: 37,
    # 174: Tango -> 4: 舞蹈
    174: 4,
    # 175: Tennis serve with ball bouncing -> 30: 球拍运动
    175: 30,
    # 176: Throwing darts -> 999: 不在统一大类
    176: 999,
    # 177: Trimming branches or hedges -> 999: 不在统一大类
    177: 999,
    # 178: Triple jump -> 999: 不在统一大类
    178: 999,
    # 179: Tug of war -> 999: 不在统一大类
    179: 999,
    # 180: Tumbling -> 38: 体操运动
    180: 38,
    # 181: Using parallel bars -> 38: 体操运动
    181: 38,
    # 182: Using the balance beam -> 38: 体操运动
    182: 38,
    # 183: Using the monkey bar -> 999: 不在统一大类
    183: 999,
    # 184: Using the pommel horse -> 38: 体操运动
    184: 38,
    # 185: Using the rowing machine -> 999: 不在统一大类
    185: 999,
    # 186: Using uneven bars -> 38: 体操运动
    186: 38,
    # 187: Vacuuming floor -> 15: 清洁与家务
    187: 15,
    # 188: Volleyball -> 999: 不在统一大类
    188: 999,
    # 189: Wakeboarding -> 34: 水上冲浪运动
    189: 34,
    # 190: Walking the dog -> 16: 宠物护理与遛狗
    190: 16,
    # 191: Washing dishes -> 15: 清洁与家务
    191: 15,
    # 192: Washing face -> 6: 口腔与面部清洁
    192: 6,
    # 193: Washing hands -> 6: 口腔与面部清洁
    193: 6,
    # 194: Waterskiing -> 34: 水上冲浪运动
    194: 34,
    # 195: Waxing skis -> 17: 冰雪运动
    195: 17,
    # 196: Welding -> 999: 不在统一大类
    196: 999,
    # 197: Windsurfing -> 34: 水上冲浪运动
    197: 34,
    # 198: Wrapping presents -> 18: 节日与庆典
    198: 18,
    # 199: Zumba -> 4: 舞蹈
    199: 4,
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


def get_unified_label(activitynet_id: int) -> int:
    """
    输入 ActivityNet 原编号，返回统一大类编号。
    不在统一大类中的返回 999。
    """
    return ACTIVITYNET_TO_UNIFIED.get(activitynet_id, 999)


def read_data(file_path: str) -> list[str]:
    """从 txt 文件中读取所有行"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def remap_and_save(data: list[str], output_file: str) -> dict:
    """
    将数据中的原始类标重映射到统一大类，排除 999 类，
    写入 output_file，并返回统计信息字典。

    统计信息结构：
    {
        "total_input":   int,   # 输入总行数
        "total_kept":    int,   # 保留行数
        "total_excluded":int,   # 排除行数（999 或解析失败）
        "parse_error":   int,   # 解析失败行数
        "per_unified":   {unified_id: count, ...},  # 每个统一大类的样本数
    }
    """
    stats = {
        "total_input": len(data),
        "total_kept": 0,
        "total_excluded": 0,
        "parse_error": 0,
        "per_unified": defaultdict(int),
    }

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    kept_lines = []

    for line in data:
        parts = line.split(',')
        if len(parts) <= 2:
            # 字段不足，跳过
            stats["parse_error"] += 1
            stats["total_excluded"] += 1
            continue

        try:
            orig_label = int(parts[2])
        except ValueError:
            print(f"  [警告] 无法解析类标，跳过行: {line}")
            stats["parse_error"] += 1
            stats["total_excluded"] += 1
            continue

        unified_label = get_unified_label(orig_label)

        if unified_label == 999:
            # 不在统一大类，排除
            stats["total_excluded"] += 1
            continue

        # 替换第三个字段（index=2）为统一大类编号
        parts[2] = str(unified_label)
        new_line = ','.join(parts)
        kept_lines.append(new_line)

        stats["total_kept"] += 1
        stats["per_unified"][unified_label] += 1

    # 写出结果
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
    print(f"  {'-'*40}")
    for uid in sorted(stats["per_unified"].keys()):
        name = UNIFIED_LABEL_NAMES.get(uid, "未知")
        count = stats["per_unified"][uid]
        print(f"  {uid:<8} {name:<20} {count:>8}")
    print(f"{'=' * 60}\n")


def main():
    # ── 输入 / 输出路径配置 ──────────────────────────────────────
    task_list = [
        {
            "input":  "../Json/train.txt",
            "output": "../Json_remapped/train.txt",
            "label":  "train",
        },
        {
            "input":  "../Json/test.txt",
            "output": "../Json_remapped/test.txt",
            "label":  "test",
        },
        {
            "input":  "../Json/val.txt",
            "output": "../Json_remapped/val.txt",
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
