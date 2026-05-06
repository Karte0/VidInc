import random
import os

# 读取数据集辅助信息
def read_data(file_path):
    """
    从txt文件中读取数据
    :param file_path: 数据集辅助信息文件路径
    :return: 数据列表，每条数据是一个字符串
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 保存分组结果
def process_and_save(data, output_dir, groups):
    """
    保存分组结果
    :param data: 数据列表，每条数据是一个字符串
    :param output_dir: 输出文件夹路径
    :param groups: 分组后的类别列表
    """
    # 创建类别到数据的映射
    category_to_data = {i: [] for i in range(200)}
    for entry in data:
        parts = entry.split(',')
        if len(parts) > 2:  # 确保数据中至少有三个字段
            try:
                category = int(parts[2])  # 只提取第二个逗号之后的第一个数字
                category_to_data[category].append(entry)
            except ValueError:
                print(f"跳过无效类标: {entry}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存每组数据到文件
    for i, group in enumerate(groups):
        filename = os.path.join(output_dir, f"{i + 1}.txt")
        with open(filename, 'w') as f:
            for category in group:
                for entry in category_to_data[category]:
                    f.write(f"{entry}\n")
        print(f"生成文件: {filename}")

# 主函数
def main():
    # 输入文件路径列表（3个数据集）
    input_files = ["../Json/train.txt", "../Json/test.txt", "../Json/val.txt"]  # 数据集文件路径
    output_dirs = ["../split", "../split_test", "../split_val"]  # 每个数据集的输出目录

    # 类别列表（0-199）
    categories = list(range(200))

    # 随机打乱类别
    random.shuffle(categories)

    # 将 200 类分为 10 组，每组 20 类
    groups = [categories[i * 20:(i + 1) * 20] for i in range(10)]

    # 遍历每个数据集
    for input_file, output_dir in zip(input_files, output_dirs):
        print(f"处理数据集: {input_file}")
        # 读取数据
        data = read_data(input_file)

        # 保存分组结果
        process_and_save(data, output_dir, groups)

if __name__ == "__main__":
    main()

