#!/usr/bin/env python3
"""
比较两个数据文件中各变量的平均绝对百分比误差（MAPE）。
文件格式：.csv 逗号分隔，第一行为单位，第二行为列名，从第三行开始是数据。
用法：python compare_fds_smoke.py <基准文件.csv> <测试文件.csv> [--n_points 采样点数N]
"""

import sys
import argparse
import time
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_data(filepath):
    """读取数据文件，第一行为单位，第二行为列名，从第三行开始是数据，逗号分隔。"""
    try:
        # 第一行是单位，第二行是列名，所以 header=1 表示第二行作为列名
        # usecols 在读取前无法确定，先读取所有数据
        df = pd.read_csv(filepath, sep=',', header=1)
    except FileNotFoundError:
        print(f"错误：文件 {filepath} 不存在。")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件 {filepath} 失败：{e}")
        sys.exit(1)

    if df.empty:
        print(f"错误：文件 {filepath} 为空。")
        sys.exit(1)

    # 清理列名（去除可能的空格）
    df.columns = [col.strip() for col in df.columns]

    # 移除 pandas 自动添加的Unnamed列（文件末尾多余逗号导致）
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"提示：已移除空列 {unnamed_cols}")

    # 确保第一列是时间，转换为浮点数
    time_col = df.columns[0]
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    original_rows = len(df)
    df = df.dropna(subset=[time_col])
    dropped = original_rows - len(df)
    if dropped > 0:
        print(f"提示：文件 {filepath} 中有 {dropped} 行时间数据无效，已被忽略。")
    if df.empty:
        print(f"错误：文件 {filepath} 中没有有效的时间数据。")
        sys.exit(1)
    # 按时间排序
    df = df.sort_values(by=time_col).reset_index(drop=True)
    return df


def get_data_as_float(df, col_name):
    """安全地将 DataFrame 列转换为 float64 数组。"""
    try:
        return df[col_name].values.astype(np.float64)
    except (ValueError, TypeError) as e:
        print(f"警告：列 '{col_name}' 包含非数值数据，转换为 float 失败：{e}")
        return np.full(len(df), np.nan, dtype=np.float64)


def validate_headers(base_df, test_df):
    """检查两个文件的列名是否一致，返回差异信息。"""
    base_cols = list(base_df.columns)
    test_cols = list(test_df.columns)

    if base_cols == test_cols:
        return None

    print("警告：基准文件和测试文件的列名不完全一致。")
    print(f"  基准文件列名：{base_cols}")
    print(f"  测试文件列名：{test_cols}")

    base_set = set(base_cols)
    test_set = set(test_cols)
    only_in_base = base_set - test_set
    only_in_test = test_set - base_set

    if only_in_base:
        print(f"  仅在基准文件中存在：{list(only_in_base)}")
    if only_in_test:
        print(f"  仅在测试文件中存在：{list(only_in_test)}")

    return {
        'only_in_base': list(only_in_base),
        'only_in_test': list(only_in_test),
    }


def print_data_stats(label, df, var_cols):
    """打印数据文件的基本统计信息。"""
    time_col = df.columns[0]
    t = df[time_col].values.astype(np.float64)
    print(f"\n{label}:")
    print(f"  时间范围: {t.min():.6f} ~ {t.max():.6f} ({len(t)} 个数据点)")
    print(f"  变量数量: {len(var_cols)}")
    if len(var_cols) > 0:
        print(f"  变量列表: {', '.join(var_cols)}")


def interpolate_test_to_base(test_df, test_t, base_t, var_cols):
    """将测试文件线性插值到基准文件的时间点上，返回插值后的二维数组。"""
    # 检查时间范围
    if base_t.min() < test_t.min() or base_t.max() > test_t.max():
        print("警告：基准时间超出测试文件时间范围，外插可能导致不准确。")

    n_points = len(base_t)
    n_vars = len(var_cols)
    result = np.zeros((n_points, n_vars), dtype=np.float64)

    for j, col in enumerate(var_cols):
        y = get_data_as_float(test_df, col)
        if np.all(np.isnan(y)):
            result[:, j] = np.nan
            continue
        # 移除 NaN 点用于创建插值函数
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:
            result[:, j] = np.nan
            continue
        f = interp1d(
            test_t[valid_mask],
            y[valid_mask],
            kind='linear',
            fill_value='extrapolate'
        )
        result[:, j] = f(base_t)

    return result


def find_sampling_points(t_base, base_data, n_points=10, threshold=1e-12):
    """
    寻找 n_points 个采样时间点，使得所有变量在这些点上的基准值均 >= threshold。
    采用逐步扩大采样区间的方式（从中间收缩区域开始，逐渐扩展到全区间）。
    返回采样时间点数组，若无法满足则返回全区间均匀点并警告。
    """
    t_min, t_max = t_base.min(), t_base.max()
    t_range = t_max - t_min

    if t_range == 0:
        print("警告：时间范围为 0，无法采样。")
        return np.array([t_min])

    # 尝试不同的边界收缩比例，从 0.1 到 0，步长 0.05
    margins = np.arange(0.1, -0.05, -0.05)
    for margin in margins:
        low = t_min + margin * t_range
        high = t_max - margin * t_range
        if low >= high:
            continue
        t_samples = np.linspace(low, high, n_points)
        # 检查所有变量是否满足阈值
        base_at_samples = np.column_stack([
            np.interp(t_samples, t_base, base_data[:, j]) for j in range(base_data.shape[1])
        ])
        if np.all(np.abs(base_at_samples) >= threshold):
            return t_samples

    # 若全区间仍无法满足，使用全区间均匀点并警告
    print("警告：无法找到所有变量均满足阈值的采样点，将使用全区间均匀采样点。")
    return np.linspace(t_min, t_max, n_points)


def plot_comparison_plots(t_samples, base_sampled, test_sampled, var_cols, output_dir="comparison_plots"):
    """
    为每个变量绘制基准和测试采样点对比图，横坐标为时间，纵坐标为变量值。
    保存图片为 TIFF 格式，以变量名命名。
    """
    # 变量单位映射字典
    unit_map = {
        'P1_TEMP': '°C',
        'P1_CO': 'mol/mol',
        'P1_VISI': 'm',
    }

    # 变量显示名称映射字典
    display_name_map = {
        'P1_TEMP': 'Temperature',
        'P1_CO': 'CO',
        'P1_VISI': 'Visibility',
    }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    n_vars = len(var_cols)
    for j, col in enumerate(var_cols):
        base_vals = base_sampled[:, j]
        test_vals = test_sampled[:, j]

        # 创建图表（方形）
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制基准采样点（用折线连接）
        ax.plot(t_samples, base_vals, 'b-o', label='Base', linewidth=2, markersize=6)

        # 绘制测试采样点（用折线连接）
        ax.plot(t_samples, test_vals, 'r-s', label='Test', linewidth=2, markersize=6)

        # 获取变量单位和显示名称
        unit = unit_map.get(col, '')
        display_name = display_name_map.get(col, col)

        # 设置标签和标题（横坐标单位为s，适合论文使用的大字号）
        if unit:
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel(f'{display_name} ({unit})', fontsize=16)
        else:
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel(display_name, fontsize=16)
        ax.set_title(f'Comparison: {display_name}', fontsize=18)
        ax.legend(loc='best', fontsize=14)
        # 设置刻度标签字号
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 保存为 TIFF 格式
        safe_name = col.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_file = os.path.join(output_dir, f"{safe_name}.tiff")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  对比图已保存: {output_file}")

    print(f"\n所有变量对比图已保存至目录: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="比较两个数据文件中各变量的平均绝对百分比误差（MAPE）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python compare_fds_smoke.py reference.dat test.dat --n_points 20"
    )
    parser.add_argument("base_file", help="基准文件路径")
    parser.add_argument("test_file", help="测试文件路径")
    parser.add_argument(
        "--n_points",
        type=int,
        default=10,
        help="采样点数量（默认: 10）"
    )
    args = parser.parse_args()

    start_time = time.time()

    # 读取数据
    print("=" * 60)
    print("  数据文件 MAPE 比较工具")
    print("=" * 60)

    base_df = read_data(args.base_file)
    test_df = read_data(args.test_file)

    # 提取时间列和变量列
    time_col = base_df.columns[0]
    var_cols = base_df.columns[1:].tolist()

    if len(var_cols) == 0:
        print("错误：文件中没有变量列（除时间列外）。")
        sys.exit(1)

    t_base = get_data_as_float(base_df, time_col)
    base_data = np.column_stack([get_data_as_float(base_df, col) for col in var_cols])

    # 打印数据统计信息
    print_data_stats("基准文件", base_df, var_cols)
    test_t = get_data_as_float(test_df, time_col)
    print_data_stats("测试文件", test_df, var_cols)

    # 验证列名一致性
    col_diff = validate_headers(base_df, test_df)

    # 将测试文件插值到基准时间点
    test_data_interp = interpolate_test_to_base(test_df, test_t, t_base, var_cols)

    # 寻找合适的采样时间点
    t_samples = find_sampling_points(t_base, base_data, n_points=args.n_points, threshold=1e-12)

    # 使用 np.interp 直接在采样点取值（比创建 interp1d 更高效）
    base_sampled = np.column_stack([
        np.interp(t_samples, t_base, base_data[:, j]) for j in range(len(var_cols))
    ])
    test_sampled = np.column_stack([
        np.interp(t_samples, t_base, test_data_interp[:, j]) for j in range(len(var_cols))
    ])

    # 计算每个变量的 MAPE
    mape_results = {}
    print("\n" + "=" * 60)
    print("  采样点 MAPE 计算结果")
    print("=" * 60)
    print(f"  {'变量名':<30} {'MAPE':>15}")
    print(f"  {'-' * 30} {'-' * 15}")

    for j, col in enumerate(var_cols):
        base_vals = base_sampled[:, j]
        test_vals = test_sampled[:, j]
        # 过滤 NaN 和接近零的值
        valid_mask = np.isfinite(base_vals) & np.isfinite(test_vals) & (np.abs(base_vals) >= 1e-12)
        if not valid_mask.all():
            ignored = (~valid_mask).sum()
            print(f"  提示：变量 {col} 有 {ignored} 个采样点被忽略。")
        if valid_mask.sum() == 0:
            mape = np.nan
        else:
            ape = np.abs((test_vals[valid_mask] - base_vals[valid_mask]) / base_vals[valid_mask]) * 100
            mape = np.mean(ape)
        mape_results[col] = mape

    for col, mape in mape_results.items():
        if np.isnan(mape):
            print(f"  {col:<28} {'N/A':>15}")
        else:
            print(f"  {col:<28} {mape:>14.4f}%")

    # 保存采样点数据
    base_out_df = pd.DataFrame(
        np.column_stack([t_samples, base_sampled]),
        columns=base_df.columns
    )
    test_out_df = pd.DataFrame(
        np.column_stack([t_samples, test_sampled]),
        columns=test_df.columns
    )

    base_out_file = "base_sampled.csv"
    test_out_file = "test_sampled.csv"
    base_out_df.to_csv(base_out_file, index=False, sep=' ')
    test_out_df.to_csv(test_out_file, index=False, sep=' ')
    print(f"\n采样点基准数据已保存至 {base_out_file}")
    print(f"采样点测试数据已保存至 {test_out_file}")

    # 绘制每个变量的对比图
    print("\n" + "=" * 60)
    print("  绘制变量对比图")
    print("=" * 60)
    plot_comparison_plots(t_samples, base_sampled, test_sampled, var_cols)

    elapsed = time.time() - start_time
    print(f"\n处理完成，耗时 {elapsed:.3f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()