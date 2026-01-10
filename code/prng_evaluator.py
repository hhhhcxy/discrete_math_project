import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import os
import argparse

class PRNGEvaluator:
    def __init__(self, data):
        """
        初始化评估器
        :param data: 包含随机数的列表或numpy数组 (float 0.0-1.0 或 整数)
        """
        self.data = np.array(data)
        self.n = len(self.data)
        
        # 如果数据主要是整数（如 0-2^32），先归一化到 [0, 1] 方便绘图
        if np.max(self.data) > 1.0:
            self.normalized_data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        else:
            self.normalized_data = self.data

    def test_uniformity(self, bins=100):
        """
        测试均匀分布：绘制直方图并进行卡方检验
        """
        print("-" * 30)
        print("【均匀分布测试 (Uniformity Test)】")
        
        # 1. 绘制直方图
        observed_counts, bin_edges = np.histogram(self.data, bins=bins)
        expected_count = self.n / bins
        
        # 2. 卡方检验 (Chi-Square Test)
        # 零假设 H0: 数据服从均匀分布
        chi2_stat, p_value = stats.chisquare(observed_counts)
        
        print(f"  - 样本数量: {self.n}")
        print(f"  - Chi-Square 统计量: {chi2_stat:.4f}")
        print(f"  - P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("  => 结果: 拒绝零假设 (数据可能不均匀，P < 0.05)")
            result_text = "Fail (Non-Uniform)"
            color = 'red'
        else:
            print("  => 结果: 无法拒绝零假设 (数据看起来是均匀的)")
            result_text = "Pass (Uniform)"
            color = 'green'
            
        return observed_counts, bin_edges, result_text, color

    def test_independence(self):
        """
        测试独立性：计算自相关系数并准备散点图数据
        """
        print("-" * 30)
        print("【独立性测试 (Independence Test)】")
        
        # 构造 (x_i, x_{i+1}) 对
        x = self.normalized_data[:-1]
        y = self.normalized_data[1:]
        
        # 计算滞后1的自相关系数 (Lag-1 Autocorrelation)
        # 相关系数越接近0，线性相关性越低
        correlation_matrix = np.corrcoef(x, y)
        correlation = correlation_matrix[0, 1]
        
        print(f"  - Lag-1 自相关系数: {correlation:.6f}")
        
        if abs(correlation) > 0.05: # 阈值可根据样本量调整
            print("  => 结果: 存在明显相关性 (Fail)")
            result_text = f"Fail (Corr={correlation:.2f})"
            color = 'red'
        else:
            print("  => 结果: 相关性极低 (Pass)")
            result_text = f"Pass (Corr={correlation:.2f})"
            color = 'green'
            
        return x, y, result_text, color

    def plot_results(self, title="PRNG Evaluation"):
        """
        可视化展示：左图为直方图，右图为散点图
        """
        plt.figure(figsize=(12, 5))
        
        # --- 1. 直方图 (Uniformity) ---
        ax1 = plt.subplot(1, 2, 1)
        obs, edges, res_uni, col_uni = self.test_uniformity()
        ax1.hist(self.data, bins=len(obs), color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_title(f"Histogram (Uniformity)\nResult: {res_uni}", color=col_uni, fontweight='bold')
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")

        # --- 2. 散点图 (Independence) ---
        ax2 = plt.subplot(1, 2, 2)
        x, y, res_ind, col_ind = self.test_independence()
        # 使用小点和低透明度，防止点太多糊在一起
        ax2.scatter(x, y, s=1, alpha=0.5, c='purple')
        ax2.set_title(f"Scatter Plot (Independence)\nResult: {res_ind}", color=col_ind, fontweight='bold')
        ax2.set_xlabel("$X_i$")
        ax2.set_ylabel("$X_{i+1}$")
        ax2.set_aspect('equal', 'box') # 保持正方形比例，更容易看出图案

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

# --- 辅助函数：从文件读取 ---
def load_numbers_from_file(filename):
    """
    假设文件格式：每行一个数字
    """
    numbers = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    numbers.append(float(line))
        return numbers
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []

def build_default_file(name: str, out_dir: str = "out") -> str:
    return os.path.join(out_dir, f"{name}_numbers.txt")

# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRNG 评估器：读取 out/{name}_numbers.txt 并进行可视化评估")
    parser.add_argument("--name", type=str, default="bbs", help="生成器名称，用于默认文件名 out/{name}_numbers.txt")
    parser.add_argument("--file", type=str, default=None, help="自定义输入文件路径，优先级高于 --name")
    args = parser.parse_args()

    target_file = args.file or build_default_file(args.name)

    if os.path.exists(target_file):
        print(f"正在分析文件: {target_file} ...")
        data = load_numbers_from_file(target_file)
        if data:
            evaluator = PRNGEvaluator(data)
            evaluator.plot_results(title=f"Analysis of {os.path.basename(target_file)}")
    else:
        print(f"未找到文件 '{target_file}'。")