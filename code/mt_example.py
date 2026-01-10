import argparse
import os
from typing import Iterator, List

class MersenneTwister:
    """
    梅森旋转算法（Mersenne Twister MT19937）
    周期：2^19937 - 1，高维等分布性，被广泛用于科学计算和模拟
    """
    def __init__(self, seed: int):
        self.n = 624
        self.m = 397
        self.matrix_a = 0x9908b0df
        self.upper_mask = 0x80000000
        self.lower_mask = 0x7fffffff
        self.mt = [0] * self.n
        self.mti = self.n + 1
        
        # 初始化状态向量
        self.mt[0] = seed & 0xffffffff
        for i in range(1, self.n):
            s = 1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i
            self.mt[i] = s & 0xffffffff

    def _twist(self):
        """扭转变换：周期性更新整个状态向量"""
        for i in range(self.n - self.m):
            y = (self.mt[i] & self.upper_mask) | (self.mt[i+1] & self.lower_mask)
            self.mt[i] = self.mt[i + self.m] ^ (y >> 1) ^ (self.matrix_a if y & 1 else 0)
        
        for i in range(self.n - self.m, self.n - 1):
            y = (self.mt[i] & self.upper_mask) | (self.mt[i+1] & self.lower_mask)
            self.mt[i] = self.mt[i - self.n + self.m] ^ (y >> 1) ^ (self.matrix_a if y & 1 else 0)
        
        y = (self.mt[self.n - 1] & self.upper_mask) | (self.mt[0] & self.lower_mask)
        self.mt[self.n - 1] = self.mt[self.m - 1] ^ (y >> 1) ^ (self.matrix_a if y & 1 else 0)
        self.mti = 0

    def next(self) -> int:
        """生成下一个 32 位伪随机数"""
        if self.mti >= self.n:
            self._twist()
        
        y = self.mt[self.mti]
        # 卷绕变换（Tempering）
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        self.mti += 1
        return y & 0xffffffff

    def generate(self, n: int) -> Iterator[int]:
        for _ in range(n):
            yield self.next()

def build_output_path(name: str, out_dir: str = "out") -> str:
    filename = f"{name}_numbers.txt"
    return os.path.join(out_dir, filename)

def write_numbers(numbers: List[int], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for num in numbers:
            f.write(f"{num}\n")
    print(f"[OK] Generated {len(numbers)} numbers to {path}")

def main():
    parser = argparse.ArgumentParser(description="梅森旋转算法示例：生成随机数文件以供评估")
    parser.add_argument("--name", type=str, default="mt", help="生成器名称，用于输出文件命名")
    parser.add_argument("--seed", type=int, default=12345, help="初始种子")
    parser.add_argument("--count", type=int, default=20000, help="生成数量")
    parser.add_argument("--out", type=str, default=None, help="输出文件路径（留空则使用 out/{name}_numbers.txt）")
    args = parser.parse_args()

    out_path = args.out or build_output_path(args.name)
    mt = MersenneTwister(seed=args.seed)
    nums = list(mt.generate(args.count))
    write_numbers(nums, out_path)

if __name__ == "__main__":
    main()
