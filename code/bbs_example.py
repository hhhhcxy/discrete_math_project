import argparse
import os
import math
from typing import Iterator, List

class BBS:
    """
    Blum-Blum-Shub 生成器（演示版）
    安全性基于整数分解难题与二次剩余难题；实际使用需大素数 p,q。
    """
    def __init__(self, p: int, q: int, seed: int):
        assert p % 4 == 3 and q % 4 == 3, "p,q 必须满足 ≡ 3 (mod 4)"
        self.n = p * q
        self.x = seed % self.n
        if math.gcd(self.x, self.n) != 1:
            self.x = (self.x + 1) % self.n

    def next_bit(self) -> int:
        self.x = pow(self.x, 2, self.n)
        return self.x & 1

    def next_word(self, k: int = 32) -> int:
        v = 0
        for i in range(k):
            v |= (self.next_bit() << i)
        return v

    def generate(self, n: int, k: int = 32) -> Iterator[int]:
        for _ in range(n):
            yield self.next_word(k)

def build_output_path(name: str, out_dir: str = "out") -> str:
    return os.path.join(out_dir, f"{name}_numbers.txt")

def write_numbers(numbers: List[int], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for num in numbers:
            f.write(f"{num}\n")
    print(f"[OK] Generated {len(numbers)} numbers to {path}")

def main():
    parser = argparse.ArgumentParser(description="BBS 示例：生成随机数文件以供评估")
    parser.add_argument("--name", type=str, default="bbs", help="生成器名称，用于输出文件命名")
    parser.add_argument("--p", type=int, default=1000003, help="素数 p（需满足 p ≡ 3 (mod 4)）")
    parser.add_argument("--q", type=int, default=1000039, help="素数 q（需满足 q ≡ 3 (mod 4)）")
    parser.add_argument("--seed", type=int, default=2025, help="初始种子（需与 n 互素）")
    parser.add_argument("--count", type=int, default=20000, help="生成的整数数量")
    parser.add_argument("--bits", type=int, default=32, help="每个整数由多少位拼接而成")
    parser.add_argument("--out", type=str, default=None, help="输出文件路径（留空则使用 out/{name}_numbers.txt）")
    args = parser.parse_args()

    out_path = args.out or build_output_path(args.name)
    bbs = BBS(p=args.p, q=args.q, seed=args.seed)
    nums = list(bbs.generate(args.count, k=args.bits))
    write_numbers(nums, out_path)

if __name__ == "__main__":
    main()
