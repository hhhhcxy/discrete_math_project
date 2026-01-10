import argparse
import os
from typing import Iterator, List

class LCG:
    def __init__(self, seed: int, a: int , c: int , m: int ):
        self.state = seed % m
        self.a = a
        self.c = c
        self.m = m

    def next(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

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
    parser = argparse.ArgumentParser(description="LCG 示例：生成随机数文件以供评估")
    parser.add_argument("--name", type=str, default="lcg", help="生成器名称，用于输出文件命名")
    parser.add_argument("--seed", type=int, default=12345, help="初始种子")
    parser.add_argument("--count", type=int, default=20000, help="生成数量")
    parser.add_argument("--a", type=int, default=65, help="乘数 a")
    parser.add_argument("--c", type=int, default=1, help="增量 c")
    parser.add_argument("--m", type=int, default=2**16, help="模 m")
    parser.add_argument("--out", type=str, default=None, help="输出文件路径（留空则使用 out/{name}_numbers.txt）")
    args = parser.parse_args()

    out_path = args.out or build_output_path(args.name)
    lcg = LCG(seed=args.seed, a=args.a, c=args.c, m=args.m)
    nums = list(lcg.generate(args.count))
    write_numbers(nums, out_path)

if __name__ == "__main__":
    main()
