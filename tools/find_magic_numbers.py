#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict

TRIVIAL_NUMBERS = {0, 1, 2, -1}
EXCLUDE_PATTERNS = [
    r'printf\(',
    r'#define',
    r'constexpr',
    r'//',
    r'0x[0-9a-fA-F]+',
]

def should_exclude_line(line):
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, line):
            return True
    return False

def extract_numbers(content):
    float_pattern = r'\b\d+\.\d+f?\b'
    int_pattern = r'\b\d{3,}\b'

    numbers = defaultdict(list)

    for line_num, line in enumerate(content.split('\n'), 1):
        if should_exclude_line(line):
            continue

        for match in re.finditer(float_pattern, line):
            num_str = match.group()
            try:
                num = float(num_str.rstrip('f'))
                if num not in TRIVIAL_NUMBERS:
                    numbers[num_str].append(line_num)
            except ValueError:
                pass

        for match in re.finditer(int_pattern, line):
            num_str = match.group()
            try:
                num = int(num_str)
                if num not in TRIVIAL_NUMBERS:
                    numbers[num_str].append(line_num)
            except ValueError:
                pass

    return numbers

def scan_directory(directory):
    all_numbers = defaultdict(lambda: defaultdict(list))

    for cu_file in Path(directory).rglob("*.cu"):
        try:
            with open(cu_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                numbers = extract_numbers(content)

                for num_str, line_nums in numbers.items():
                    all_numbers[num_str][str(cu_file.relative_to(directory))].extend(line_nums)
        except Exception as e:
            print(f"Error scanning {cu_file}: {e}")

    for cuh_file in Path(directory).rglob("*.cuh"):
        if "constants.cuh" in str(cuh_file):
            continue

        try:
            with open(cuh_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                numbers = extract_numbers(content)

                for num_str, line_nums in numbers.items():
                    all_numbers[num_str][str(cuh_file.relative_to(directory))].extend(line_nums)
        except Exception as e:
            print(f"Error scanning {cuh_file}: {e}")

    return all_numbers

def main():
    project_dir = Path(__file__).parent.parent

    print("=== Magic Number Detection ===\n")
    print("Scanning for hardcoded numeric literals...\n")

    all_numbers = scan_directory(project_dir)

    sorted_by_frequency = sorted(all_numbers.items(), key=lambda x: sum(len(lines) for lines in x[1].values()), reverse=True)

    print(f"Found {len(sorted_by_frequency)} unique magic numbers\n")

    for num_str, files in sorted_by_frequency[:20]:
        total_occurrences = sum(len(lines) for lines in files.values())
        print(f"{num_str} ({total_occurrences} occurrences):")
        for file_path, line_nums in sorted(files.items()):
            print(f"  {file_path}: lines {line_nums[:5]}" + (" ..." if len(line_nums) > 5 else ""))

    if len(sorted_by_frequency) > 20:
        print(f"\n... and {len(sorted_by_frequency) - 20} more")

if __name__ == "__main__":
    main()
