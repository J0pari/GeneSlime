#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict

KNOWN_KEYWORDS = {
    'int', 'float', 'double', 'bool', 'void', 'char', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'size_t', 'const', 'static', 'inline', 'constexpr', 'auto', 'extern', 'struct', 'class',
    'if', 'else', 'for', 'while', 'do', 'return', 'switch', 'case', 'break', 'continue',
    'dim3', 'cudaError_t', 'cudaStream_t', 'curandState', 'half',
    '__device__', '__host__', '__global__', '__forceinline__', '__shared__',
    'blockIdx', 'blockDim', 'threadIdx', 'gridDim', 'warpSize',
}

def extract_defined_symbols(config_path):
    defined = set()

    try:
        with open(config_path, 'r') as f:
            for line in f:
                match = re.search(r'constexpr\s+\w+\s+(\w+)', line)
                if match:
                    defined.add(match.group(1))

                match = re.search(r'enum\s+\w+.*?\{([^}]+)\}', line)
                if match:
                    for enum_val in match.group(1).split(','):
                        name = enum_val.split('=')[0].strip()
                        if name:
                            defined.add(name)
    except Exception as e:
        print(f"Error reading {config_path}: {e}")

    return defined

def extract_used_symbols(file_path):
    used = set()

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            for match in re.finditer(r'\b[A-Z_][A-Z0-9_]{2,}\b', content):
                symbol = match.group()
                if symbol not in KNOWN_KEYWORDS:
                    used.add(symbol)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return used

def find_symbol_locations(file_path, symbols):
    locations = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for symbol in symbols:
                    if re.search(rf'\b{symbol}\b', line):
                        locations[symbol].append(line_num)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return locations

def main():
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / "config" / "constants.cuh"

    print("=== Undefined Symbol Detection ===\n")

    defined_symbols = extract_defined_symbols(config_path)
    print(f"Found {len(defined_symbols)} defined symbols in constants.cuh\n")

    all_used = set()
    file_usage = defaultdict(set)

    for cu_file in project_dir.rglob("*.cu"):
        if "test" in str(cu_file):
            continue

        used = extract_used_symbols(cu_file)
        all_used.update(used)

        for symbol in used:
            file_usage[symbol].add(cu_file)

    for cuh_file in project_dir.rglob("*.cuh"):
        if "constants.cuh" in str(cuh_file) or "test" in str(cuh_file):
            continue

        used = extract_used_symbols(cuh_file)
        all_used.update(used)

        for symbol in used:
            file_usage[symbol].add(cuh_file)

    undefined = all_used - defined_symbols

    print(f"Used symbols: {len(all_used)}")
    print(f"Undefined symbols: {len(undefined)}\n")

    if undefined:
        print("Undefined symbols by file:")
        for symbol in sorted(undefined):
            print(f"\n{symbol}:")
            for file_path in sorted(file_usage[symbol]):
                rel_path = file_path.relative_to(project_dir)
                locations = find_symbol_locations(file_path, {symbol})
                if locations[symbol]:
                    print(f"  {rel_path}: lines {locations[symbol][:5]}" + (" ..." if len(locations[symbol]) > 5 else ""))

        print(f"\nSuggested additions to constants.cuh:")
        for symbol in sorted(undefined):
            print(f"constexpr int {symbol} = ???;")
    else:
        print("No undefined symbols found!")

if __name__ == "__main__":
    main()
