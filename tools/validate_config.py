#!/usr/bin/env python3
import re
from pathlib import Path

REQUIRED_CONSTANTS = {
    "GRID_SIZE": "int",
    "CHANNELS": "int",
    "NUM_HEADS": "int",
    "NUM_SEGMENTS": "int",
    "SEGMENT_PAYLOAD_SIZE": "int",
    "BEHAVIOR_DIM": "int",
    "BLOCK_SIZE_1D": "int",
    "BLOCK_SIZE_2D": "int",
    "WARP_SIZE": "int",
    "MAX_ARCHIVE_CELLS": "int",
    "MAX_POPULATION_SIZE": "int",
    "MAX_PATTERNS": "int",
    "MAX_WRITE_EVENTS": "int",
    "MAX_TIMESTEPS": "int",
}

REQUIRED_RELATIONSHIPS = [
    ("CHANNELS % NUM_HEADS == 0", "CHANNELS must be divisible by NUM_HEADS"),
    ("GRID_SIZE % BLOCK_SIZE_2D == 0", "GRID_SIZE must be divisible by BLOCK_SIZE_2D"),
    ("SEGMENT_PAYLOAD_SIZE >= 64", "SEGMENT_PAYLOAD_SIZE minimum is 64"),
    ("BEHAVIOR_DIM >= 2", "BEHAVIOR_DIM minimum is 2"),
]

def parse_constants(file_path):
    constants = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'constexpr\s+(\w+)\s+(\w+)\s*=\s*([^;]+);', line)
            if match:
                type_name, const_name, value = match.groups()
                try:
                    if 'float' in type_name or '.' in value or 'f' in value:
                        constants[const_name] = float(value.rstrip('f'))
                    else:
                        constants[const_name] = int(value)
                except:
                    pass
    return constants

def validate_schema(config_path):
    constants = parse_constants(config_path)

    print("=== Config Schema Validation ===\n")

    missing = []
    for name, expected_type in REQUIRED_CONSTANTS.items():
        if name not in constants:
            missing.append(name)

    if missing:
        print(f"[FAIL] Missing {len(missing)} required constants:")
        for name in missing:
            print(f"  - {name}")
        return False
    else:
        print(f"[PASS] All {len(REQUIRED_CONSTANTS)} required constants defined")

    print(f"\n=== Relationship Checks ===\n")

    failed = []
    for expr, description in REQUIRED_RELATIONSHIPS:
        try:
            result = eval(expr, {}, constants)
            if result:
                print(f"[PASS] {description}")
            else:
                print(f"[FAIL] {description}")
                failed.append(description)
        except Exception as e:
            print(f"[FAIL] {description} (evaluation failed: {e})")
            failed.append(description)

    if failed:
        print(f"\n[FAIL] {len(failed)} relationship checks failed")
        return False

    print("\n=== Validation Complete ===")
    return True

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "constants.cuh"
    success = validate_schema(config_path)
    exit(0 if success else 1)
