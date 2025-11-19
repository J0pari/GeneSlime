#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_KERNELS = [
    "evaluate_organism_kernel",
    "evaluate_population_batch_kernel",
    "compute_affinity_map_kernel",
    "compute_flow_field_kernel",
    "reintegration_tracking_kernel",
    "assemble_population_ca_weights_kernel",
    "archive_insertion_kernel",
    "batch_archive_insertion_kernel",
    "select_parents_hybrid_kernel",
    "reproduce_population_kernel",
    "create_offspring_kernel",
    "compute_fitness_kernel",
    "compute_population_metrics_kernel",
    "update_regulatory_state_kernel",
    "init_curand_states",
    "compute_total_mass_kernel",
]

def find_files_by_extension(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return files

def check_ptx_files(ptx_dir):
    print(f"\n[PTX] Checking {ptx_dir}...")
    ptx_files = find_files_by_extension(ptx_dir, [".ptx", ".ii", ".cudafe1.cpp"])

    found_kernels = set()
    for ptx_file in ptx_files:
        try:
            with open(ptx_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for kernel in EXPECTED_KERNELS:
                    if kernel in content:
                        found_kernels.add(kernel)
        except Exception as e:
            print(f"  Error reading {ptx_file}: {e}")

    print(f"  Found {len(found_kernels)}/{len(EXPECTED_KERNELS)} kernels in PTX/intermediate files")
    return found_kernels

def check_executable(exe_path):
    print(f"\n[EXE] Checking {exe_path}...")
    if not os.path.exists(exe_path):
        print(f"  Executable not found: {exe_path}")
        return set()

    try:
        result = subprocess.run(
            ["cuobjdump", "--list-elf", exe_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        found_kernels = set()
        for kernel in EXPECTED_KERNELS:
            if kernel in result.stdout:
                found_kernels.add(kernel)

        print(f"  Found {len(found_kernels)}/{len(EXPECTED_KERNELS)} kernels in executable")
        return found_kernels
    except FileNotFoundError:
        print("  cuobjdump not found - skipping executable check")
        return set()
    except Exception as e:
        print(f"  Error checking executable: {e}")
        return set()

def main():
    script_dir = Path(__file__).parent.parent
    ptx_dir = script_dir / "build"
    exe_path = script_dir / "geneslime.exe"

    if len(sys.argv) > 1:
        ptx_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        exe_path = Path(sys.argv[2])

    print("=== Kernel Verification ===")

    ptx_kernels = check_ptx_files(ptx_dir) if ptx_dir.exists() else set()
    exe_kernels = check_executable(exe_path) if exe_path.exists() else set()

    all_found = ptx_kernels | exe_kernels
    missing = set(EXPECTED_KERNELS) - all_found

    print(f"\n=== Summary ===")
    print(f"Total expected: {len(EXPECTED_KERNELS)}")
    print(f"Found: {len(all_found)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print(f"\nMissing kernels:")
        for kernel in sorted(missing):
            print(f"  - {kernel}")
        return 1
    else:
        print("\nAll kernels found!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
