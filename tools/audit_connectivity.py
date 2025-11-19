#!/usr/bin/env python3
"""
CUDA Call Graph Analyzer - AST-based reachability analysis

Uses libclang to parse CUDA/C++ code into proper AST, then computes
transitive closure of call graph via Kleisli composition.

Architecture:
- Include graph: files -> files (comonad extract)
- Call graph: functions -> functions (Kleisli arrows)
- Reachability: transitive closure starting from main()
"""

import sys
from pathlib import Path
from collections import defaultdict

try:
    import clang.cindex as clang
except ImportError:
    print("ERROR: clang Python bindings not installed", file=sys.stderr)
    print("Install with: pip install libclang", file=sys.stderr)
    sys.exit(1)


class CUDACallGraph:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)

        # Configure libclang to use full LLVM installation
        llvm_path = r'C:\Program Files\LLVM'
        clang.Config.set_library_file(str(Path(llvm_path) / 'bin' / 'libclang.dll'))

        self.index = clang.Index.create()

        # Function registry: name -> FunctionInfo
        self.functions = {}

        # Call graph: function_name -> set(called_function_names)
        self.call_graph = defaultdict(set)

        # Include graph: file_path -> set(included_file_paths)
        self.includes = defaultdict(set)

        # File -> functions defined in file
        self.file_functions = defaultdict(set)

    def parse_cuda_file(self, file_path):
        """Parse single CUDA file using libclang"""
        rel_path = str(file_path.relative_to(self.project_dir))

        # Parse with CUDA-aware flags
        cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
        llvm_path = r'C:\Program Files\LLVM'
        msvc_path = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
        ucrt_path = r'C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt'
        um_path = r'C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um'
        shared_path = r'C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared'

        try:
            tu = self.index.parse(
                str(file_path),
                args=[
                    '-std=c++17',
                    '-xc++',  # Parse as C++ not CUDA to avoid wrapper issues
                    '-I' + str(self.project_dir),
                    '-I' + str(self.project_dir / 'src'),
                    f'-I{cuda_path}\\include',
                    f'-I{msvc_path}\\include',
                    f'-I{ucrt_path}',
                    f'-I{um_path}',
                    f'-I{shared_path}',
                    # Define CUDA qualifiers as empty to make syntax valid C++
                    '-D__global__=',
                    '-D__device__=',
                    '-D__host__=',
                    '-D__shared__=__attribute__((annotate("shared")))',
                    '-D__constant__=__attribute__((annotate("constant")))',
                    '-D__managed__=__attribute__((annotate("managed")))',
                    '-DCUDA_ARCH=800',
                    '-D__CUDACC__',
                    # Include standard headers
                    '-include', 'stdint.h',
                    '-include', 'stddef.h',
                    '-fms-compatibility',
                    '-fms-extensions',
                    '-Wno-unknown-attributes',
                    '-Wno-ignored-attributes',
                    '-ferror-limit=100'
                ],
                options=clang.TranslationUnit.PARSE_INCOMPLETE
            )
        except Exception as e:
            print(f"Warning: Exception parsing {rel_path}: {e}", file=sys.stderr)
            return

        if not tu:
            print(f"Warning: Failed to create TU for {rel_path}", file=sys.stderr)
            return

        if tu.diagnostics:
            errors = [d for d in tu.diagnostics if d.severity >= clang.Diagnostic.Error]
            if errors:
                print(f"Warning: Parse errors in {rel_path}:", file=sys.stderr)
                for err in errors[:3]:
                    print(f"  {err}", file=sys.stderr)
                # Continue anyway - partial AST may still be useful

        # Walk AST to extract functions and includes
        self._extract_from_ast(tu.cursor, rel_path)

    def _extract_from_ast(self, cursor, current_file):
        """Recursively extract functions, calls, includes from AST"""

        # Extract includes
        if cursor.kind == clang.CursorKind.INCLUSION_DIRECTIVE:
            included_file = cursor.get_included_file()
            if included_file:
                inc_path = Path(included_file.name)
                try:
                    rel_inc = str(inc_path.relative_to(self.project_dir))
                    self.includes[current_file].add(rel_inc)
                except ValueError:
                    pass  # External include

        # Extract function definitions
        elif cursor.kind == clang.CursorKind.FUNCTION_DECL and cursor.is_definition():
            func_name = cursor.spelling

            # Determine CUDA qualifiers by reading raw source
            # (preprocessor strips them due to -D__global__= etc)
            is_global = False
            is_device = False

            extent = cursor.extent
            if extent.start.file:
                try:
                    with open(extent.start.file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        # Read a few lines before function to catch qualifiers
                        start_line = max(0, extent.start.line - 3)
                        func_decl_text = ''.join(lines[start_line:extent.start.line + 2])

                        is_global = '__global__' in func_decl_text
                        is_device = '__device__' in func_decl_text
                except:
                    pass

            is_host = not is_global and not is_device

            source_file = cursor.location.file
            if source_file:
                try:
                    file_path = str(Path(source_file.name).relative_to(self.project_dir))
                except ValueError:
                    file_path = current_file
            else:
                file_path = current_file

            self.functions[func_name] = {
                'file': file_path,
                'is_global': is_global,
                'is_device': is_device,
                'is_host': is_host,
                'calls': set(),
                'launches': set()
            }

            self.file_functions[file_path].add(func_name)

            # Extract calls from function body
            self._extract_calls_from_function(cursor, func_name)

        # Recurse to children
        for child in cursor.get_children():
            self._extract_from_ast(child, current_file)

    def _extract_calls_from_function(self, func_cursor, caller_name):
        """Extract function calls and kernel launches from function body"""

        for node in func_cursor.walk_preorder():
            # Regular function calls
            if node.kind == clang.CursorKind.CALL_EXPR:
                callee = node.referenced
                if callee and callee.kind == clang.CursorKind.FUNCTION_DECL:
                    callee_name = callee.spelling
                    self.call_graph[caller_name].add(callee_name)
                    self.functions[caller_name]['calls'].add(callee_name)

            # CUDA kernel launches (<<<>>>)
            # In CUDA mode these appear as CUDA_KERNEL_CALL_EXPR
            # In C++ mode with stripped qualifiers, detect via token pattern
            elif hasattr(clang.CursorKind, 'CUDA_KERNEL_CALL_EXPR') and node.kind == clang.CursorKind.CUDA_KERNEL_CALL_EXPR:
                children = list(node.get_children())
                if children:
                    kernel = children[0]
                    if kernel.kind == clang.CursorKind.DECL_REF_EXPR:
                        kernel_name = kernel.spelling
                        self.call_graph[caller_name].add(kernel_name)
                        self.functions[caller_name]['launches'].add(kernel_name)

        # Fallback: parse function source text for <<<>>> patterns
        extent = func_cursor.extent
        if extent.start.file:
            try:
                with open(extent.start.file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    func_text = ''.join(lines[extent.start.line-1:extent.end.line])

                    # Find kernel launches via regex
                    import re
                    for match in re.finditer(r'(\w+)\s*<<<[\s\S]*?>>>', func_text):
                        kernel_name = match.group(1)
                        self.call_graph[caller_name].add(kernel_name)
                        self.functions[caller_name]['launches'].add(kernel_name)
            except:
                pass

    def build_graph(self):
        """Parse all CUDA files in project"""
        all_files = list(self.project_dir.rglob("*.cu")) + list(self.project_dir.rglob("*.cuh"))

        for file_path in all_files:
            try:
                self.parse_cuda_file(file_path)
            except Exception as e:
                print(f"Warning: Error parsing {file_path}: {e}", file=sys.stderr)

    def find_reachable_from_main(self):
        """
        Compute transitive closure of call graph starting from main()

        Returns (reachable_files, reachable_functions)
        """

        # Find main() entry point
        if 'main' not in self.functions:
            return set(), set()

        main_func = self.functions['main']
        entry_calls = main_func['calls'] | main_func['launches']

        # BFS through call graph (Kleisli composition)
        reachable_functions = set(['main'])
        queue = list(entry_calls)

        while queue:
            func_name = queue.pop(0)
            if func_name in reachable_functions:
                continue

            reachable_functions.add(func_name)

            # Add transitive callees
            if func_name in self.functions:
                callees = self.functions[func_name]['calls'] | self.functions[func_name]['launches']
                for callee in callees:
                    if callee not in reachable_functions:
                        queue.append(callee)

        # Derive reachable files from reachable functions
        reachable_files = set()
        for func_name in reachable_functions:
            if func_name in self.functions:
                reachable_files.add(self.functions[func_name]['file'])

        # Add files reachable via includes
        files_queue = list(reachable_files)
        visited_files = set()

        while files_queue:
            file_path = files_queue.pop(0)
            if file_path in visited_files:
                continue
            visited_files.add(file_path)

            for inc in self.includes.get(file_path, []):
                if inc not in visited_files:
                    files_queue.append(inc)
                    reachable_files.add(inc)

        return reachable_files, reachable_functions

    def find_unreachable_kernels(self):
        """Find __global__ kernels not reachable from main()"""
        _, reachable = self.find_reachable_from_main()

        unreachable = {}
        for name, info in self.functions.items():
            if info['is_global'] and name not in reachable:
                unreachable[name] = info

        return unreachable

    def find_unreachable_device_functions(self):
        """Find __device__ functions not reachable from main()"""
        _, reachable = self.find_reachable_from_main()

        unreachable = {}
        for name, info in self.functions.items():
            if info['is_device'] and not info['is_global'] and name not in reachable:
                unreachable[name] = info

        return unreachable

    def analyze_coverage(self):
        """Compute execution coverage metrics"""
        total_kernels = sum(1 for f in self.functions.values() if f['is_global'])
        total_device = sum(1 for f in self.functions.values() if f['is_device'] and not f['is_global'])

        _, reachable = self.find_reachable_from_main()

        reachable_kernels = sum(
            1 for name, info in self.functions.items()
            if info['is_global'] and name in reachable
        )

        reachable_device = sum(
            1 for name, info in self.functions.items()
            if info['is_device'] and not info['is_global'] and name in reachable
        )

        return {
            'total_kernels': total_kernels,
            'reachable_kernels': reachable_kernels,
            'kernel_coverage': reachable_kernels / total_kernels if total_kernels > 0 else 0,
            'total_device': total_device,
            'reachable_device': reachable_device,
            'device_coverage': reachable_device / total_device if total_device > 0 else 0
        }

    def generate_report(self):
        """Generate comprehensive connectivity audit report"""
        print("=== CUDA Call Graph Analysis (AST-based) ===\n")

        total_funcs = len(self.functions)
        total_kernels = sum(1 for f in self.functions.values() if f['is_global'])
        total_device = sum(1 for f in self.functions.values() if f['is_device'] and not f['is_global'])
        total_host = sum(1 for f in self.functions.values() if f['is_host'])

        print(f"Functions parsed: {total_funcs}")
        print(f"  __global__ kernels: {total_kernels}")
        print(f"  __device__ functions: {total_device}")
        print(f"  __host__ functions: {total_host}\n")

        # Coverage analysis
        coverage = self.analyze_coverage()
        print("=== Execution Coverage ===\n")
        print(f"__global__ kernels: {coverage['reachable_kernels']}/{coverage['total_kernels']} ({coverage['kernel_coverage']*100:.1f}%)")
        print(f"__device__ functions: {coverage['reachable_device']}/{coverage['total_device']} ({coverage['device_coverage']*100:.1f}%)\n")

        if coverage['kernel_coverage'] < 0.5:
            print("[CRITICAL] Less than 50% of kernels reachable from main()")
        elif coverage['kernel_coverage'] < 0.8:
            print("[WARNING] Less than 80% of kernels reachable from main()")
        else:
            print("[PASS] Good kernel coverage")
        print()

        # Unreachable kernels
        unreachable_kernels = self.find_unreachable_kernels()
        if unreachable_kernels:
            print(f"=== Unreachable __global__ Kernels ({len(unreachable_kernels)}) ===\n")

            by_file = defaultdict(list)
            for name, info in unreachable_kernels.items():
                by_file[info['file']].append(name)

            for file_path in sorted(by_file.keys()):
                print(f"{file_path}:")
                for func_name in sorted(by_file[file_path]):
                    print(f"  - {func_name}")
            print()

        # Unreachable device functions (summarized)
        unreachable_device = self.find_unreachable_device_functions()
        if unreachable_device:
            print(f"=== Unreachable __device__ Functions ({len(unreachable_device)}) ===")
            print(f"(Device functions called only from unreachable kernels)\n")

            by_file = defaultdict(list)
            for name, info in unreachable_device.items():
                by_file[info['file']].append(name)

            for file_path in sorted(by_file.keys())[:5]:
                print(f"{file_path}: {len(by_file[file_path])} functions")

            if len(by_file) > 5:
                print(f"... and {len(by_file) - 5} more files")
            print()

        # Summary
        print("=== Summary ===\n")
        unreachable_count = len(unreachable_kernels)

        if unreachable_count == 0:
            print("[PASS] All kernels are reachable from main()")
            return True
        else:
            print(f"[FAIL] {unreachable_count} unreachable kernels")
            print(f"\nRecommendation: Wire unreachable kernels into execution pipeline")
            return False


def main():
    project_dir = Path(__file__).parent.parent

    graph = CUDACallGraph(project_dir)
    graph.build_graph()
    success = graph.generate_report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
