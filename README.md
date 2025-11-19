# GeneSlime

Evolving Flow-Lenia cellular automata with context-dependent genome assembly on CUDA.

## Architecture

**Flow-Lenia CA** (arXiv:2506.08569v1): Mass-conservative continuous cellular automata with affinity maps, flow fields, and reintegration tracking. Multi-head attention structure implemented with WMMA tensor cores.

**Segmented Genomes**: 16 segments × 64 weights = 1024 parameters. Segments have permission levels (L0-L3) controlling mutation rates and structural modifications. Three stochastic variants per segment selected based on execution context.

**MAP-Elites Archive**: Quality-diversity optimization with Voronoi tessellation in behavioral space. Uses learned DIRESA embeddings (reconstruction + triplet loss) rather than hand-crafted features. SVD compression for reference genomes, delta encoding for behavioral neighbors.

**Stigmergic Substrate**: Four-layer memory with different decay rates (fast/medium/slow/structural). Threshold-gated writes allow organisms to modify environment for future generations.

**Crisis Detection**: Population-level metrics trigger structural mutation modes when stagnation detected.

## Implementation

```
core/           Flow-Lenia simulation, genome assembly, organism evaluation
memory/         Archive compression, stigmergic field, parallel compaction
learning/       Autodiff tape, DIRESA encoder/decoder, training loop
lifecycle/      Parent selection, mutation operators, regulatory control
metrics/        Fitness functions, population statistics
kernels/        SVD (Jacobi), tensor cores (WMMA), stream compaction
debug/          Kernel launch tracing, parameter validation
```

Dynamic parallelism for hierarchical kernel launches (organism evaluation spawns CA simulation). Requires `-rdc=true` and `cudadevrt` linking.

## Build

```bash
cd build_scripts
build.bat
```

Targets sm_86 (Ampere/Ada). Requires CUDA 11.8+.

## Tools

```bash
python tools/validate_config.py         # Verify constants.cuh relationships
python tools/find_magic_numbers.py      # Locate hardcoded literals
python tools/find_undefined_symbols.py  # Find missing constant definitions
```

## References

Flow-Lenia: Edalat, Chan, Katumba, Channon (2024) arXiv:2506.08569v1
MAP-Elites: Mouret & Clune (2015)
DIRESA behavioral embeddings adapted from archive compression techniques
