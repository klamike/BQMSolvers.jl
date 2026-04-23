# BQMSolvers

`BQMSolvers.jl` wraps popular LP/QP solvers behind a uniform
`QuadraticModel` interface, and adds a threaded batch path on top of
[`BatchQuadraticModels.jl`](https://github.com/klamike/BatchQuadraticModels.jl).

Supported solvers (one function each, loaded via package extensions):

- `clarabel`
- `copt`
- `cplex`
- `cuopt`
- `cupdlpx`
- `gurobi`
- `highs`
- `xpress`

Each function is defined only when the corresponding solver package is
`using`'d.

## Scalar

```julia
using QuadraticModels, BQMSolvers, HiGHS

qp = QuadraticModel(
    [1.0, 2.0],
    [1, 2], [1, 2], [4.0, 2.0];
    Arows = [1, 1], Acols = [1, 2], Avals = [1.0, 1.0],
    lcon = [1.0], ucon = [1.0],
    lvar = [0.0, 0.0], uvar = [Inf, Inf],
)
stats = highs(qp)
```

## Batch

For a `BatchQuadraticModels.BatchQuadraticModel` — `ObjRHSBatchQuadraticModel`
(shared A/Q, per-instance c/bounds) or `UniformBatchQuadraticModel`
(per-instance A/Q nzvals) — call any supported solver with the batch
model and it dispatches to `solve_batch_threaded`, which runs one scalar
solve per instance across `Threads.nthreads()`:

```julia
using BatchQuadraticModels, BQMSolvers, HiGHS

bqp = BatchQuadraticModels.ObjRHSBatchQuadraticModel(qp, 64; c = c_batch)
stats = highs(bqp; threads = 1)   # returns Vector of per-instance stats
```

Thread safety / performance:

- Each task extracts its instance as a fresh scalar `QuadraticModel` and
  calls the solver independently — tasks share no mutable state.
- For `ObjRHSBatchQuadraticModel`, A and H pointers are shared across
  every extracted instance (zero-alloc for the big matrices). Only the
  per-instance `c` and bound vectors are materialized (O(nvar + ncon)).
- For `UniformBatchQuadraticModel`, each instance's `SparseMatrixCSC` is
  rebuilt from the batch `nzvals` column (unavoidable — per-instance
  numeric values).
- Pass the solver's thread-limit option (`threads = 1`, `Threads = 1`,
  `"Threads" => 1`, etc.) so the inner solver stays single-threaded —
  otherwise outer Julia threads fight inner solver threads for CPUs.
  Set `JULIA_NUM_THREADS=N` to match your intended outer parallelism and
  `LinearAlgebra.BLAS.set_num_threads(1)`.
- The default schedule is `:static`; pass `schedule = :dynamic` via the
  solver call when per-instance solve times are very uneven.

To use the underlying helpers directly:

```julia
qp_i = BQMSolvers.extract_instance(bqp, i)             # single instance
stats = BQMSolvers.solve_batch_threaded(highs, bqp; threads = 1)
```
