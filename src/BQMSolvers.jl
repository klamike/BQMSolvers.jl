module BQMSolvers

export clarabel, copt, cplex, cupdlpx, cuopt, gurobi, highs, xpress

using LinearAlgebra: Symmetric
using BatchQuadraticModels
using SparseArrays

# Scalar solver stubs. Actual implementations live in package extensions.
function gurobi(m, k...)   error("gurobi is not available; make sure to load the package first, e.g. `using Gurobi`") end
function xpress(m, k...)   error("xpress is not available; make sure to load the package first, e.g. `using Xpress`") end
function cplex(m, k...)    error("cplex is not available; make sure to load the package first, e.g. `using CPLEX`") end
function highs(m, k...)    error("highs is not available; make sure to load the package first, e.g. `using HiGHS`") end
function cuopt(m, k...)    error("cuopt is not available; make sure to load the package first, e.g. `using cuOpt`") end
function cupdlpx(m, k...)  error("cupdlpx is not available; make sure to load the package first, e.g. `using CuPDLPx`") end
function clarabel(m, k...) error("clarabel is not available; make sure to load the package first, e.g. `using Clarabel`") end
function copt(m, k...)     error("copt is not available; make sure to load the package first, e.g. `using COPT`") end

# ----------------------------------------------------------------------------
# Batch → scalar extraction.
#
# ObjRHS case: A and Q are shared across the batch (`SparseOperator`); only
# c/c0/bounds vary — we share A/H pointers with every instance's scalar QP.
#
# Uniform case: A and/or Q have per-instance nzvals stored expanded in
# `HostBatchSparseOperator.nzvals :: (nnz_expanded, nbatch)`. The first
# `length(op.rows)` entries of each column hold the original lower-triangle
# values matching `(op.rows, op.cols)`; remaining entries (if any) are
# scatter duplicates for symmetric Hessians. Per-instance extraction rebuilds
# a fresh `SparseMatrixCSC` per (instance, matrix), which is unavoidable —
# each instance has different numerical values.
# ----------------------------------------------------------------------------

# Per-instance c / bound vectors are materialized as fresh `Vector{T}` so the
# extracted scalar QP has a uniform vector type across its fields —
# `NLPModels.NLPModelMeta{T, S}` and `SolverCore.GenericExecutionStats` both
# insist on that uniformity, which defeats any attempt to hand through
# `SubArray` views of the batch matrices. O(nvar + ncon) per instance, cheap
# vs a solve; the big A/Q arrays still share pointers across the batch.
_col_view(m::AbstractMatrix, i::Int) = @view m[:, i]
_col_view(v::AbstractVector, ::Int)  = v

_maybe_scalar(v::AbstractVector, i::Int) = v[i]
_maybe_scalar(x::Number, ::Int)          = x

# Shared A / lower-triangular Q for an ObjRHS batch: just unwrap the
# SparseOperator once. Every extracted scalar QP shares the same underlying
# CSC arrays — zero alloc for the big matrices.
_shared_A(bqp) = BatchQuadraticModels.operator_sparse_matrix(bqp.A)
_shared_Q(bqp) = BatchQuadraticModels.operator_sparse_matrix(bqp.Q)

# Per-instance A / Q for a Uniform batch. `HostBatchSparseOperator.nzvals`
# is `(nnz_expanded, nbatch)` — the first `length(op.rows)` rows of each
# column hold the original lower-triangular values (matching op.rows,
# op.cols); remaining rows are symmetric-scatter duplicates we drop.
function _instance_matrix(op::BatchQuadraticModels.HostBatchSparseOperator, i::Int,
                           nrows::Int, ncols::Int)
    nnz_orig = length(op.rows)
    vals = @view op.nzvals[1:nnz_orig, i]
    return sparse(op.rows, op.cols, Vector(vals), nrows, ncols)
end

_instance_A(bqp::BatchQuadraticModels.BatchQuadraticModel, ::Int,
             ::BatchQuadraticModels.AbstractSparseOperator) =
    BatchQuadraticModels.operator_sparse_matrix(bqp.A)
_instance_A(bqp::BatchQuadraticModels.BatchQuadraticModel, i::Int,
             op::BatchQuadraticModels.HostBatchSparseOperator) =
    _instance_matrix(op, i, bqp.meta.ncon, bqp.meta.nvar)

_instance_Q(bqp::BatchQuadraticModels.BatchQuadraticModel, ::Int,
             ::BatchQuadraticModels.AbstractSparseOperator) =
    BatchQuadraticModels.operator_sparse_matrix(bqp.Q)
function _instance_Q(bqp::BatchQuadraticModels.BatchQuadraticModel, i::Int,
                      op::BatchQuadraticModels.HostBatchSparseOperator)
    n = bqp.meta.nvar
    return _instance_matrix(op, i, n, n)
end

function _scalar_qp_alloc(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}, i::Int,
                           A::AbstractMatrix, Q::AbstractMatrix) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    c    = Vector{T}(_col_view(bqp.c_batch, i))
    c0   = T(_maybe_scalar(bqp.c0_batch, i))
    lvar = Vector{T}(_col_view(bqp.meta.lvar, i))
    uvar = Vector{T}(_col_view(bqp.meta.uvar, i))
    lcon = Vector{T}(_col_view(bqp.meta.lcon, i))
    ucon = Vector{T}(_col_view(bqp.meta.ucon, i))
    data = BatchQuadraticModels.QPData(A, c, Q;
        lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, c0 = c0,
    )
    return BatchQuadraticModels.QuadraticModel(data)
end

"""
    extract_instance(bqp, i)

Return a scalar `BatchQuadraticModels.QuadraticModel` representing the `i`-th
instance of a `BatchQuadraticModel`. `ObjRHSBatchQuadraticModel` shares
underlying A/Q arrays across every extracted instance (zero alloc for the
big matrices); `UniformBatchQuadraticModel` rebuilds a per-instance
`SparseMatrixCSC` from the batch `nzvals` column.
"""
function extract_instance(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}, i::Int) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    1 <= i <= bqp.meta.nbatch || throw(BoundsError(bqp, i))
    return _scalar_qp_alloc(bqp, i, _instance_A(bqp, i, bqp.A), _instance_Q(bqp, i, bqp.Q))
end

"""
    solve_batch_threaded(solver_fn, bqp; schedule = :static, kwargs...)

Solve every instance of an `ObjRHSBatchQuadraticModel` by extracting each
instance as a scalar view and calling `solver_fn` in parallel across
`Threads.nthreads()` tasks. Returns a `Vector` of per-instance stats.

Thread-safety: each task works on a zero-alloc scalar view of the batch.
A/Q triplets are shared — the QuadraticModel constructor and the scalar
solvers don't mutate them. If the underlying `solver_fn` maintains a
global handle (e.g. Gurobi's `Env`), pass an env factory / per-call
option.

Performance: pass `threads=1` / `Threads=1` / similar in `kwargs` so the
inner solver is single-threaded — otherwise outer Julia threads and
inner solver threads will fight for CPUs. Set `JULIA_NUM_THREADS=N` to
match outer parallelism; `BLAS.set_num_threads(1)` inside the process
is also a good idea. `schedule = :dynamic` helps when per-instance
solve times are very uneven.
"""
function solve_batch_threaded(solver_fn, bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT};
                               schedule::Symbol = :static, kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    nbatch = bqp.meta.nbatch
    stats  = Vector{Any}(undef, nbatch)
    if schedule === :dynamic
        Threads.@threads :dynamic for i in 1:nbatch
            qp_i = extract_instance(bqp, i)
            stats[i] = solver_fn(qp_i; kwargs...)
        end
    else
        Threads.@threads for i in 1:nbatch
            qp_i = extract_instance(bqp, i)
            stats[i] = solver_fn(qp_i; kwargs...)
        end
    end
    return stats
end

end
