module BQMSolvers

export clarabel, copt, cplex, cupdlpx, cuopt, gurobi, highs, xpress

using LinearAlgebra: Symmetric
using QuadraticModels
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

# Materialize column `i` of `m` (or replicate shared `v`) as a fresh `Vector{T}`.
# QuadraticModels.jl's `QPData` calls `similar(c)` internally and expects the
# result's type to match `c`'s — that breaks for `SubArray{Matrix}` (similar
# returns Vector), so we allocate a plain Vector instead. One Vector per
# (c, lvar, uvar, lcon, ucon) = O(nvar + ncon) per instance — tiny next to a
# solve, and the big A/H matrices are still shared.
_col_vec(::Type{T}, m::AbstractMatrix, i::Int) where {T} = Vector{T}(@view m[:, i])
_col_vec(::Type{T}, v::AbstractVector, ::Int) where {T}  = Vector{T}(v)

_maybe_scalar(v::AbstractVector, i::Int) = v[i]
_maybe_scalar(x::Number, ::Int)          = x

function _scalar_qp_view(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}, i::Int,
                          A::AbstractMatrix, H::AbstractMatrix) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    c    = _col_vec(T, bqp.c_batch,    i)
    c0   = T(_maybe_scalar(bqp.c0_batch, i))
    lvar = _col_vec(T, bqp.meta.lvar,  i)
    uvar = _col_vec(T, bqp.meta.uvar,  i)
    lcon = _col_vec(T, bqp.meta.lcon,  i)
    ucon = _col_vec(T, bqp.meta.ucon,  i)
    return QuadraticModels.QuadraticModel(
        c, H; A = A,
        lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar, c0 = c0,
    )
end

# Shared A/H matrices for an ObjRHS batch. `bqp.A` wraps a SparseMatrixCSC
# (non-symmetric); `bqp.Q` wraps `Symmetric(SparseMatrixCSC, :L)`, which
# QuadraticModel's constructor handles directly.
_shared_A(bqp) = BatchQuadraticModels.operator_sparse_matrix(bqp.A)
_shared_H(bqp) = bqp.Q.op

# Per-instance A / H for a Uniform batch. `HostBatchSparseOperator.nzvals`
# is `(nnz_expanded, nbatch)` — the first `nnz(rows)` rows of each column
# are the original lower-triangular values (matching `(op.rows, op.cols)`),
# and any extra rows are symmetric-scatter duplicates we drop.
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

# For Q we wrap in Symmetric(:L) so the QuadraticModel ctor recognises it.
_instance_H(bqp::BatchQuadraticModels.BatchQuadraticModel, ::Int,
             ::BatchQuadraticModels.AbstractSparseOperator) = bqp.Q.op
function _instance_H(bqp::BatchQuadraticModels.BatchQuadraticModel, i::Int,
                      op::BatchQuadraticModels.HostBatchSparseOperator)
    n = bqp.meta.nvar
    return Symmetric(_instance_matrix(op, i, n, n), :L)
end

"""
    extract_instance(bqp, i)

Return a scalar `QuadraticModels.QuadraticModel` representing the `i`-th
instance of a `BatchQuadraticModel`. `ObjRHSBatchQuadraticModel` shares A/H
pointers across every extracted instance (zero-alloc for the matrices);
`UniformBatchQuadraticModel` rebuilds a per-instance `SparseMatrixCSC` from
the batch `nzvals` column.
"""
function extract_instance(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}, i::Int) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    1 <= i <= bqp.meta.nbatch || throw(BoundsError(bqp, i))
    return _scalar_qp_view(bqp, i, _instance_A(bqp, i, bqp.A), _instance_H(bqp, i, bqp.Q))
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
