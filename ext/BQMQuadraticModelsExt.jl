# Compatibility extension: lets users still pass a
# `QuadraticModels.QuadraticModel` (from the QuadraticModels.jl registry) to
# any BQMSolvers entry point. We convert to BQM's `QuadraticModel` and
# dispatch through the scalar methods defined in each solver extension.

module BQMQuadraticModelsExt

using BQMSolvers
using BatchQuadraticModels
using QuadraticModels
using SparseArrays
using SparseMatricesCOO

# Build a `BatchQuadraticModels.QuadraticModel` mirroring `qm`. COO inputs
# stay COO; dense / operator A/H are converted through `SparseMatrixCOO` so
# `sparse_operator` inside BQM can wrap them.
function _to_bqm(qm::QuadraticModels.QuadraticModel{T, S}) where {T, S}
    meta = qm.meta
    A = _to_sparse(qm.data.A, meta.ncon, meta.nvar)
    H = _to_sparse(qm.data.H, meta.nvar, meta.nvar)
    data = BatchQuadraticModels.QPData(A, qm.data.c, H;
        lcon = Vector{T}(meta.lcon), ucon = Vector{T}(meta.ucon),
        lvar = Vector{T}(meta.lvar), uvar = Vector{T}(meta.uvar),
        c0 = T(qm.data.c0),
    )
    return BatchQuadraticModels.QuadraticModel(data;
        x0 = Vector{T}(meta.x0), y0 = Vector{T}(meta.y0),
        minimize = meta.minimize, name = meta.name,
    )
end

_to_sparse(A::SparseMatrixCSC, ::Int, ::Int) = A
_to_sparse(A::SparseMatrixCOO, m::Int, n::Int) = sparse(A.rows, A.cols, A.vals, m, n)
_to_sparse(A::AbstractMatrix, m::Int, n::Int) = sparse(A)

for fn in (:clarabel, :copt, :cplex, :cuopt, :cupdlpx, :gurobi, :highs, :xpress)
    @eval function BQMSolvers.$fn(qm::QuadraticModels.QuadraticModel; kwargs...)
        return BQMSolvers.$fn(_to_bqm(qm); kwargs...)
    end
end

end
