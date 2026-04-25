using LinearAlgebra
using BatchQuadraticModels
using SolverCore
using SparseArrays
using SparseMatricesCOO

# Any scalar LP/QP model from BatchQuadraticModels that the solver extensions
# can consume uniformly. `LinearModel` carries an empty Hessian.
const BQMScalar = Union{BatchQuadraticModels.QuadraticModel, BatchQuadraticModels.LinearModel}

# COO triplets for A. `qp.data.A` is a `SparseOperator` wrapping a
# `SparseMatrixCSC`; `findnz` gives us row/col/val vectors.
function _A_coo(qp::BQMScalar)
    A = BatchQuadraticModels.operator_sparse_matrix(qp.data.A)
    return findnz(A)
end

# COO triplets for the lower-triangular Hessian. `qp.data.Q` wraps
# `Symmetric(Q_csc, :L)`; the CSC underneath is already lower-triangular.
function _Q_coo(qp::BatchQuadraticModels.QuadraticModel{T}) where {T}
    Q = BatchQuadraticModels.operator_sparse_matrix(qp.data.Q)
    return findnz(Q)
end
_Q_coo(qp::BatchQuadraticModels.LinearModel{T}) where {T} =
    (Int[], Int[], T[])

# Hessian nnz — `LinearModel` has no Q field, so the fallback is 0.
_nnzh(qp::BatchQuadraticModels.QuadraticModel) = qp.meta.nnzh
_nnzh(::BatchQuadraticModels.LinearModel)      = 0

function _sparse_csr(
    I,
    J,
    V,
    m = isempty(I) ? 0 : maximum(I),
    n = isempty(J) ? 0 : maximum(J),
)
    csrrowptr = zeros(Int, m + 1)
    coolen = length(I)
    min(length(J), length(V)) >= coolen ||
        throw(ArgumentError("J and V need length >= length(I) = $coolen"))
    @inbounds for k in 1:coolen
        Ik = I[k]
        1 <= Ik <= m || throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
        csrrowptr[Ik + 1] += 1
    end
    countsum = 1
    csrrowptr[1] = 1
    @inbounds for i in 2:(m + 1)
        overwritten = csrrowptr[i]
        csrrowptr[i] = countsum
        countsum += overwritten
    end
    csrcolval = zeros(Int, length(I))
    csrnzval = zeros(eltype(V), length(I))
    @inbounds for k in 1:coolen
        Ik, Jk = I[k], J[k]
        1 <= Jk <= n || throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
        csrk = csrrowptr[Ik + 1]
        csrrowptr[Ik + 1] = csrk + 1
        csrcolval[csrk] = Jk
        csrnzval[csrk] = V[k]
    end
    return csrrowptr[1:end-1], csrcolval, csrnzval
end

function _row_sense_rhs_range(l::Real, u::Real, plus_inf::Real)
    if isfinite(l) && isfinite(u)
        if l == u
            return Cchar('E'), float(u), 0.0
        else
            return Cchar('R'), float(u), float(u - l)
        end
    elseif isfinite(l)
        return Cchar('G'), float(l), 0.0
    elseif isfinite(u)
        return Cchar('L'), float(u), 0.0
    else
        return Cchar('N'), float(plus_inf), 0.0
    end
end

function _status_symbol(value, mapping)
    return get(mapping, value, :unknown)
end
