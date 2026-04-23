module BQMHiGHSExt

using HiGHS
using BQMSolvers

include("_common.jl")

const _highs_statuses = Dict(
    kHighsModelStatusOptimal => :acceptable,
    kHighsModelStatusInfeasible => :infeasible,
    kHighsModelStatusUnboundedOrInfeasible => :unbounded,
    kHighsModelStatusUnbounded => :unbounded,
    kHighsModelStatusTimeLimit => :max_time,
    kHighsModelStatusIterationLimit => :max_iter,
    kHighsModelStatusInterrupt => :user,
    kHighsModelStatusModelError => :exception,
    kHighsModelStatusPresolveError => :exception,
    kHighsModelStatusSolveError => :exception,
    kHighsModelStatusPostsolveError => :exception,
    kHighsModelStatusUnknown => :unknown,
)

function _highs_hessian(QM)
    I = Int[]
    J = Int[]
    V = Float64[]
    for k in eachindex(QM.data.H.vals)
        i = Int(QM.data.H.rows[k])
        j = Int(QM.data.H.cols[k])
        row, col = i > j ? (i, j) : (j, i)
        push!(I, row)
        push!(J, col)
        push!(V, Float64(QM.data.H.vals[k]))
    end
    Q = sparse(I, J, V, QM.meta.nvar, QM.meta.nvar)
    return Q
end

function BQMSolvers.highs(QM::QuadraticModel{T,S}; kwargs...) where {T,S}
    return BQMSolvers.highs(_coo_model(QM); kwargs...)
end

function BQMSolvers.highs(
    QM::QuadraticModel{T,S,M1,M2};
    kwargs...,
) where {T,S,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}
    length(QM.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    A = sparse(QM.data.A.rows, QM.data.A.cols, QM.data.A.vals, QM.meta.ncon, QM.meta.nvar)
    col_value = zeros(Float64, QM.meta.nvar)
    col_dual = zeros(Float64, QM.meta.nvar)
    row_value = zeros(Float64, QM.meta.ncon)
    row_dual = zeros(Float64, QM.meta.ncon)
    col_basis_status = zeros(HiGHS.HighsInt, QM.meta.nvar)
    row_basis_status = zeros(HiGHS.HighsInt, QM.meta.ncon)
    model_status = Ref{HiGHS.HighsInt}(kHighsModelStatusNotset)
    sense = QM.meta.minimize ? kHighsObjSenseMinimize : kHighsObjSenseMaximize
    timed = if QM.meta.nnzh > 0
        Q = _highs_hessian(QM)
        @timed Highs_qpCall(
            QM.meta.nvar,
            QM.meta.ncon,
            length(A.nzval),
            length(Q.nzval),
            kHighsMatrixFormatColwise,
            kHighsHessianFormatTriangular,
            sense,
            Float64(QM.data.c0),
            Float64.(QM.data.c),
            Float64.(QM.meta.lvar),
            Float64.(QM.meta.uvar),
            Float64.(QM.meta.lcon),
            Float64.(QM.meta.ucon),
            convert(Vector{HiGHS.HighsInt}, A.colptr[1:end-1] .- 1),
            convert(Vector{HiGHS.HighsInt}, A.rowval .- 1),
            Float64.(A.nzval),
            convert(Vector{HiGHS.HighsInt}, Q.colptr[1:end-1] .- 1),
            convert(Vector{HiGHS.HighsInt}, Q.rowval .- 1),
            Float64.(Q.nzval),
            col_value,
            col_dual,
            row_value,
            row_dual,
            col_basis_status,
            row_basis_status,
            model_status,
        )
    else
        @timed Highs_lpCall(
            QM.meta.nvar,
            QM.meta.ncon,
            length(A.nzval),
            kHighsMatrixFormatColwise,
            sense,
            Float64(QM.data.c0),
            Float64.(QM.data.c),
            Float64.(QM.meta.lvar),
            Float64.(QM.meta.uvar),
            Float64.(QM.meta.lcon),
            Float64.(QM.meta.ucon),
            convert(Vector{HiGHS.HighsInt}, A.colptr[1:end-1] .- 1),
            convert(Vector{HiGHS.HighsInt}, A.rowval .- 1),
            Float64.(A.nzval),
            col_value,
            col_dual,
            row_value,
            row_dual,
            col_basis_status,
            row_basis_status,
            model_status,
        )
    end
    objective = QM.data.c0 + dot(QM.data.c, col_value)
    if QM.meta.nnzh > 0
        objective += 0.5 * dot(col_value, Symmetric(sparse(QM.data.H.rows, QM.data.H.cols, QM.data.H.vals, QM.meta.nvar, QM.meta.nvar), :L) * col_value)
    end
    return GenericExecutionStats(
        QM,
        status = _status_symbol(model_status[], _highs_statuses),
        solution = col_value,
        objective = objective,
        primal_feas = NaN,
        dual_feas = NaN,
        iter = Int64(-1),
        multipliers = row_dual,
        elapsed_time = timed.time,
    )
end

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# (zero-alloc, shares A/Q triplets) and calls the scalar solver above. Pass
# `threads=1` / `Threads=1` in kwargs to single-thread the inner solver so
# outer Julia threads aren't fighting it for CPUs. `schedule = :dynamic`
# helps when per-instance solve times are very uneven.
function BQMSolvers.highs(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.highs, bqp; kwargs...)
end

end
