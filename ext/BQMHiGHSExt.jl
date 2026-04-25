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

# Upper-triangular Hessian in CSC — HiGHS's convention. BQM stores lower-tri,
# so swap (i,j) before packing.
function _highs_hessian(qp, nvar::Int)
    Hrows, Hcols, Hvals = _Q_coo(qp)
    I = Int[]; J = Int[]; V = Float64[]
    @inbounds for k in eachindex(Hvals)
        i = Int(Hrows[k]); j = Int(Hcols[k])
        row, col = i > j ? (i, j) : (j, i)
        push!(I, row); push!(J, col); push!(V, Float64(Hvals[k]))
    end
    return sparse(I, J, V, nvar, nvar)
end

# Set one HiGHS option by name, dispatching on the Julia value type.
function _highs_setopt!(h, name::String, value)
    if value isa Bool
        Highs_setBoolOptionValue(h, name, value ? Cint(1) : Cint(0))
    elseif value isa Integer
        Highs_setIntOptionValue(h, name, HiGHS.HighsInt(value))
    elseif value isa AbstractFloat
        Highs_setDoubleOptionValue(h, name, Float64(value))
    elseif value isa AbstractString || value isa Symbol
        Highs_setStringOptionValue(h, name, String(value))
    else
        error("Unsupported HiGHS option type for $(name): $(typeof(value))")
    end
end

function BQMSolvers.highs(qp::BQMScalar; kwargs...)
    length(qp.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    Arows, Acols, Avals = _A_coo(qp)
    A = sparse(Arows, Acols, Avals, ncon, nvar)
    sense = qp.meta.minimize ? kHighsObjSenseMinimize : kHighsObjSenseMaximize
    c0 = Float64(qp.data.c0[])

    h = Highs_create()
    try
        # Default to quiet; caller can override via `output_flag = true`.
        Highs_setBoolOptionValue(h, "output_flag", Cint(0))
        for (k, v) in kwargs
            _highs_setopt!(h, String(k), v)
        end
        Highs_passLp(
            h, nvar, ncon,
            length(A.nzval), kHighsMatrixFormatColwise, sense,
            c0, Float64.(qp.data.c),
            Float64.(qp.meta.lvar), Float64.(qp.meta.uvar),
            Float64.(qp.meta.lcon), Float64.(qp.meta.ucon),
            convert(Vector{HiGHS.HighsInt}, A.colptr[1:end-1] .- 1),
            convert(Vector{HiGHS.HighsInt}, A.rowval .- 1),
            Float64.(A.nzval),
        )
        if _nnzh(qp) > 0
            Q = _highs_hessian(qp, nvar)
            Highs_passHessian(
                h, nvar, length(Q.nzval), kHighsHessianFormatTriangular,
                convert(Vector{HiGHS.HighsInt}, Q.colptr[1:end-1] .- 1),
                convert(Vector{HiGHS.HighsInt}, Q.rowval .- 1),
                Float64.(Q.nzval),
            )
        end
        timed = @timed Highs_run(h)
        model_status = Highs_getModelStatus(h)

        col_value = zeros(Float64, nvar)
        col_dual  = zeros(Float64, nvar)
        row_value = zeros(Float64, ncon)
        row_dual  = zeros(Float64, ncon)
        Highs_getSolution(h, col_value, col_dual, row_value, row_dual)

        objective = c0 + dot(qp.data.c, col_value)
        if _nnzh(qp) > 0
            Hrows, Hcols, Hvals = _Q_coo(qp)
            objective += 0.5 * dot(col_value, Symmetric(sparse(Hrows, Hcols, Hvals, nvar, nvar), :L) * col_value)
        end

        iter_ref = Ref{HiGHS.HighsInt}(0)
        Highs_getIntInfoValue(h, "ipm_iteration_count", iter_ref)
        iter_ipm = Int64(iter_ref[])
        Highs_getIntInfoValue(h, "simplex_iteration_count", iter_ref)
        iter_splx = Int64(iter_ref[])

        return GenericExecutionStats(
            qp,
            status = _status_symbol(model_status, _highs_statuses),
            solution = col_value,
            objective = objective,
            primal_feas = NaN,
            dual_feas = NaN,
            iter = iter_ipm > 0 ? iter_ipm : iter_splx,
            multipliers = row_dual,
            elapsed_time = timed.time,
        )
    finally
        Highs_destroy(h)
    end
end

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# and calls the scalar solver above. Pass `threads=1` / `Threads=1` in
# `kwargs` to single-thread the inner solver so outer Julia threads aren't
# fighting it for CPUs. `schedule = :dynamic` helps with uneven workloads.
function BQMSolvers.highs(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.highs, bqp; kwargs...)
end

end
