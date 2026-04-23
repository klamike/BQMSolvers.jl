module BQMCOPTExt

using COPT
using BQMSolvers

include("_common.jl")

const _copt_statuses = Dict(
    COPT_LPSTATUS_OPTIMAL => :acceptable,
    COPT_LPSTATUS_INFEASIBLE => :infeasible,
    COPT_LPSTATUS_UNBOUNDED => :unbounded,
    COPT_LPSTATUS_NUMERICAL => :exception,
    COPT_LPSTATUS_IMPRECISE => :acceptable,
    COPT_LPSTATUS_TIMEOUT => :max_time,
    COPT_LPSTATUS_UNFINISHED => :exception,
    COPT_LPSTATUS_INTERRUPTED => :user,
)

function _copt_check(ret::Integer, where_::AbstractString)
    ret == COPT_RETCODE_OK || error("COPT error $(ret) in $(where_)")
end

function _copt_setparam(prob, name::String, value)
    if value isa Integer || value isa Bool
        _copt_check(COPT_SetIntParam(prob, name, Int32(value)), "COPT_SetIntParam")
    elseif value isa AbstractFloat
        _copt_check(COPT_SetDblParam(prob, name, Float64(value)), "COPT_SetDblParam")
    else
        error("Unsupported COPT parameter type for $(name)")
    end
end

function _copt_row_data(QM)
    sense = Vector{Cchar}(undef, QM.meta.ncon)
    bound = Vector{Float64}(undef, QM.meta.ncon)
    upper = zeros(Float64, QM.meta.ncon)
    for i in 1:QM.meta.ncon
        if isfinite(QM.meta.lcon[i]) && isfinite(QM.meta.ucon[i])
            if QM.meta.lcon[i] == QM.meta.ucon[i]
                sense[i] = COPT_EQUAL
                bound[i] = Float64(QM.meta.ucon[i])
            else
                sense[i] = COPT_RANGE
                bound[i] = Float64(QM.meta.ucon[i])
                upper[i] = Float64(QM.meta.ucon[i] - QM.meta.lcon[i])
            end
        elseif isfinite(QM.meta.lcon[i])
            sense[i] = COPT_GREATER_EQUAL
            bound[i] = Float64(QM.meta.lcon[i])
        elseif isfinite(QM.meta.ucon[i])
            sense[i] = COPT_LESS_EQUAL
            bound[i] = Float64(QM.meta.ucon[i])
        else
            sense[i] = COPT_FREE
            bound[i] = 0.0
        end
    end
    return sense, bound, upper
end

function _copt_hessian(QM)
    I = Cint[]
    J = Cint[]
    V = Float64[]
    for k in eachindex(QM.data.H.vals)
        push!(I, Cint(QM.data.H.rows[k] - 1))
        push!(J, Cint(QM.data.H.cols[k] - 1))
        push!(V, QM.data.H.rows[k] == QM.data.H.cols[k] ? Float64(QM.data.H.vals[k]) / 2 : Float64(QM.data.H.vals[k]))
    end
    return I, J, V
end

function BQMSolvers.copt(QM::QuadraticModel{T,S}; kwargs...) where {T,S}
    return BQMSolvers.copt(_coo_model(QM); kwargs...)
end

function BQMSolvers.copt(
    QM::QuadraticModel{T,S,M1,M2};
    kwargs...,
) where {T,S,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}
    length(QM.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    env_ref = Ref{Ptr{copt_env}}(C_NULL)
    prob_ref = Ref{Ptr{copt_prob}}(C_NULL)
    _copt_check(COPT_CreateEnv(env_ref), "COPT_CreateEnv")
    try
        _copt_check(COPT_CreateProb(env_ref[], prob_ref), "COPT_CreateProb")
        prob = prob_ref[]
        for (k, v) in kwargs
            _copt_setparam(prob, string(k), v)
        end
        A = sparse(QM.data.A.rows, QM.data.A.cols, QM.data.A.vals, QM.meta.ncon, QM.meta.nvar)
        rowSense, rowBound, rowUpper = _copt_row_data(QM)
        colLower = [isfinite(v) ? Float64(v) : -COPT_INFINITY for v in QM.meta.lvar]
        colUpper = [isfinite(v) ? Float64(v) : COPT_INFINITY for v in QM.meta.uvar]
        colMatBeg = convert(Vector{Cint}, A.colptr[1:end-1] .- 1)
        colMatCnt = convert(Vector{Cint}, diff(A.colptr))
        colMatIdx = convert(Vector{Cint}, A.rowval .- 1)
        colMatElem = Float64.(A.nzval)
        _copt_check(
            COPT_LoadProb(
                prob,
                QM.meta.nvar,
                QM.meta.ncon,
                QM.meta.minimize ? COPT_MINIMIZE : COPT_MAXIMIZE,
                Float64(QM.data.c0),
                Float64.(QM.data.c),
                colMatBeg,
                colMatCnt,
                colMatIdx,
                colMatElem,
                C_NULL,
                colLower,
                colUpper,
                rowSense,
                rowBound,
                rowUpper,
                C_NULL,
                C_NULL,
            ),
            "COPT_LoadProb",
        )
        if QM.meta.nnzh > 0
            I, J, V = _copt_hessian(QM)
            _copt_check(COPT_SetQuadObj(prob, length(I), I, J, V), "COPT_SetQuadObj")
        end
        t = @timed _copt_check(COPT_Solve(prob), "COPT_Solve")
        status_ref = Ref{Cint}(0)
        _copt_check(COPT_GetIntAttr(prob, "LpStatus", status_ref), "COPT_GetIntAttr(LpStatus)")
        has_sol_ref = Ref{Cint}(0)
        _copt_check(COPT_GetIntAttr(prob, "HasLpSol", has_sol_ref), "COPT_GetIntAttr(HasLpSol)")
        x = fill(NaN, QM.meta.nvar)
        y = fill(NaN, QM.meta.ncon)
        rc = fill(NaN, QM.meta.nvar)
        if has_sol_ref[] != 0
            _copt_check(COPT_GetLpSolution(prob, x, C_NULL, y, rc), "COPT_GetLpSolution")
        end
        obj_ref = Ref{Cdouble}(NaN)
        COPT_GetDblAttr(prob, "LpObjval", obj_ref)
        simplex_ref = Ref{Cint}(0)
        barrier_ref = Ref{Cint}(0)
        COPT_GetIntAttr(prob, "SimplexIter", simplex_ref)
        COPT_GetIntAttr(prob, "BarrierIter", barrier_ref)
        return GenericExecutionStats(
            QM,
            status = _status_symbol(status_ref[], _copt_statuses),
            solution = x,
            objective = obj_ref[],
            primal_feas = NaN,
            dual_feas = NaN,
            iter = Int64(max(simplex_ref[], barrier_ref[])),
            multipliers = y,
            elapsed_time = t.time,
        )
    finally
        prob_ref[] == C_NULL || COPT_DeleteProb(prob_ref)
        env_ref[] == C_NULL || COPT_DeleteEnv(env_ref)
    end
end

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# (zero-alloc, shares A/Q triplets) and calls the scalar solver above. Pass
# `threads=1` / `Threads=1` in kwargs to single-thread the inner solver so
# outer Julia threads aren't fighting it for CPUs. `schedule = :dynamic`
# helps when per-instance solve times are very uneven.
function BQMSolvers.copt(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.copt, bqp; kwargs...)
end

end
