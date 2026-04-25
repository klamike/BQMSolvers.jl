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

function _copt_row_data(qp)
    ncon = qp.meta.ncon
    sense = Vector{Cchar}(undef, ncon)
    bound = Vector{Float64}(undef, ncon)
    upper = zeros(Float64, ncon)
    for i in 1:ncon
        l = qp.meta.lcon[i]; u = qp.meta.ucon[i]
        if isfinite(l) && isfinite(u)
            if l == u
                sense[i] = COPT_EQUAL
                bound[i] = Float64(u)
            else
                sense[i] = COPT_RANGE
                bound[i] = Float64(u)
                upper[i] = Float64(u - l)
            end
        elseif isfinite(l)
            sense[i] = COPT_GREATER_EQUAL
            bound[i] = Float64(l)
        elseif isfinite(u)
            sense[i] = COPT_LESS_EQUAL
            bound[i] = Float64(u)
        else
            sense[i] = COPT_FREE
            bound[i] = 0.0
        end
    end
    return sense, bound, upper
end

function _copt_hessian(qp)
    Hrows, Hcols, Hvals = _Q_coo(qp)
    I = Cint[]; J = Cint[]; V = Float64[]
    @inbounds for k in eachindex(Hvals)
        push!(I, Cint(Hrows[k] - 1))
        push!(J, Cint(Hcols[k] - 1))
        push!(V, Hrows[k] == Hcols[k] ? Float64(Hvals[k]) / 2 : Float64(Hvals[k]))
    end
    return I, J, V
end

function BQMSolvers.copt(qp::BQMScalar; kwargs...)
    length(qp.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    env_ref = Ref{Ptr{copt_env}}(C_NULL)
    prob_ref = Ref{Ptr{copt_prob}}(C_NULL)
    _copt_check(COPT_CreateEnv(env_ref), "COPT_CreateEnv")
    try
        _copt_check(COPT_CreateProb(env_ref[], prob_ref), "COPT_CreateProb")
        prob = prob_ref[]
        for (k, v) in kwargs
            _copt_setparam(prob, string(k), v)
        end
        Arows, Acols, Avals = _A_coo(qp)
        A = sparse(Arows, Acols, Avals, ncon, nvar)
        rowSense, rowBound, rowUpper = _copt_row_data(qp)
        colLower = [isfinite(v) ? Float64(v) : -COPT_INFINITY for v in qp.meta.lvar]
        colUpper = [isfinite(v) ? Float64(v) : COPT_INFINITY for v in qp.meta.uvar]
        colMatBeg = convert(Vector{Cint}, A.colptr[1:end-1] .- 1)
        colMatCnt = convert(Vector{Cint}, diff(A.colptr))
        colMatIdx = convert(Vector{Cint}, A.rowval .- 1)
        colMatElem = Float64.(A.nzval)
        _copt_check(
            COPT_LoadProb(
                prob, nvar, ncon,
                qp.meta.minimize ? COPT_MINIMIZE : COPT_MAXIMIZE,
                Float64(qp.data.c0[]),
                Float64.(qp.data.c),
                colMatBeg, colMatCnt, colMatIdx, colMatElem, C_NULL,
                colLower, colUpper,
                rowSense, rowBound, rowUpper,
                C_NULL, C_NULL,
            ),
            "COPT_LoadProb",
        )
        if _nnzh(qp) > 0
            I, J, V = _copt_hessian(qp)
            _copt_check(COPT_SetQuadObj(prob, length(I), I, J, V), "COPT_SetQuadObj")
        end
        t = @timed _copt_check(COPT_Solve(prob), "COPT_Solve")
        status_ref = Ref{Cint}(0)
        _copt_check(COPT_GetIntAttr(prob, "LpStatus", status_ref), "COPT_GetIntAttr(LpStatus)")
        has_sol_ref = Ref{Cint}(0)
        _copt_check(COPT_GetIntAttr(prob, "HasLpSol", has_sol_ref), "COPT_GetIntAttr(HasLpSol)")
        x = fill(NaN, nvar)
        y = fill(NaN, ncon)
        rc = fill(NaN, nvar)
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
            qp,
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

function BQMSolvers.copt(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.copt, bqp; kwargs...)
end

end
