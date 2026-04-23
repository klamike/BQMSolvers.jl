module BQMCPLEXExt

using CPLEX
using BQMSolvers

include("_common.jl")

const _cplex_statuses = Dict(
    1 => :acceptable,
    2 => :unbounded,
    3 => :infeasible,
    4 => :infeasible,
    10 => :max_iter,
    11 => :max_time,
    12 => :exception,
    13 => :user,
)

function _cplex_input(QM::QuadraticModel{T,S}; kwargs...) where {T,S}
    return _cplex_input(_coo_model(QM); kwargs...)
end

function _cplex_input(
    QM::QuadraticModel{T,S,M1,M2};
    method = 1,
    display = 1,
    kwargs...,
) where {T,S,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}
    env = CPLEX.Env()
    CPXsetintparam(env, CPXPARAM_ScreenOutput, display)
    CPXsetdblparam(env, CPXPARAM_TimeLimit, 3600)
    for (k, v) in kwargs
        if k == :presolve
            CPXsetintparam(env, CPXPARAM_Preprocessing_Presolve, v)
        elseif k == :scaling
            CPXsetintparam(env, CPXPARAM_Read_Scale, v)
        elseif k == :crossover
            CPXsetintparam(env, CPXPARAM_SolutionType, v)
        elseif k == :threads
            CPXsetintparam(env, CPXPARAM_Threads, v)
        end
    end
    CPXsetintparam(env, CPXPARAM_LPMethod, method)
    CPXsetintparam(env, CPXPARAM_QPMethod, method)
    status_p = Ref{Cint}()
    lp = CPXcreateprob(env, status_p, "")
    CPXnewcols(env, lp, QM.meta.nvar, QM.data.c, QM.meta.lvar, QM.meta.uvar, C_NULL, C_NULL)
    CPXchgobjsen(env, lp, QM.meta.minimize ? CPX_MIN : CPX_MAX)
    CPXchgobjoffset(env, lp, QM.data.c0)
    if QM.meta.nnzh > 0
        Q = sparse(QM.data.H.rows, QM.data.H.cols, QM.data.H.vals, QM.meta.nvar, QM.meta.nvar)
        diag_matrix = spdiagm(0 => diag(Q))
        Q = Q + Q' - diag_matrix
        qmatcnt = zeros(Int, QM.meta.nvar)
        for k in 1:QM.meta.nvar
            qmatcnt[k] = Q.colptr[k + 1] - Q.colptr[k]
        end
        CPXcopyquad(
            env,
            lp,
            convert(Vector{Cint}, Q.colptr[1:end-1] .- 1),
            convert(Vector{Cint}, qmatcnt),
            convert(Vector{Cint}, Q.rowval .- 1),
            Q.nzval,
        )
    end
    Acsrrowptr, Acsrcolval, Acsrnzval = _sparse_csr(
        QM.data.A.rows,
        QM.data.A.cols,
        QM.data.A.vals,
        QM.meta.ncon,
        QM.meta.nvar,
    )
    sense = fill(Cchar('N'), QM.meta.ncon)
    rhs = zeros(Float64, QM.meta.ncon)
    drange = zeros(Float64, QM.meta.ncon)
    for j in 1:QM.meta.ncon
        sense[j], rhs[j], drange[j] = _row_sense_rhs_range(QM.meta.lcon[j], QM.meta.ucon[j], Inf)
    end
    CPXaddrows(
        env,
        lp,
        0,
        QM.meta.ncon,
        length(Acsrcolval),
        rhs,
        sense,
        convert(Vector{Cint}, Acsrrowptr .- 1),
        convert(Vector{Cint}, Acsrcolval .- 1),
        Acsrnzval,
        C_NULL,
        C_NULL,
    )
    return env, lp
end

function BQMSolvers.cplex(QM::QuadraticModel; method = 4, display = 1, kwargs...)
    env = CPLEX.Env()
    lp = C_NULL
    try
        env, lp = _cplex_input(QM; method = method, display = display, kwargs...)
        t = @timed begin
            if QM.meta.nnzh > 0
                CPXqpopt(env, lp)
            else
                CPXlpopt(env, lp)
            end
        end
        x = Vector{Cdouble}(undef, QM.meta.nvar)
        y = Vector{Cdouble}(undef, QM.meta.ncon)
        s = Vector{Cdouble}(undef, QM.meta.nvar)
        CPXgetx(env, lp, x, 0, QM.meta.nvar - 1)
        CPXgetpi(env, lp, y, 0, QM.meta.ncon - 1)
        CPXgetdj(env, lp, s, 0, QM.meta.nvar - 1)
        primal_feas = Vector{Cdouble}(undef, 1)
        dual_feas = Vector{Cdouble}(undef, 1)
        objval_p = Vector{Cdouble}(undef, 1)
        CPXgetdblquality(env, lp, primal_feas, CPX_MAX_PRIMAL_RESIDUAL)
        CPXgetdblquality(env, lp, dual_feas, CPX_MAX_DUAL_RESIDUAL)
        CPXgetobjval(env, lp, objval_p)
        return GenericExecutionStats(
            QM,
            status = _status_symbol(CPXgetstat(env, lp), _cplex_statuses),
            solution = x,
            objective = objval_p[1],
            primal_feas = primal_feas[1],
            dual_feas = dual_feas[1],
            iter = Int64(CPXgetbaritcnt(env, lp)),
            multipliers = y,
            elapsed_time = t[2],
        )
    finally
        if lp != C_NULL
            CPXfreeprob(env, Ref(lp))
        end
        finalize(env)
    end
end

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# (zero-alloc, shares A/Q triplets) and calls the scalar solver above. Pass
# `threads=1` / `Threads=1` in kwargs to single-thread the inner solver so
# outer Julia threads aren't fighting it for CPUs. `schedule = :dynamic`
# helps when per-instance solve times are very uneven.
function BQMSolvers.cplex(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.cplex, bqp; kwargs...)
end

end
