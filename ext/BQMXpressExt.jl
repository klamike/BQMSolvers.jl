module BQMXpressExt

using BQMSolvers
using Xpress

include("_common.jl")

const _xpress_statuses = Dict(
    0 => :unknown,
    1 => :acceptable,
    2 => :infeasible,
    3 => :exception,
    4 => :max_eval,
    5 => :unbounded,
    6 => :exception,
    7 => :exception,
    8 => :exception,
)

function BQMSolvers.xpress(qp::BQMScalar; method = "b", kwargs...)
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    Xpress.init()
    prob = Xpress.XpressProblem(; finalize_env = false)
    try
        for (k, v) in kwargs
            if k == :presolve
                Xpress.setintcontrol(prob, Xpress.Lib.XPRS_PRESOLVE, v)
            elseif k == :scaling
                Xpress.setintcontrol(prob, Xpress.Lib.XPRS_SCALING, v)
            elseif k == :crossover
                Xpress.setintcontrol(prob, Xpress.Lib.XPRS_CROSSOVER, v)
            elseif k == :threads
                Xpress.setintcontrol(prob, Xpress.Lib.XPRS_THREADS, v)
            elseif k == :bargapstop
                Xpress.setdblcontrol(prob, Xpress.Lib.XPRS_BARGAPSTOP, v)
            elseif k == :barprimalstop
                Xpress.setdblcontrol(prob, Xpress.Lib.XPRS_BARPRIMALSTOP, v)
            elseif k == :bardualstop
                Xpress.setdblcontrol(prob, Xpress.Lib.XPRS_BARDUALSTOP, v)
            end
        end
        srowtypes = fill(Cchar('N'), ncon)
        rhs = zeros(Float64, ncon)
        drange = zeros(Float64, ncon)
        for j in 1:ncon
            srowtypes[j], rhs[j], drange[j] =
                _row_sense_rhs_range(qp.meta.lcon[j], qp.meta.ucon[j], Xpress.Lib.XPRS_PLUSINFINITY)
        end
        Arows, Acols, Avals = _A_coo(qp)
        A = sparse(Arows, Acols, Avals, ncon, nvar)
        lvar = [isfinite(v) ? Float64(v) : Xpress.Lib.XPRS_MINUSINFINITY for v in qp.meta.lvar]
        uvar = [isfinite(v) ? Float64(v) : Xpress.Lib.XPRS_PLUSINFINITY for v in qp.meta.uvar]
        if _nnzh(qp) > 0
            Hrows, Hcols, Hvals = _Q_coo(qp)
            Xpress.loadqp(
                prob, qp.meta.name, nvar, ncon,
                srowtypes, rhs, drange,
                Vector{Float64}(qp.data.c),
                convert(Vector{Cint}, A.colptr .- 1), C_NULL,
                convert(Vector{Cint}, A.rowval .- 1), A.nzval,
                lvar, uvar,
                length(Hvals),
                convert(Vector{Cint}, Hrows .- 1),
                convert(Vector{Cint}, Hcols .- 1),
                Vector{Float64}(Hvals),
            )
        else
            Xpress.loadlp(
                prob, "", nvar, ncon,
                srowtypes, rhs, drange,
                Vector{Float64}(qp.data.c),
                convert(Vector{Cint}, A.colptr .- 1), C_NULL,
                convert(Vector{Cint}, A.rowval .- 1), A.nzval,
                lvar, uvar,
            )
        end
        Xpress.chgobjsense(prob, qp.meta.minimize ? :minimize : :maximize)
        Xpress.chgobj(prob, [0], [-Float64(qp.data.c0[])])
        start_time = time()
        Xpress.lpoptimize(prob, method)
        elapsed_time = time() - start_time
        x = zeros(nvar); y = zeros(ncon); s = zeros(nvar)
        Xpress.getsol(prob, x, C_NULL, y, s)
        return GenericExecutionStats(
            qp,
            status = _status_symbol(Xpress.getintattrib(prob, Xpress.Lib.XPRS_LPSTATUS), _xpress_statuses),
            solution = x,
            objective = Xpress.getdblattrib(prob, Xpress.Lib.XPRS_LPOBJVAL),
            primal_feas = Xpress.getdblattrib(prob, Xpress.Lib.XPRS_BARPRIMALINF),
            dual_feas = Xpress.getdblattrib(prob, Xpress.Lib.XPRS_BARDUALINF),
            iter = Int64(Xpress.getintattrib(prob, Xpress.Lib.XPRS_BARITER)),
            multipliers = y,
            elapsed_time = elapsed_time,
        )
    finally
        Xpress.destroyprob(prob)
    end
end

function BQMSolvers.xpress(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.xpress, bqp; kwargs...)
end

end
