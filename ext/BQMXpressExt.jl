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

function BQMSolvers.xpress(QM::QuadraticModel{T,S}; kwargs...) where {T,S}
    return BQMSolvers.xpress(_coo_model(QM); kwargs...)
end

function BQMSolvers.xpress(
    QM::QuadraticModel{T,S,M1,M2};
    method = "b",
    kwargs...,
) where {T,S,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}
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
        srowtypes = fill(Cchar('N'), QM.meta.ncon)
        rhs = zeros(Float64, QM.meta.ncon)
        drange = zeros(Float64, QM.meta.ncon)
        for j in 1:QM.meta.ncon
            srowtypes[j], rhs[j], drange[j] =
                _row_sense_rhs_range(QM.meta.lcon[j], QM.meta.ucon[j], Xpress.Lib.XPRS_PLUSINFINITY)
        end
        A = sparse(QM.data.A.rows, QM.data.A.cols, QM.data.A.vals, QM.meta.ncon, QM.meta.nvar)
        lvar = [isfinite(v) ? Float64(v) : Xpress.Lib.XPRS_MINUSINFINITY for v in QM.meta.lvar]
        uvar = [isfinite(v) ? Float64(v) : Xpress.Lib.XPRS_PLUSINFINITY for v in QM.meta.uvar]
        if QM.meta.nnzh > 0
            Xpress.loadqp(
                prob,
                QM.meta.name,
                QM.meta.nvar,
                QM.meta.ncon,
                srowtypes,
                rhs,
                drange,
                QM.data.c,
                convert(Vector{Cint}, A.colptr .- 1),
                C_NULL,
                convert(Vector{Cint}, A.rowval .- 1),
                A.nzval,
                lvar,
                uvar,
                QM.meta.nnzh,
                convert(Vector{Cint}, QM.data.H.rows .- 1),
                convert(Vector{Cint}, QM.data.H.cols .- 1),
                QM.data.H.vals,
            )
        else
            Xpress.loadlp(
                prob,
                "",
                QM.meta.nvar,
                QM.meta.ncon,
                srowtypes,
                rhs,
                drange,
                QM.data.c,
                convert(Vector{Cint}, A.colptr .- 1),
                C_NULL,
                convert(Vector{Cint}, A.rowval .- 1),
                A.nzval,
                lvar,
                uvar,
            )
        end
        Xpress.chgobjsense(prob, QM.meta.minimize ? :minimize : :maximize)
        Xpress.chgobj(prob, [0], [-QM.data.c0])
        start_time = time()
        Xpress.lpoptimize(prob, method)
        elapsed_time = time() - start_time
        x = zeros(QM.meta.nvar)
        y = zeros(QM.meta.ncon)
        s = zeros(QM.meta.nvar)
        Xpress.getsol(prob, x, C_NULL, y, s)
        return GenericExecutionStats(
            QM,
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

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# (zero-alloc, shares A/Q triplets) and calls the scalar solver above. Pass
# `threads=1` / `Threads=1` in kwargs to single-thread the inner solver so
# outer Julia threads aren't fighting it for CPUs. `schedule = :dynamic`
# helps when per-instance solve times are very uneven.
function BQMSolvers.xpress(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.xpress, bqp; kwargs...)
end

end
