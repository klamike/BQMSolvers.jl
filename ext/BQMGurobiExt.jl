module BQMGurobiExt

using Gurobi
using BQMSolvers

include("_common.jl")

const _gurobi_statuses = Dict(
    1 => :unknown,
    2 => :acceptable,
    3 => :infeasible,
    4 => :infeasible,
    5 => :unbounded,
    6 => :exception,
    7 => :max_iter,
    8 => :exception,
    9 => :max_time,
    10 => :exception,
    11 => :user,
    12 => :exception,
    13 => :exception,
    14 => :exception,
    15 => :exception,
)

function BQMSolvers.gurobi(qp::BQMScalar; kwargs...)
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    env = Gurobi.Env(Dict{String,Any}(string(k) => v for (k, v) in kwargs))
    model = Ref{Ptr{Cvoid}}()
    try
        GRBnewmodel(
            env, model, "", nvar,
            Vector{Float64}(qp.data.c),
            Vector{Float64}(qp.meta.lvar),
            Vector{Float64}(qp.meta.uvar),
            C_NULL, C_NULL,
        )
        GRBsetdblattr(model.x, "ObjCon", Float64(qp.data.c0[]))
        GRBsetintattr(model.x, "ModelSense", qp.meta.minimize ? 1 : -1)
        if _nnzh(qp) > 0
            Hrows, Hcols, Hvals = _Q_coo(qp)
            hvals = zeros(eltype(Hvals), length(Hvals))
            @inbounds for i in eachindex(Hvals)
                hvals[i] = Hrows[i] == Hcols[i] ? Hvals[i] / 2 : Hvals[i]
            end
            GRBaddqpterms(
                model.x, length(Hcols),
                convert(Vector{Cint}, Hrows .- 1),
                convert(Vector{Cint}, Hcols .- 1),
                hvals,
            )
        end
        Arows, Acols, Avals = _A_coo(qp)
        Acsrrowptr, Acsrcolval, Acsrnzval = _sparse_csr(Arows, Acols, Avals, ncon, nvar)
        GRBaddrangeconstrs(
            model.x, ncon, length(Acsrcolval),
            convert(Vector{Cint}, Acsrrowptr .- 1),
            convert(Vector{Cint}, Acsrcolval .- 1),
            Acsrnzval,
            Vector{Float64}(qp.meta.lcon),
            Vector{Float64}(qp.meta.ucon),
            C_NULL,
        )
        GRBoptimize(model.x)
        x = zeros(nvar); y = zeros(ncon); s = zeros(nvar)
        GRBgetdblattrarray(model.x, "X", 0, nvar, x)
        GRBgetdblattrarray(model.x, "Pi", 0, ncon, y)
        GRBgetdblattrarray(model.x, "RC", 0, nvar, s)
        status   = Ref{Cint}()
        baritcnt = Ref{Cint}()
        objval   = Ref{Float64}()
        p_feas   = Ref{Float64}()
        d_feas   = Ref{Float64}()
        elapsed  = Ref{Float64}()
        GRBgetintattr(model.x, "Status", status)
        GRBgetintattr(model.x, "BarIterCount", baritcnt)
        GRBgetdblattr(model.x, "ObjVal", objval)
        GRBgetdblattr(model.x, "ConstrResidual", p_feas)
        GRBgetdblattr(model.x, "DualResidual", d_feas)
        GRBgetdblattr(model.x, "Runtime", elapsed)
        return GenericExecutionStats(
            qp,
            status = _status_symbol(status[], _gurobi_statuses),
            solution = x,
            objective = objval[],
            iter = Int64(baritcnt[]),
            primal_feas = p_feas[],
            dual_feas = d_feas[],
            multipliers = y,
            elapsed_time = elapsed[],
        )
    finally
        if model[] != C_NULL
            GRBfreemodel(model[])
        end
        finalize(env)
    end
end

function BQMSolvers.gurobi(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.gurobi, bqp; kwargs...)
end

end
