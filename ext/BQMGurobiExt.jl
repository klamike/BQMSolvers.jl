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

function BQMSolvers.gurobi(QM::QuadraticModel{T,S}; kwargs...) where {T,S}
    return BQMSolvers.gurobi(_coo_model(QM); kwargs...)
end

function BQMSolvers.gurobi(
    QM::QuadraticModel{T,S,M1,M2};
    kwargs...,
) where {T,S,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}
    env = Gurobi.Env(Dict{String,Any}(string(k) => v for (k, v) in kwargs))
    model = Ref{Ptr{Cvoid}}()
    try
        GRBnewmodel(
            env,
            model,
            "",
            QM.meta.nvar,
            QM.data.c,
            QM.meta.lvar,
            QM.meta.uvar,
            C_NULL,
            C_NULL,
        )
        GRBsetdblattr(model.x, "ObjCon", QM.data.c0)
        GRBsetintattr(model.x, "ModelSense", QM.meta.minimize ? 1 : -1)
        if QM.meta.nnzh > 0
            hvals = zeros(eltype(QM.data.H.vals), length(QM.data.H.vals))
            for i in eachindex(QM.data.H.vals)
                hvals[i] = QM.data.H.rows[i] == QM.data.H.cols[i] ? QM.data.H.vals[i] / 2 : QM.data.H.vals[i]
            end
            GRBaddqpterms(
                model.x,
                length(QM.data.H.cols),
                convert(Vector{Cint}, QM.data.H.rows .- 1),
                convert(Vector{Cint}, QM.data.H.cols .- 1),
                hvals,
            )
        end
        Acsrrowptr, Acsrcolval, Acsrnzval = _sparse_csr(
            QM.data.A.rows,
            QM.data.A.cols,
            QM.data.A.vals,
            QM.meta.ncon,
            QM.meta.nvar,
        )
        GRBaddrangeconstrs(
            model.x,
            QM.meta.ncon,
            length(Acsrcolval),
            convert(Vector{Cint}, Acsrrowptr .- 1),
            convert(Vector{Cint}, Acsrcolval .- 1),
            Acsrnzval,
            QM.meta.lcon,
            QM.meta.ucon,
            C_NULL,
        )
        GRBoptimize(model.x)
        x = zeros(QM.meta.nvar)
        y = zeros(QM.meta.ncon)
        s = zeros(QM.meta.nvar)
        GRBgetdblattrarray(model.x, "X", 0, QM.meta.nvar, x)
        GRBgetdblattrarray(model.x, "Pi", 0, QM.meta.ncon, y)
        GRBgetdblattrarray(model.x, "RC", 0, QM.meta.nvar, s)
        status = Ref{Cint}()
        baritcnt = Ref{Cint}()
        objval = Ref{Float64}()
        p_feas = Ref{Float64}()
        d_feas = Ref{Float64}()
        elapsed_time = Ref{Float64}()
        GRBgetintattr(model.x, "Status", status)
        GRBgetintattr(model.x, "BarIterCount", baritcnt)
        GRBgetdblattr(model.x, "ObjVal", objval)
        GRBgetdblattr(model.x, "ConstrResidual", p_feas)
        GRBgetdblattr(model.x, "DualResidual", d_feas)
        GRBgetdblattr(model.x, "Runtime", elapsed_time)
        return GenericExecutionStats(
            QM,
            status = _status_symbol(status[], _gurobi_statuses),
            solution = x,
            objective = objval[],
            iter = Int64(baritcnt[]),
            primal_feas = p_feas[],
            dual_feas = d_feas[],
            multipliers = y,
            elapsed_time = elapsed_time[],
        )
    finally
        if model[] != C_NULL
            GRBfreemodel(model[])
        end
        finalize(env)
    end
end

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# (zero-alloc, shares A/Q triplets) and calls the scalar solver above. Pass
# `threads=1` / `Threads=1` in kwargs to single-thread the inner solver so
# outer Julia threads aren't fighting it for CPUs. `schedule = :dynamic`
# helps when per-instance solve times are very uneven.
function BQMSolvers.gurobi(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.gurobi, bqp; kwargs...)
end

end
