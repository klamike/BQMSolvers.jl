module BQMCuPDLPxExt

using CuPDLPx
using BQMSolvers

include("_common.jl")

const Lib = CuPDLPx.LibCuPDLPx

const _cupdlpx_statuses = Dict(
    Lib.TERMINATION_REASON_OPTIMAL => :acceptable,
    Lib.TERMINATION_REASON_PRIMAL_INFEASIBLE => :infeasible,
    Lib.TERMINATION_REASON_DUAL_INFEASIBLE => :unbounded,
    Lib.TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED => :infeasible,
    Lib.TERMINATION_REASON_TIME_LIMIT => :max_time,
    Lib.TERMINATION_REASON_ITERATION_LIMIT => :max_iter,
    Lib.TERMINATION_REASON_FEAS_POLISH_SUCCESS => :acceptable,
)

function _cupdlpx_setparam(params::Lib.pdhg_parameters_t, field::Symbol, value)
    if hasfield(Lib.pdhg_parameters_t, field)
        args = map(fieldnames(Lib.pdhg_parameters_t)) do f
            f == field ? convert(fieldtype(Lib.pdhg_parameters_t, f), value) : getfield(params, f)
        end
        return Lib.pdhg_parameters_t(args...)
    end
    crit = params.termination_criteria
    if hasfield(Lib.termination_criteria_t, field)
        crit_args = map(fieldnames(Lib.termination_criteria_t)) do f
            f == field ? convert(fieldtype(Lib.termination_criteria_t, f), value) : getfield(crit, f)
        end
        crit = Lib.termination_criteria_t(crit_args...)
        args = map(fieldnames(Lib.pdhg_parameters_t)) do f
            f == :termination_criteria ? crit : getfield(params, f)
        end
        return Lib.pdhg_parameters_t(args...)
    end
    error("Unsupported CuPDLPx parameter: $(field)")
end

function BQMSolvers.cupdlpx(qp::BQMScalar; kwargs...)
    length(qp.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    _nnzh(qp) == 0 || error("CuPDLPx only supports linear objectives.")
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    Arows, Acols, Avals = _A_coo(qp)
    A = sparse(Arows, Acols, Avals, ncon, nvar)
    colptr = convert(Vector{Cint}, A.colptr .- 1)
    rowval = convert(Vector{Cint}, A.rowval .- 1)
    nzval  = Float64.(A.nzval)
    c      = Float64.(qp.data.c)
    lcon   = Float64.(qp.meta.lcon)
    ucon   = Float64.(qp.meta.ucon)
    lvar   = Float64.(qp.meta.lvar)
    uvar   = Float64.(qp.meta.uvar)
    obj_const = Ref{Cdouble}(Float64(qp.data.c0[]))
    if !qp.meta.minimize
        c .*= -1
        obj_const[] = -obj_const[]
    end
    desc_val = Lib.matrix_desc_t(ntuple(_ -> 0x00, sizeof(Lib.matrix_desc_t)))
    desc_ref = Ref(desc_val)
    desc_ptr = Base.unsafe_convert(Ptr{Lib.matrix_desc_t}, desc_ref)
    desc_ptr.m = Cint(ncon)
    desc_ptr.n = Cint(nvar)
    desc_ptr.fmt = Lib.matrix_csc
    desc_ptr.data.csc = Lib.MatrixCSC(length(rowval), pointer(colptr), pointer(rowval), pointer(nzval))
    params_ref = Ref{Lib.pdhg_parameters_t}()
    Lib.set_default_parameters(Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref))
    params = params_ref[]
    for (k, v) in kwargs
        params = _cupdlpx_setparam(params, k, v)
    end
    params_ref = Ref(params)
    prob = C_NULL
    result_ptr = C_NULL
    try
        GC.@preserve desc_ref colptr rowval nzval c obj_const lcon ucon lvar uvar begin
            prob = Lib.create_lp_problem(
                pointer(c), desc_ptr,
                pointer(lcon), pointer(ucon),
                pointer(lvar), pointer(uvar),
                Base.unsafe_convert(Ptr{Cdouble}, obj_const),
            )
        end
        GC.@preserve params_ref begin
            result_ptr = Lib.solve_lp_problem(prob, Base.unsafe_convert(Ptr{Lib.pdhg_parameters_t}, params_ref))
        end
        result = unsafe_load(result_ptr)
        x = copy(unsafe_wrap(Vector{Float64}, result.primal_solution, nvar))
        y = copy(unsafe_wrap(Vector{Float64}, result.dual_solution, ncon))
        obj = qp.meta.minimize ? result.primal_objective_value : -result.primal_objective_value
        return GenericExecutionStats(
            qp,
            status = _status_symbol(result.termination_reason, _cupdlpx_statuses),
            solution = x,
            objective = obj,
            primal_feas = result.relative_primal_residual,
            dual_feas = result.relative_dual_residual,
            iter = Int64(result.total_count),
            multipliers = y,
            elapsed_time = result.cumulative_time_sec,
        )
    finally
        result_ptr == C_NULL || Lib.cupdlpx_result_free(result_ptr)
        prob == C_NULL || Lib.lp_problem_free(prob)
    end
end

function BQMSolvers.cupdlpx(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.cupdlpx, bqp; kwargs...)
end

end
