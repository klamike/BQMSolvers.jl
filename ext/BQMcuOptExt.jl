module BQMcuOptExt

using BQMSolvers
using cuOpt

include("_common.jl")

const _cuopt_statuses = Dict(
    cuOpt.CUOPT_TERIMINATION_STATUS_OPTIMAL => :acceptable,
    cuOpt.CUOPT_TERIMINATION_STATUS_INFEASIBLE => :infeasible,
    cuOpt.CUOPT_TERIMINATION_STATUS_UNBOUNDED => :unbounded,
    cuOpt.CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT => :max_iter,
    cuOpt.CUOPT_TERIMINATION_STATUS_TIME_LIMIT => :max_time,
    cuOpt.CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR => :exception,
    cuOpt.CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE => :acceptable,
    cuOpt.CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND => :acceptable,
    cuOpt.CUOPT_TERIMINATION_STATUS_CONCURRENT_LIMIT => :unknown,
)

function _cuopt_check(status, where_)
    status == cuOpt.CUOPT_SUCCESS || error("cuOpt error in $(where_): $(status)")
end

function _cuopt_setparam(settings, name::String, value)
    if value isa Integer
        return _cuopt_check(
            cuOpt.cuOptSetIntegerParameter(settings, name, Int32(value)),
            "cuOptSetIntegerParameter",
        )
    elseif value isa AbstractFloat
        return _cuopt_check(
            cuOpt.cuOptSetFloatParameter(settings, name, Float64(value)),
            "cuOptSetFloatParameter",
        )
    else
        return _cuopt_check(
            cuOpt.cuOptSetParameter(settings, name, string(value)),
            "cuOptSetParameter",
        )
    end
end

function _cuopt_csr(I, J, V,
                     m = isempty(I) ? 0 : maximum(I),
                     n = isempty(J) ? 0 : maximum(J))
    row_offsets = zeros(Int32, m + 1)
    coolen = length(I)
    min(length(J), length(V)) >= coolen ||
        throw(ArgumentError("J and V need length >= length(I) = $coolen"))
    @inbounds for k in 1:coolen
        Ik = I[k]
        1 <= Ik <= m || throw(ArgumentError("row indices I[k] must satisfy 1 <= I[k] <= m"))
        row_offsets[Ik + 1] += 1
    end
    @inbounds for i in 1:m
        row_offsets[i + 1] += row_offsets[i]
    end
    next_slot = copy(row_offsets)
    col_indices = zeros(Int32, coolen)
    values = zeros(Float64, coolen)
    @inbounds for k in 1:coolen
        Ik, Jk = I[k], J[k]
        1 <= Jk <= n || throw(ArgumentError("column indices J[k] must satisfy 1 <= J[k] <= n"))
        p = next_slot[Ik] + 1
        next_slot[Ik] = p
        col_indices[p] = Int32(Jk - 1)
        values[p] = Float64(V[k])
    end
    return row_offsets, col_indices, values
end

function _cuopt_qcsr(qp, nvar::Int)
    Hrows, Hcols, Hvals = _Q_coo(qp)
    I = Int[]; J = Int[]; V = Float64[]
    @inbounds for k in eachindex(Hvals)
        i = Int(Hrows[k]); j = Int(Hcols[k])
        push!(I, i); push!(J, j)
        push!(V, i == j ? Float64(Hvals[k]) / 2 : Float64(Hvals[k]))
    end
    return _cuopt_csr(I, J, V, nvar, nvar)
end

function _cuopt_problem(qp::BQMScalar)
    length(qp.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    Arows, Acols, Avals = _A_coo(qp)
    Arowptr, Acolind, Avals_csr = _cuopt_csr(Arows, Acols, Avals, ncon, nvar)
    qrowptr, qcolind, qvals = _cuopt_qcsr(qp, nvar)
    sense = qp.meta.minimize ? cuOpt.CUOPT_MINIMIZE : cuOpt.CUOPT_MAXIMIZE
    problem_ref = Ref{cuOpt.cuOptOptimizationProblem}()
    if isempty(qvals)
        variable_types = fill(Cchar(cuOpt.CUOPT_CONTINUOUS), nvar)
        _cuopt_check(
            cuOpt.cuOptCreateRangedProblem(
                Int32(ncon), Int32(nvar), sense,
                Float64(qp.data.c0[]),
                Float64.(qp.data.c),
                Arowptr, Acolind, Avals_csr,
                Float64.(qp.meta.lcon), Float64.(qp.meta.ucon),
                Float64.(qp.meta.lvar), Float64.(qp.meta.uvar),
                variable_types, problem_ref,
            ),
            "cuOptCreateRangedProblem",
        )
    else
        _cuopt_check(
            cuOpt.cuOptCreateQuadraticRangedProblem(
                Int32(ncon), Int32(nvar), sense,
                Float64(qp.data.c0[]),
                Float64.(qp.data.c),
                qrowptr, qcolind, qvals,
                Arowptr, Acolind, Avals_csr,
                Float64.(qp.meta.lcon), Float64.(qp.meta.ucon),
                Float64.(qp.meta.lvar), Float64.(qp.meta.uvar),
                problem_ref,
            ),
            "cuOptCreateQuadraticRangedProblem",
        )
    end
    return problem_ref
end

function _cuopt_stats(qp, solution)
    status_ref = Ref{Int32}(0)
    _cuopt_check(
        cuOpt.cuOptGetTerminationStatus(solution, status_ref),
        "cuOptGetTerminationStatus",
    )
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    x = zeros(Float64, nvar)
    y = zeros(Float64, ncon)
    _cuopt_check(cuOpt.cuOptGetPrimalSolution(solution, x), "cuOptGetPrimalSolution")
    cuOpt.cuOptGetDualSolution(solution, y)
    obj = Ref{Float64}(NaN)
    solve_time = Ref{Float64}(NaN)
    cuOpt.cuOptGetObjectiveValue(solution, obj)
    cuOpt.cuOptGetSolveTime(solution, solve_time)
    return GenericExecutionStats(
        qp,
        status = _status_symbol(status_ref[], _cuopt_statuses),
        solution = x,
        objective = obj[],
        primal_feas = NaN,
        dual_feas = NaN,
        iter = Int64(-1),
        multipliers = y,
        elapsed_time = solve_time[],
    )
end

function BQMSolvers.cuopt(qp::BQMScalar; kwargs...)
    problem_ref = _cuopt_problem(qp)
    settings_ref = Ref{cuOpt.cuOptSolverSettings}()
    _cuopt_check(cuOpt.cuOptCreateSolverSettings(settings_ref), "cuOptCreateSolverSettings")
    for (k, v) in kwargs
        _cuopt_setparam(settings_ref[], string(k), v)
    end
    solution_ref = Ref{cuOpt.cuOptSolution}()
    try
        _cuopt_check(
            cuOpt.cuOptSolve(problem_ref[], settings_ref[], solution_ref),
            "cuOptSolve",
        )
        return _cuopt_stats(qp, solution_ref[])
    finally
        cuOpt.cuOptDestroySolution(solution_ref)
        cuOpt.cuOptDestroySolverSettings(settings_ref)
        cuOpt.cuOptDestroyProblem(problem_ref)
    end
end

function BQMSolvers.cuopt(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.cuopt, bqp; kwargs...)
end

end
