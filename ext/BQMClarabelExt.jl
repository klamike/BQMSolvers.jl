module BQMClarabelExt

using Clarabel
using BQMSolvers

include("_common.jl")

const _clarabel_statuses = Dict(
    :optimal => :acceptable,
    :Optimal => :acceptable,
    Clarabel.SOLVED => :acceptable,
    Clarabel.ALMOST_SOLVED => :acceptable,
    Clarabel.PRIMAL_INFEASIBLE => :infeasible,
    Clarabel.ALMOST_PRIMAL_INFEASIBLE => :infeasible,
    Clarabel.DUAL_INFEASIBLE => :unbounded,
    Clarabel.ALMOST_DUAL_INFEASIBLE => :unbounded,
    Clarabel.MAX_ITERATIONS => :max_iter,
    Clarabel.MAX_TIME => :max_time,
    Clarabel.NUMERICAL_ERROR => :exception,
    Clarabel.INSUFFICIENT_PROGRESS => :stalled,
)

# Clarabel wants the upper-triangular Hessian. BQM's `Q_coo` is lower-tri,
# so swap (i,j) before packing.
function _clarabel_hessian(qp, nvar::Int)
    Hrows, Hcols, Hvals = _Q_coo(qp)
    I = Int[]; J = Int[]; V = Float64[]
    @inbounds for k in eachindex(Hvals)
        i = Int(Hrows[k]); j = Int(Hcols[k])
        row, col = i < j ? (i, j) : (j, i)
        push!(I, row); push!(J, col); push!(V, Float64(Hvals[k]))
    end
    return sparse(I, J, V, nvar, nvar)
end

function _push_bound_row!(I, J, V, b, cones, coeffs, sign, rhs)
    row = length(b) + 1
    for (col, val) in coeffs
        push!(I, row)
        push!(J, col)
        push!(V, sign * val)
    end
    push!(b, rhs)
    push!(cones, Clarabel.NonnegativeConeT(1))
    return row
end

function _push_eq_row!(I, J, V, b, cones, coeffs, rhs)
    row = length(b) + 1
    for (col, val) in coeffs
        push!(I, row)
        push!(J, col)
        push!(V, val)
    end
    push!(b, rhs)
    push!(cones, Clarabel.ZeroConeT(1))
    return row
end

function BQMSolvers.clarabel(qp::BQMScalar; kwargs...)
    length(qp.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    nvar = qp.meta.nvar; ncon = qp.meta.ncon
    Arows, Acols, Avals = _A_coo(qp)
    row_terms = [Tuple{Int,Float64}[] for _ in 1:ncon]
    @inbounds for k in eachindex(Avals)
        push!(row_terms[Int(Arows[k])], (Int(Acols[k]), Float64(Avals[k])))
    end
    I = Int[]; J = Int[]; V = Float64[]
    b = Float64[]
    cones = Clarabel.SupportedCone[]
    for i in 1:nvar
        if isfinite(qp.meta.lvar[i])
            _push_bound_row!(I, J, V, b, cones, [(i, 1.0)], -1.0, -Float64(qp.meta.lvar[i]))
        end
        if isfinite(qp.meta.uvar[i])
            _push_bound_row!(I, J, V, b, cones, [(i, 1.0)], 1.0, Float64(qp.meta.uvar[i]))
        end
    end
    row_map = Vector{Any}(undef, ncon)
    for i in 1:ncon
        l = qp.meta.lcon[i]
        u = qp.meta.ucon[i]
        coeffs = row_terms[i]
        if isfinite(l) && isfinite(u)
            if l == u
                idx = _push_eq_row!(I, J, V, b, cones, coeffs, Float64(u))
                row_map[i] = (:eq, idx)
            else
                idx1 = _push_bound_row!(I, J, V, b, cones, coeffs, -1.0, -Float64(l))
                idx2 = _push_bound_row!(I, J, V, b, cones, coeffs, 1.0, Float64(u))
                row_map[i] = (:interval, idx1, idx2)
            end
        elseif isfinite(l)
            idx = _push_bound_row!(I, J, V, b, cones, coeffs, -1.0, -Float64(l))
            row_map[i] = (:lower, idx)
        elseif isfinite(u)
            idx = _push_bound_row!(I, J, V, b, cones, coeffs, 1.0, Float64(u))
            row_map[i] = (:upper, idx)
        else
            row_map[i] = nothing
        end
    end
    A = sparse(I, J, V, length(b), nvar)
    q = Float64.(qp.data.c)
    P = _clarabel_hessian(qp, nvar)
    c0 = Float64(qp.data.c0[])
    sense = qp.meta.minimize ? 1.0 : -1.0
    solver_kwargs = Dict{Symbol,Any}(kwargs)
    if haskey(solver_kwargs, :verbose)
        v = solver_kwargs[:verbose]
        solver_kwargs[:verbose] = v isa Bool ? v : Bool(v != 0)
    end
    solver = Clarabel.Solver(sense * P, sense * q, A, b, cones; solver_kwargs...)
    Clarabel.solve!(solver)
    info = Clarabel.get_info(solver)
    solution = Clarabel.get_solution(solver)
    x = copy(solution.x)
    z = copy(solution.z)
    y = fill(NaN, ncon)
    for i in 1:ncon
        map_i = row_map[i]
        if map_i === nothing
            continue
        elseif map_i[1] == :eq || map_i[1] == :upper
            y[i] = z[map_i[2]]
        elseif map_i[1] == :lower
            y[i] = -z[map_i[2]]
        else
            y[i] = z[map_i[3]] - z[map_i[2]]
        end
    end
    if !qp.meta.minimize
        y .*= -1
    end
    return GenericExecutionStats(
        qp,
        status = _status_symbol(solution.status, _clarabel_statuses),
        solution = x,
        objective = qp.meta.minimize ? solution.obj_val : -solution.obj_val,
        primal_feas = info.res_primal,
        dual_feas = info.res_dual,
        iter = Int64(solution.iterations),
        multipliers = y,
        elapsed_time = solution.solve_time,
    )
end

function BQMSolvers.clarabel(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.clarabel, bqp; kwargs...)
end

end
