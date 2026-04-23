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

function _clarabel_hessian(QM)
    I = Int[]
    J = Int[]
    V = Float64[]
    for k in eachindex(QM.data.H.vals)
        i = Int(QM.data.H.rows[k])
        j = Int(QM.data.H.cols[k])
        row, col = i < j ? (i, j) : (j, i)
        push!(I, row)
        push!(J, col)
        push!(V, Float64(QM.data.H.vals[k]))
    end
    return sparse(I, J, V, QM.meta.nvar, QM.meta.nvar)
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

function BQMSolvers.clarabel(QM::QuadraticModel{T,S}; kwargs...) where {T,S}
    return BQMSolvers.clarabel(_coo_model(QM); kwargs...)
end

function BQMSolvers.clarabel(
    QM::QuadraticModel{T,S,M1,M2};
    kwargs...,
) where {T,S,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}
    length(QM.meta.jinf) == 0 || error("infeasible bound metadata is unsupported here")
    row_terms = [Tuple{Int,Float64}[] for _ in 1:QM.meta.ncon]
    for k in eachindex(QM.data.A.vals)
        push!(row_terms[Int(QM.data.A.rows[k])], (Int(QM.data.A.cols[k]), Float64(QM.data.A.vals[k])))
    end
    I = Int[]
    J = Int[]
    V = Float64[]
    b = Float64[]
    cones = Clarabel.SupportedCone[]
    for i in 1:QM.meta.nvar
        if isfinite(QM.meta.lvar[i])
            _push_bound_row!(I, J, V, b, cones, [(i, 1.0)], -1.0, -Float64(QM.meta.lvar[i]))
        end
        if isfinite(QM.meta.uvar[i])
            _push_bound_row!(I, J, V, b, cones, [(i, 1.0)], 1.0, Float64(QM.meta.uvar[i]))
        end
    end
    row_map = Vector{Any}(undef, QM.meta.ncon)
    for i in 1:QM.meta.ncon
        l = QM.meta.lcon[i]
        u = QM.meta.ucon[i]
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
    A = sparse(I, J, V, length(b), QM.meta.nvar)
    q = Float64.(QM.data.c)
    P = _clarabel_hessian(QM)
    c0 = Float64(QM.data.c0)
    sense = QM.meta.minimize ? 1.0 : -1.0
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
    y = fill(NaN, QM.meta.ncon)
    for i in 1:QM.meta.ncon
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
    if !QM.meta.minimize
        y .*= -1
    end
    return GenericExecutionStats(
        QM,
        status = _status_symbol(solution.status, _clarabel_statuses),
        solution = x,
        objective = QM.meta.minimize ? solution.obj_val : -solution.obj_val,
        primal_feas = info.res_primal,
        dual_feas = info.res_dual,
        iter = Int64(solution.iterations),
        multipliers = y,
        elapsed_time = solution.solve_time,
    )
end

# Threaded batch dispatch. Each task extracts a scalar view of its instance
# (zero-alloc, shares A/Q triplets) and calls the scalar solver above. Pass
# `threads=1` / `Threads=1` in kwargs to single-thread the inner solver so
# outer Julia threads aren't fighting it for CPUs. `schedule = :dynamic`
# helps when per-instance solve times are very uneven.
function BQMSolvers.clarabel(bqp::BatchQuadraticModels.BatchQuadraticModel{T, MT, VT}; kwargs...) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BQMSolvers.solve_batch_threaded(BQMSolvers.clarabel, bqp; kwargs...)
end

end
