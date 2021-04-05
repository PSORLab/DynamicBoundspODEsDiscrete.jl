export AdamsMoulton

struct AdamsMoulton <: AbstractStateContractorName
    steps::Int
end

mutable struct AdamsMoultonFunctor{S} <: AbstractStateContractor
    f!
    Jx!
    Jp!
    nx
    np
    method_step::Int64
    η::Interval{Float64}
    μX::Vector{S}
    ρP::Vector{S}
    Dk::Vector{S}
    Jxsto::FixedCircularBuffer{Matrix{S}}
    Jpsto::FixedCircularBuffer{Matrix{S}}
    Jxsum::Vector{S}
    Jpsum::Matrix{S}
    Jxvec::Vector{S}
    Jpvec::Vector{S}
    fval::FixedCircularBuffer{Vector{Float64}}
    fk_apriori::FixedCircularBuffer{Vector{S}}
    Rk::Vector{S}
    Y0::Matrix{Float64}
    Y::Matrix{Float64}
    Jxmid_sto::Matrix{S}
    precond::LinearAlgebra.LU{Float64,Array{Float64,2}}
    JxAff::Matrix{S}
    YJxAff::Matrix{S}
    YJxΔx::Vector{S}
    Xj_delta::Vector{S}
    Ysumx::Vector{S}
    YsumP::Matrix{S}
    YJpΔp::Vector{S}
    coeffs::Vector{Float64}
    Δⱼ₊₁::Vector{S}
    is_adaptive::Bool
    γ::Float64
    lohners_start::LohnersFunctor
    A_Q::FixedCircularBuffer{Matrix{Float64}}
    A_inv::FixedCircularBuffer{Matrix{Float64}}
    Δ::FixedCircularBuffer{Vector{S}}
    X::FixedCircularBuffer{Vector{S}}
    xval::FixedCircularBuffer{Vector{Float64}}
end

function AdamsMoultonFunctor(f::F, Jx!::JX, Jp!::JP, nx::Int, np::Int, s::S,
                             t::T, steps::Int, lohners_start) where {F,JX,JP,S,T}
    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)

    method_step = steps
    Dk = zeros(S, nx)

    lu_mat = zeros(nx, nx)
    for i = 1:nx
        lu_mat[i,i] = 1.0
    end
    precond = lu(lu_mat)
    Y0 = zeros(nx, nx)
    Y = zeros(nx, nx)
    Jxmid_sto = zeros(S, nx, nx)
    YJxAff = zeros(S, nx, nx)
    JxAff = zeros(S, nx, nx)
    Xj_delta = zeros(S, nx)
    YJxΔx = zeros(S, nx)
    Ysumx = zeros(S, nx)
    YsumP = zeros(S, nx, nx)
    YJpΔp = zeros(S, nx)

    Jxsto = FixedCircularBuffer{Matrix{S}}(method_step)
    Jpsto = FixedCircularBuffer{Matrix{S}}(method_step)
    for i = 1:method_step
        push!(Jxsto, zeros(S, nx, nx))
        push!(Jpsto, zeros(S, nx, np))
    end
    Jxsum = zeros(S, nx)
    Jpsum = zeros(S, nx, nx)
    Jxvec = zeros(S, nx)
    Jpvec = zeros(S, nx)

    fval = FixedCircularBuffer{Vector{Float64}}(method_step)
    fk_apriori = FixedCircularBuffer{Vector{S}}(method_step)
    A_Q = FixedCircularBuffer{Matrix{Float64}}(method_step)
    A_inv = FixedCircularBuffer{Matrix{Float64}}(method_step)
    Δ = FixedCircularBuffer{Vector{S}}(method_step)
    X = FixedCircularBuffer{Vector{S}}(method_step)
    xval = FixedCircularBuffer{Vector{Float64}}(method_step)
    for i = 1:method_step
        push!(xval, zeros(Float64, nx))
        push!(fval, zeros(Float64, nx))
        push!(fk_apriori, zeros(S, nx))
        push!(A_Q, Float64.(Matrix(I, nx, nx)))
        push!(A_inv, Float64.(Matrix(I, nx, nx)))
        push!(X, zeros(S, nx))
        push!(Δ, zeros(S, nx))
    end

    Rk = zeros(S, nx)
    coeffs = zeros(Float64, method_step + 1)
    Δⱼ₊₁ = zeros(S, nx)

    is_adaptive = false
    γ = 0.0

    AdamsMoultonFunctor{S}(f, Jx!, Jp!, nx, np, method_step, η, μX, ρP,
                           Dk, Jxsto, Jpsto, Jxsum, Jpsum, Jxvec, Jpvec, fval,
                           fk_apriori, Rk, Y0, Y, Jxmid_sto, precond, JxAff, YJxAff,
                           YJxΔx, Xj_delta, Ysumx, YsumP, YJpΔp, coeffs, Δⱼ₊₁,
                           is_adaptive, γ, lohners_start, A_Q, A_inv,
                           Δ, X, xval)
end

function compute_coefficients!(d::AdamsMoultonFunctor{S}, h::Float64, t::Float64, s::Int) where S
    if !d.is_adaptive
        if s == 1
            d.coeffs[1] = 1.0
            d.coeffs[2] = 0.5
        elseif s == 2
            d.coeffs[1] = 0.5
            d.coeffs[2] = 0.5
            d.coeffs[3] = -1.0/12.0
        elseif s == 3
            d.coeffs[1] = 5.0/12.0
            d.coeffs[2] = 8.0/12.0
            d.coeffs[3] = -1.0/12.0
            d.coeffs[4] = -1.0/24.0
        elseif s == 4
            d.coeffs[1] = 9.0/24.0
            d.coeffs[2] = 19.0/24.0
            d.coeffs[3] = -5.0/24.0
            d.coeffs[4] = 1.0/24.0
            d.coeffs[5] = -19.0/720.0
        else
            #compute_fixed_higher_coeffs!(d, s)
        end
    else
        #compute_adaptive_coeffs!(d, h, t, s)
    end

    d.γ = d.coeffs[s + 1]

    return nothing
end

function compute_Rk!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{T},
                     h::Float64, s::Int) where T<:Number

    #println("------------")
    #println("-compute_Rk-")
    #println("------------")

    coeff = d.γ*h^(d.method_step + 1)
    cycle_copyto!(d.fk_apriori, contract.fk_apriori, contract.step_count - 1)

    @__dot__ d.Rk = d.fk_apriori[1]
    #println("d.Rk[1] = $(d.Rk)")
    for i = 2:s
        @__dot__ d.Rk = d.Rk ∪ d.fk_apriori[i]
        #println("d.fk_apriori[$i] = $(d.fk_apriori[i])")
        #println(" d.Rk[$i] = $(d.Rk)")
    end
    #@show d.γ
    #@show h^(d.method_step + 1)
    @__dot__ d.Rk *= coeff

    return nothing
end

# TODO: Check block
function compute_real_sum!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{T},
                           result::StepResult{T}, h::Float64, t::Float64,
                           s::Int) where T<:Number

    old_xval = mid.(contract.Xj_0)
    cycle_eval!(d.f!, d.fval, old_xval, contract.pval, t)
    @__dot__ d.Dk = old_xval
    #@show old_xval
    for i = 1:s
        @__dot__ d.Dk += h*d.coeffs[i]*d.fval[i]
        #println("$i-th coeffs = $(d.coeffs[i]), fvals = $(d.fval[i])")
    end
    return nothing
end

function compute_jacobian_sum!(d::AdamsMoultonFunctor{T},
                           contract::ContractorStorage{T},
                           h::Float64, t::Float64, s::Int) where T<:Number

    μ!(d.μX, contract.Xj_0, contract.xval, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)
    cycle_eval!(d.Jx!, d.Jxsto, d.μX, d.ρP, t)
    cycle_eval!(d.Jp!, d.Jpsto, d.μX, d.ρP, t)

    #println("------------")
    #println("-compute_jacobian_sum!-")
    #println("------------")
    #@show d.Jxsto
    #@show d.Jpsto

    @__dot__ d.JxAff = d.Jxsto[1]

    #@show d.JxAff

    # contract.Xj_apriori GOOD
    # d.Xj_delta GOOD
    @__dot__ d.Xj_delta = contract.Xj_apriori - mid(contract.Xj_apriori)

    # TODO: CHECK RIGHT DELTAS...
    d.Jxsum = (h*d.coeffs[1])*d.Jxsto[1]*d.Xj_delta
    @show h, d.coeffs[1], d.Jxsto[1], d.Xj_delta
    if s > 1
        d.Jxsum += ((d.η*I + h*d.coeffs[2]*d.Jxsto[2])*contract.A_Q[1])*contract.Δ[1]
        @show h, d.coeffs[2], d.Jxsto[2], contract.A_Q[1], contract.Δ[1]
    end
    for i = 3:s
        d.Jxsum += (h*d.coeffs[i])*(d.Jxsto[i]*contract.A_Q[i])*contract.Δ[i]
        #@show h, d.coeffs[i], d.Jxsto[i], contract.A_Q[i], contract.Δ[i]
    end

    # JpSum is correct
    @__dot__ d.Jpsum = h*d.coeffs[1]*d.Jpsto[1]
    for i = 2:s
        @__dot__ d.Jpsum += h*d.coeffs[i]*d.Jpsto[i]
    end

    return nothing
end

function compute_X!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}

    #println("------------")
    #println("-compute_X!-")
    #println("------------")

    #mul!(d.Jxvec, d.JxAff, d.Xj_delta)
    mul!(d.Jpvec, d.Jpsum, contract.rP)

    #@show d.Jxvec
    @show d.Jxsum
    # d.Jpvec is correct...
    @show d.Jpvec
    @show d.Dk
    @show d.Rk
    #@__dot__ contract.X_computed = d.Jxvec + d.Jxsum + d.Jpvec + d.Dk + d.Rk
    @__dot__ contract.X_computed = d.Jxsum + d.Jpvec + d.Dk + d.Rk
    @show contract.X_computed
    @show contract.Xj_apriori
    @__dot__ contract.X_computed = contract.X_computed ∩ contract.Xj_apriori
    @show contract.X_computed
    return nothing
end

function compute_xval!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}, t) where {S, T<:Number}
    @__dot__ contract.xval_computed = mid(contract.X_computed)
    @show contract.xval_computed
    d.f!(d.fval[1], contract.xval_computed, contract.pval, t)
    return nothing
end

function compute_Ainv!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}, s::Int) where {S, T<:Number}
    if s > 1
        mul!(d.Jxmid_sto, d.Jxsto[1], contract.A_Q[2])
        @__dot__ contract.B = mid(d.Jxmid_sto)
        calculateQ!(contract.A_Q[1], contract.B, d.nx)
        calculateQinv!(contract.A_inv[1], contract.A_Q[1], d.nx)
    end
    return nothing
end

function update_delta!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}, s::Int) where {S, T<:Number}
    println("update delta for s = $s order")
    if s > 0
        println("ran delta update...")
        @show d.Y0
        @show contract.A_Q[2]
        d.Y = I - d.Y0*contract.A_Q[2]
        @show d.Y
        d.precond = lu!(d.Y)
        d.YJxAff = d.precond\d.JxAff
        @show d.YJxAff
        mul!(d.YJxΔx, d.YJxAff, d.Xj_delta)
        @show d.YJxΔx
        d.Ysumx = d.precond\d.Jxsum
        d.YsumP = d.precond\d.Jpsum
        @show d.Ysumx
        @show d.YsumP
        mul!(d.YJpΔp, d.YsumP, contract.rP)
        @show d.YJpΔp
        @__dot__ d.Δⱼ₊₁ = d.YJxΔx + d.Ysumx + d.YJpΔp
        @show d.Δⱼ₊₁
    end
    return nothing
end


function store_starting_buffer!(d::AdamsMoultonFunctor{T},
                                contract::ContractorStorage{T},
                                result::StepResult{T}, t::Float64) where T

    count = contract.step_count - 1
    cycle_copyto!(d.X, result.Xⱼ, count)
    cycle_copyto!(d.xval, result.xⱼ, count)
    cycle_copyto!(d.fk_apriori, contract.fk_apriori, count)
    cycle_copyto!(d.Δ, result.Δ[1], count)
    cycle_copyto!(d.A_Q, result.A_Q[1], count)
    cycle_copyto!(d.A_inv, result.A_inv[1], count)

    # update Jacobian storage
    μ!(d.μX, result.Xⱼ, result.xⱼ, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)

    cycle_eval!(d.Jx!, d.Jxsto, d.μX, d.ρP, t)
    cycle_eval!(d.Jp!, d.Jpsto, d.μX, d.ρP, t)
    cycle_eval!(d.f!, d.fval, result.xⱼ, contract.pval, t)

    return nothing
end

function (d::AdamsMoultonFunctor{T})(contract::ContractorStorage{S},
                                  result::StepResult{S},
                                  count::Int) where {S, T<:Number}


    println("\n BEGIN MAIN - step: $(contract.step_count-1), method step: $(d.method_step) \n")

    t = contract.times[1]
    s = min(contract.step_count-1, d.method_step)
    #@show s
    #@show d.method_step
    if s < d.method_step - 1
        #println("---- LOHNERS STEP -----")
        d.lohners_start(contract, result, count)
        (count == 0) && store_starting_buffer!(d, contract, result, t)
        return nothing
    end

    #println("---- PILMS STEP -----")
    h = contract.hj_computed
    #@show contract.hj_computed
    compute_coefficients!(d, h, t, s)
    compute_Rk!(d, contract, h, s)
    compute_real_sum!(d, contract, result, h, t, s)
    compute_jacobian_sum!(d, contract, h, t, s)
    compute_X!(d, contract)
    compute_xval!(d, contract, t)
    compute_Ainv!(d, contract, s)
    update_delta!(d, contract, s)
    return nothing
end

function get_Δ(d::AdamsMoultonFunctor)
    if true
        return copy(get_Δ(d.lohners_start))
    end
    return copy(d.Δⱼ₊₁)
end
function advance_contractor_buffer!(d::AdamsMoultonFunctor)
    return nothing
end

function state_contractor(m::AdamsMoulton, f, Jx!, Jp!, nx, np, style, s, h)
    lohners_functor = state_contractor(LohnerContractor{m.steps}(), f, Jx!, Jp!, nx, np, style, s, h)
    AdamsMoultonFunctor(f, Jx!, Jp!, nx, np, style, s, m.steps, lohners_functor)
end
state_contractor_k(m::AdamsMoulton) = m.steps + 1
state_contractor_γ(m::AdamsMoulton) = 0.0
state_contractor_steps(m::AdamsMoulton) = m.steps
state_contractor_integrator(m::AdamsMoulton) = CVODE_Adams()
