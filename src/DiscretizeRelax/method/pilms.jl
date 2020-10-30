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
    Jxsto::CircularBuffer{Matrix{S}}
    Jpsto::CircularBuffer{Matrix{S}}
    Jxsum::Vector{S}
    Jpsum::Matrix{S}
    Jxvec::Vector{S}
    Jpvec::Vector{S}
    fval::CircularBuffer{Vector{Float64}}
    fk_apriori::CircularBuffer{Vector{S}}
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
end

function AdamsMoultonFunctor(f::F, Jx!::JX, Jp!::JP, nx::Int, np::Int, s::S, t::T, steps::Int) where {F,JX,JP,S,T}
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

    Jxsto = CircularBuffer{Matrix{S}}(method_step)
    Jpsto = CircularBuffer{Matrix{S}}(method_step)
    for i = 1:method_step
        push!(Jxsto, zeros(S, nx, nx))
        push!(Jpsto, zeros(S, nx, np))
    end
    Jxsum = zeros(S, nx)
    Jpsum = zeros(S, nx, nx)
    Jxvec = zeros(S, nx)
    Jpvec = zeros(S, nx)

    fval = CircularBuffer{Vector{Float64}}(method_step)
    fk_apriori = CircularBuffer{Vector{S}}(method_step)
    for i = 1:method_step
        push!(fval, zeros(Float64, nx))
        #$push!(f̃, zeros(S, nx))
        push!(fk_apriori, zeros(S, nx))
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
                           is_adaptive, γ)
end

function compute_coefficients!(d::AdamsMoultonFunctor{S}, h::Float64, t::Float64, s::Int) where S
    if !d.is_adaptive
        if s == 0
            d.coeffs[1] = 1.0
        elseif s == 1
            d.coeffs[1] = 0.5
            d.coeffs[2] = 0.5
        elseif s == 2
            d.coeffs[1] = 5.0/12.0
            d.coeffs[2] = 2.0/3.0
            d.coeffs[3] = -1.0/12.0
        elseif s == 3
            d.coeffs[1] = 9.0/24.0
            d.coeffs[2] = 19.0/24.0
            d.coeffs[3] = -5.0/24.0
            d.coeffs[4] = 1.0/24.0
        elseif s == 4
            d.coeffs[1] = 251.0/720.0
            d.coeffs[2] = 646.0/720.0
            d.coeffs[3] = -264.0/720.0
            d.coeffs[4] = 106.0/720.0
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
    coeff = d.γ*h^(d.method_step + 1)
    pushfirst!(d.fk_apriori, contract.fk_apriori)
    @__dot__ d.Rk = d.fk_apriori[1]
    for i = 2:s
        @__dot__ d.Rk = d.Rk ∪ d.fk_apriori[i]
    end
    @__dot__ d.Rk *= coeff
    return nothing
end

function compute_real_sum!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{T},
                           result::StepResult{T}, h::Float64, t::Float64,
                           s::Int) where T<:Number
    eval_cycle!(d.f!, d.fval, contract.xval, contract.pval, t)
    @__dot__ d.Dk = result.xⱼ + (d.coeffs[s + 1]*h^(s + 2))*contract.fk_apriori
    for i = 1:s
        @__dot__ d.Dk += (h^i)*d.coeffs[i]*d.fval[i]
    end
    return nothing
end

function compute_jacobian_sum!(d::AdamsMoultonFunctor{T},
                           contract::ContractorStorage{T},
                           h::Float64, t::Float64, s::Int) where T<:Number

    μ!(d.μX, contract.Xj_0, contract.xval, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)
    eval_cycle!(d.Jx!, d.Jxsto, d.μX, d.ρP, t)
    eval_cycle!(d.Jp!, d.Jpsto, d.μX, d.ρP, t)

    @show d.Jxsto
    @show d.Jpsto

    @__dot__ d.Y0 = mid(d.Jxsto[1])
    @__dot__ d.JxAff = d.Jxsto[1] - d.Y0

    d.Jxsum = ((I - d.Jxsto[2])*contract.A[1].Q)*contract.Δ[1]
    for i = 2:s
        d.Jxsum += (h*d.coeffs[i])*(d.Jxsto[i]*contract.A[2].Q)*contract.Δ[2]
    end
    @__dot__ d.Jpsum = h*d.coeffs[1]*d.Jpsto[1]
    for i = 2:s
        @__dot__ d.Jpsum += h*d.coeffs[i]*d.Jpsto[i]
    end

    return nothing
end

function compute_X!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}

    mul!(d.Jxvec, d.JxAff, d.Xj_delta)
    mul!(d.Jpvec, d.Jpsum, contract.rP)
    @__dot__ contract.X_computed = contract.xval + d.Jxvec + d.Jxsum + d.Jpvec
    @show contract.X_computed, contract.Xj_0, d.Jxvec, d.Jxsum, d.Jpvec

    @__dot__ contract.X_computed = contract.X_computed ∩ contract.Xj_0
    return nothing
end

function compute_xval!(contract::ContractorStorage{S}) where S
    @__dot__ contract.xval_computed = mid(contract.X_computed)
    @show "compute_xval!", contract.xval_computed, contract.X_computed
    return nothing
end

function compute_Ainv!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}
    mul!(d.Jxmid_sto, d.Jxsto[1], contract.A[2].Q)
    @__dot__ contract.B = mid(d.Jxmid_sto)
    calculateQ!(contract.A[1], contract.B, d.nx)
    calculateQinv!(contract.A[1])
    return nothing
end

function update_delta!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}
    d.Y = I - d.Y0*contract.A[2].Q
    d.precond = lu!(d.Y)
    d.YJxAff = d.precond\d.JxAff
    mul!(d.YJxΔx, d.YJxAff, d.Xj_delta)
    d.Ysumx = d.precond\d.Jxsum
    d.YsumP = d.precond\d.Jpsum
    mul!(d.YJpΔp, d.YsumP, contract.rP)
    @__dot__ d.Δⱼ₊₁ = d.YJxΔx + d.Ysumx + d.YJpΔp
    return nothing
end

function (d::AdamsMoultonFunctor{T})(contract::ContractorStorage{S},
                                  result::StepResult{S},
                                  count::Int) where {S, T<:Number}

    d.is_adaptive = contract.is_adaptive

    t = contract.times[1]
    h = contract.hj_computed
    s = min(contract.step_count, d.method_step)

    @show " "
    @show " "

    # compute coefficients for linear multistep method
    compute_coefficients!(d, h, t, s)
    compute_Rk!(d, contract, h, s)
    compute_real_sum!(d, contract, result, h, t, s)
    compute_jacobian_sum!(d, contract, h, t, s)
    compute_X!(d, contract)
    compute_xval!(contract)
    compute_Ainv!(d, contract)
    update_delta!(d, contract)
    return nothing
end

get_Δ(d::AdamsMoultonFunctor) = d.Δⱼ₊₁

function state_contractor(m::AdamsMoulton, f, Jx!, Jp!, nx, np, style, s, h)
    AdamsMoultonFunctor(f, Jx!, Jp!, nx, np, style, s, m.steps)
end
state_contractor_k(m::AdamsMoulton) = m.steps
state_contractor_γ(m::AdamsMoulton) = 0.0
state_contractor_steps(m::AdamsMoulton) = m.steps
state_contractor_integrator(m::AdamsMoulton) = CVODE_Adams()
