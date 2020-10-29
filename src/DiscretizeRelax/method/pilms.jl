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
    fval::CircularBuffer{Vector{Float64}}
    fk_apriori::CircularBuffer{Vector{S}}
    Rk::Vector{S}
    Y0::Matrix{Float64}
    Y::Matrix{Float64}
    precond::Matrix{Float64}
    JxAff::Matrix{S}
    YJxAff::Matrix{S}
    YJxΔx::Vector{S}
    Xj_delta::Vector{S}
    Ysumx::Vector{S}
    YsumP::Matrix{S}
    YJpΔp::Vector{S}
    coeffs::Vector{Float64}
end

function AdamsMoultonFunctor(f::F, Jx!::JX, Jp!::JP, nx::Int, np::Int, s::S, t::T, steps::Int) where {F,JX,JP,S,T}
    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)

    method_step = steps
    Dk = zeros(S, nx)

    Y = zeros(nx, nx)
    Y0 = zeros(nx, nx)
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

    fval = CircularBuffer{Vector{Float64}}(method_step)
    fk_apriori = CircularBuffer{Vector{S}}(method_step)
    for i = 1:method_step
        push!(fval, zeros(Float64, nx))
        #$push!(f̃, zeros(S, nx))
        push!(fk_apriori, zeros(S, nx))
    end

    Rk = zeros(S, nx)
    coeffs = zeros(Float64, method_step + 1)

    AdamsMoultonFunctor{S}(f, Jx!, Jp!, nx, np, method_step, η, μX, ρP,
                           Dk, Jxsto, Jpsto, Jxsum, Jpsum, fval, fk_apriori, Rk,
                           Y0, Y, precond, JxAff, YJxAff, YJxΔx, Xj_delta,
                           Ysumx, YsumP, YJpΔp, coeffs)
end


function compute_Rk!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{T},
                     h::Float64, γ::Float64, s::Int) where T<:Number
    coeff = γ*h^(d.method_step + 1)
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
    for i = 1:(s + 1)
        @__dot__  d.Dk += (h^i)*d.coeffs[i]*d.fval[i]
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

    @__dot__ d.Y0 = mid(d.Jxsto[1])
    @__dot__ d.JxAff = d.Jxsto[1] - d.Y0

    d.sum_x = ((I - d.Jxsto[2])*contract.A.Q[1])*contract.Δ[1]
    for i = 3:s
        d.sum_x += (h*d.coeffs[i])*(d.Jxsto[i]*contract.A.Q[2])*contract.Δ[2]
    end
    @__dot__ d.sum_p = h*d.coeffs[1]*d.Jpsto[i]
    for i = 2:s
        @__dot__ d.sum_p += h*d.coeffs[i]*d.Jpsto[i]
    end

    return nothing
end

function compute_X!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}
    mul!(d.Jxvec, d.JxAff, d.Xj_delta)
    mul!(d.Jpvec, d.sum_p, contract.rP)
    @__dot__ contract.X_computed = contract.xval + d.Jxvec + d.sum_x + d.Jpvec
    @__dot__ contract.X_computed = contract.X_computed ∩ contract.Xj_0
    return nothing
end

function compute_xval!(contract::ContractorStorage{S}) where S
    @__dot__ contract.xval_computed = mid(contract.X_computed)
    return nothing
end

function compute_Ainv!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}
    mul!(d.Jxmid, d.Jxsto, contract.A[2].Q)
    @__dot__ contract.B = mid(d.Jxmid)
    calculateQ!(contract.A[1], contract.B, d.nx)
    calculateQinv!(contract.A[1])
    return nothing
end

function compute_Delta!(d::AdamsMoultonFunctor{T}, contract::ContractorStorage{S}) where {S, T<:Number}
    d.Y = I - Y0*contract.A[2].Q
    d.precond = lu!(d.Y)
    d.YJxAff = d.precond\d.JxAff
    mul!(d.YJxΔx, d.YJxAff, d.Xj_delta)
    d.Ysumx = d.precond\d.sum_x
    d.YsumP = d.precond\d.sum_p
    mul!(d.YJpΔp, d.YsumP, contract.rP)
    @__dot__ d.Δⱼ₊₁ = d.YJxΔx + d.Ysumx + d.YJpΔp
    return nothing
end

function (d::AdamsMoultonFunctor{T})(contract::ContractorStorage{S},
                                  result::StepResult{S},
                                  count::Int) where {S, T<:Number}

    t = 0.0
    γ = 1.0
    s = min(contract.step_count, d.method_step)
    h = 1.0

    # compute coefficients for linear multistep method
    #compute_coefficients!(d, h, t, s, adaptive_count, contract.is_adaptive)
    compute_Rk!(d, contract, h, γ, s)
    compute_real_sum!(d, contract, result, h, t, s)
    compute_jacobian_sum!(d, contract, h, t, s)
    compute_X!(d, contract)
    compute_xval!(contract)
    compute_Ainv!(d, contract)
    update_Delta!(d, contract)
    return nothing
end

function state_contractor(m::AdamsMoulton, f, Jx!, Jp!, nx, np, style, s, h)
    AdamsMoultonFunctor(f, Jx!, Jp!, nx, np, style, s, m.steps)
end
state_contractor_k(m::AdamsMoulton) = m.steps
state_contractor_γ(m::AdamsMoulton) = 0.0
state_contractor_steps(m::AdamsMoulton) = m.steps
state_contractor_integrator(m::AdamsMoulton) = CVODE_Adams()

function get_Δ(lf::AdamsMoulton) end
