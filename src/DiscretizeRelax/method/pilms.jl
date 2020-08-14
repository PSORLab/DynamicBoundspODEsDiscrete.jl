
struct PILMS <: AbstractStateContractorName
    k::Int
end

mutable struct PILMSFunctor{S} <: AbstractStateContractor
    f!
    Jx!
    Jp!
    nx::Int64
    np::Int64
    k::Int64
    η::Interval{Float64}
    μX::Vector{S}
    ρP::Vector{S}
    Δⱼ₊₁::Vector{S}
    f̃::Vector{Vector{S}}
    coeffs::Vector{Float64}
    Jxsto::Vector{Matrix{S}}
    Jpsto::Vector{Matrix{S}}
    Xj_delta::Vector{S}
    Dk::Vector{S}
    JxAff::Matrix{S}
    sum_x::Vector{S}
    sum_p::Matrix{S}
    Y::Matrix{Float64}
    Y0::Matrix{Float64}
    precond::Matrix{Float64}
    first_step::Bool
    c::Matrix{Float64}
    β::Vector{Float64}
    g::Vector{Float64}
    Φ::Vector{Float64}
    ϕ::Vector{Float64}
end

function PILMSFunctor(f!::F, nx::Int, np::Int, k::Val{K}, s::S, t::T) where {F, K, S <: Number, T <: Number}
    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    Δⱼ₊₁ = zeros(S, nx)

    fsto = Vector{Float64}[]
    f̃ = Vector{S}[]
    for i = 1:K
        push!(fsto, zeros(Float64, nx))
        push!(f̃, zeros(S, nx))
    end
    coeffs = zeros(Float64, k)

    Jxsto = Matrix{S}[]
    for i = 1:K
        push!(Jxsto, zeros(S, nx, nx))
    end
    Jpsto = Matrix{S}[]
    for i = 1:K
        push!(Jpsto, zeros(S, nx, np))
    end

    Xj_delta = zeros(S, nx)
    Dk = zeros(S, nx)
    JxAff = zeros(S, nx, nx)
    sum_x = zeros(S, nx)
    sum_p = zeros(S, nx, nx)

    Y = zeros(nx, nx)
    Y0 = zeros(nx, nx)
    precond = zeros(nx, nx)
    first_step = false

    c = zeros(Float64, k, k)
    β = zeros(Float64, k)
    g = zeros(Float64, k)
    Φ = zeros(Float64, k)
    ϕ = zeros(Float64, k)

    PILMSFunctor(f!, Jx!, Jp!, nx, np, k, η, μX, ρP, Δⱼ₊₁, f̃, coeffs, Jxsto, Jpsto,
                 Xj_delta, Dk, JxAff, sum_x, sum_p, Y, Y0, precond, first_step)
end

function update_coefficients!(contract::ContractorStorage{S}, h::Float64, t::Float64) where S

    if first_step
        pushfirst!(contract.times, t)
        pushfirst!(contract.steps, h)
    else
        contract.times[1] = t
        contract.steps[1] = h
    end

    return nothing
end

function (d::PILMSFunctor)(contract::ContractorStorage{S}, result::StepResult{S}) where S

    hⱼ = contract.hj_computed
    t = contract.times[1]
    update_coefficients!(contract, hⱼ, t)

    s = min(current_step, d.k)

    contract.xval = mid.(contract.Xj_0)
    Xj_delta = contract.Xj_0 - xval

    Dk = result.xⱼ - contract.xval + (d.coeffs[end]*hj^(k + 2))*d.f̃[k + 3]
    eval_cycle!(d.f!, d.fsto, contract.xval, contract.pval, t)
    for i = 1:s
        Dk += hj*coeff[i]*d.fsto[i]
    end

    μ!(d.μX, contract.Xj_0, contract.xval, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)

    eval_cycle!(d.Jx!, d.Jxsto, d.μX, d.ρP, t)
    eval_cycle!(d.Jp!, d.Jpsto, d.μX, d.ρP, t)

    Y0 .= mid.(d.Jxsto[1])
    JxAff = d.Jxsto[1] - Y0

    sum_x = ((I - d.Jxsto[2])*contract.A[1])*contract.Δ[1]
    for i = 3:s
        sum_x += (hj*coeff[i])*(d.Jxsto[i]*contract.A[2])*contract.Δ[2]
    end
    sum_p = hj*coeff[1]*d.Jpsto[i]
    for i = 2:s
        sum_p += hj*coeff[i]*d.Jpsto[i]
    end

    contract.X_computed = contract.xval + JxAff*Xj_delta + sum_x + sum_p*contract.rP
    contract.X_computed = contract.X_computed .∩ contract.Xj_0
    contract.xval_computed = mid.(contract.X_computed)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = contract.A[1]
    contract.B = mid.(Jf!.Jxsto*contract.A[2].Q)
    calculateQ!(Aⱼ₊₁, contract.B, nx)
    calculateQinv!(Aⱼ₊₁)

    Y .= I - Y0*contract.A[2].Q
    precond .= inv(Y)
    d.Δⱼ₊₁ = (precond*JxAff)*Xj_delta + precond*sum_x + (precond*sum_p)*contract.rP


    pushfirst!(contract.times, t)
    pushfirst!(contract.steps, hⱼ)
    pushfirst!(contract.Δ, d.Δⱼ₊₁)

    nothing
end

function state_contractor(m::PILMS, f, Jx!, Jp!, nx, np, style, s, h)
    PILMSFunctor(f, Jx!, Jp!, nx, np, Val(m.k), style, s)
end
state_contractor_k(m::PILMSFunctor) = m.k + 1
state_contractor_γ(m::PILMSFunctor) = 0.0
state_contractor_steps(m::PILMSFunctor) = k
