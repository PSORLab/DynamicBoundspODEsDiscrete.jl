
struct AdamsMoulton <: AbstractStateContractorName
    k::Int
end

mutable struct AMFunctor{S} <: AbstractStateContractor
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
    fsto::CircularBuffer{Vector{Float64}}
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
    step_count::Int64
    is_fixed::Bool
    c::Matrix{Float64}
    β::Vector{Float64}
    g::Vector{Float64}
    Φ::Vector{Float64}
    ϕ::Vector{Float64}
    poly_buffer1::Polynomial{Float64}
    poly_buffer2::Polynomial{Float64}
    fk_apriori::Vector{Vector{S}}
    Rk::Vector{S}
end

function AMFunctor(f!::F, Jx!, Jp!, nx::Int, np::Int, k::Val{K},
                      s::S, t::T) where {F, K, S <: Number, T <: Number}

    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    Δⱼ₊₁ = zeros(S, nx)

    fsto = Vector{Float64}[]
    f̃ = Vector{S}[]
    fk_apriori = Vector{S}[]
    for i = 1:K
        push!(fsto, zeros(Float64, nx))
        push!(f̃, zeros(S, nx))
        push!(fk_apriori, zeros(S, nx))
    end
    coeffs = zeros(Float64, K + 1)

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

    step_count = 0
    is_fixed = false

    c = zeros(Float64, K + 1, K + 1)
    β = zeros(Float64, K + 1)
    g = zeros(Float64, K + 1)
    Φ = zeros(Float64, K + 1)
    ϕ = zeros(Float64, K + 1)

    poly_buffer1 = Polynomial(zeros(1))
    poly_buffer2 = Polynomial(zeros(1))

    Rk = zeros(S, nx)

    AMFunctor{S}(f!, Jx!, Jp!, nx, np, K, η, μX, ρP, Δⱼ₊₁,
                 fsto, f̃, coeffs, Jxsto, Jpsto, Xj_delta, Dk, JxAff,
                 sum_x, sum_p, Y, Y0, precond, step_count,
                 is_fixed, c, β, g, Φ, ϕ, poly_buffer1, poly_buffer2,
                 fk_apriori, Rk)
end

function update_coefficients!(v::Val{false}, d::AMFunctor, contract::ContractorStorage{S}) where S

    ts = contract.times
    n = d.k + 1

    d.β[1] = 1.0
    for i = 2:n
        d.β[i] = d.β[i - 1]*(ts[1] - ts[1 + i])/(ts[2] - ts[2 + i])
    end

    for q = 1:n
        d.c[1,q] = 1/q
    end

    for q = 1:(n-1)
        for j = 2:n
            d.c[j, q] = d.c[j - 1, q] - d.c[j - 1, q + 1]*contract.steps[1]/(ts[1] - ts[1 + i])
        end
    end

    for j = 1:n
        d.g[j] = d.c[j,1]
    end

    for j = 1:n
        coeffs[j] = d.g[j]*d.β[j]
    end

    return nothing
end

function mul_poly!(pout::Polynomial{Float64}, pin::Polynomial{Float64}, q::Int64, k::Int64)
    if k > 0
        fill!(pout.coeffs, 0.0)
        for i in 0:k, j in 0:1
            @inbounds pout.coeffs[i + j + 1] += pin[i]*(iszero(j) ? (q - 1.0) : 1.0)
        end
    else
        fill!(pout.coeffs, 0.0)
        pout.coeffs[1] = pin.coeffs[1]*(q - 1.0)
        pout.coeffs[2] = pin.coeffs[1]
    end

    return nothing
end

function update_coefficients!(v::Val{true}, d::AMFunctor, contract::ContractorStorage{S}) where S

    last_buffer_is1 = true
    for j = 0:d.k

        # initialize polynomial
        fill!(d.poly_buffer1.coeffs, 0.0)
        d.poly_buffer1.coeffs[1] = 1.0
        last_buffer_is1 = true

        # compute polynomial as a product
        for i = 0:d.k
            if i !== j
                if last_buffer_is1
                    mul_poly!(d.poly_buffer2, d.poly_buffer1, i, d.k)
                else
                    mul_poly!(d.poly_buffer1, d.poly_buffer2, i, d.k)
                end
            end
        end

        # integrate the buffer and store the coefficient
        if last_buffer_is1
            d.poly_buffer1 = Polynomials.integrate(d.poly_buffer1)
        else
            d.poly_buffer1 = Polynomials.integrate(d.poly_buffer2)
        end
        d.coeffs[j + 1] = (d.poly_buffer1(1.0) - d.poly_buffer1(0.0))
        d.coeffs[j + 1] *= ((-1)^j)/(factorial(j)*factorial(d.k - j))
    end

    return nothing
end

function compute_Rk!(d::AMFunctor{T}, fk_apriori::Vector{T}, h::Float64) where T
    pushfirst!(d.fk_apriori, fk_apriori)
    d.Rk .= d.fk_apriori[1]
    for i = 2:d.k
        d.Rk .= d.Rk .∩ d.fk_apriori[i]
    end
    coeff = h^(d.k + 2)
    return coeff*d.Rk
end

function (d::AMFunctor)(contract::ContractorStorage{S}, result::StepResult{S}) where S

    hⱼ = contract.hj_computed
    t = contract.times[1]
    update_coefficients!(Val(true), d, contract)
    compute_Rk!(d, contract.fk_apriori, hⱼ)

    s = min(contract.step_count, d.k)

    contract.xval = mid.(contract.Xj_0)
    Xj_delta = contract.Xj_0 - contract.xval

    Dk = result.xⱼ - contract.xval + (d.coeffs[end]*hⱼ^(s + 1))*d.f̃[s + 1]
    eval_cycle!(d.f!, d.fsto, contract.xval, contract.pval, t)
    for i = 1:s
        Dk += hⱼ*coeff[i]*d.fsto[i]
    end

    μ!(d.μX, contract.Xj_0, contract.xval, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)

    eval_cycle!(d.Jx!, d.Jxsto, d.μX, d.ρP, t)
    eval_cycle!(d.Jp!, d.Jpsto, d.μX, d.ρP, t)

    Y0 .= mid.(d.Jxsto[1])
    JxAff = d.Jxsto[1] - Y0

    sum_x = ((I - d.Jxsto[2])*contract.A[1])*contract.Δ[1]
    for i = 3:s
        sum_x += (hⱼ*coeff[i])*(d.Jxsto[i]*contract.A[2])*contract.Δ[2]
    end
    sum_p = hⱼ*coeff[1]*d.Jpsto[i]
    for i = 2:s
        sum_p += hⱼ*coeff[i]*d.Jpsto[i]
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

function state_contractor(m::AdamsMoulton, f, Jx!, Jp!, nx, np, style, s, h)
    AMFunctor(f, Jx!, Jp!, nx, np, Val(m.k), style, s)
end
state_contractor_k(m::AdamsMoulton) = m.k + 1
state_contractor_γ(m::AdamsMoulton) = 0.0
state_contractor_steps(m::AdamsMoulton) = m.k
