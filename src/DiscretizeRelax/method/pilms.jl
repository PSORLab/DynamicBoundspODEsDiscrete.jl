
struct AdamsMoulton <: AbstractStateContractorName
    k::Int
end

mutable struct AMFunctor{S,NP} <: AbstractStateContractor
    f!
    Jx!
    Jp!
    nx::Int64
    np::Int64
    method_step::Int64
    η::Interval{Float64}
    μX::Vector{S}
    ρP::Vector{S}
    Δⱼ₊₁::Vector{S}
    fval::CircularBuffer{Vector{Float64}}
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
    Θ::Vector{SVector{NP,Float64}}
    Θstar::Vector{SVector{NP,Float64}}
    poly_buffer1::Polynomial{Float64}
    poly_buffer2::Polynomial{Float64}
    fk_apriori::Vector{Vector{S}}
    Rk::Vector{S}
end

function AMFunctor(f!::F, Jx!, Jp!, nx::Int, np::Int, method_step::Val{K},
                      s::S, t::T) where {F, K, S <: Number, T <: Number}

    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    Δⱼ₊₁ = zeros(S, nx)

    fval = CircularBuffer{Vector{Float64}}(4)
    f̃ = Vector{S}[]
    fk_apriori = Vector{S}[]
    for i = 1:K
        push!(fval, zeros(Float64, nx))
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
    Θ = zeros(SVector{np,Float64}, K + 1)
    Θstar = zeros(SVector{np,Float64}, K + 1)

    poly_buffer1 = Polynomial(zeros(1))
    poly_buffer2 = Polynomial(zeros(1))

    Rk = zeros(S, nx)

    AMFunctor{S, np}(f!, Jx!, Jp!, nx, np, K, η, μX, ρP, Δⱼ₊₁,
                 fval, f̃, coeffs, Jxsto, Jpsto, Xj_delta, Dk, JxAff,
                 sum_x, sum_p, Y, Y0, precond, step_count,
                 is_fixed, c, β, g, Φ, ϕ, poly_buffer1, poly_buffer2,
                 fk_apriori, Rk)
end

function compute_Rk!(d::AMFunctor{T}, h::Float64) where T<:Number
    coeff = h^(d.method_step + 1)
    @__dot__ d.Rk = d.fk_apriori[1] ∪ d.Rk_temp
    @__dot__ d.Rk *= coeff
    return nothing
end

function mul_poly!(pout::Polynomial{Float64}, pin::Polynomial{Float64}, q::Int64, s::Int64)
    if s > 0
        fill!(pout.coeffs, 0.0)
        for i in 0:s, j in 0:1
            @inbounds pout.coeffs[i + j + 1] += pin[i]*(iszero(j) ? (q - 1.0) : 1.0)
        end
    else
        fill!(pout.coeffs, 0.0)
        pout.coeffs[1] = pin.coeffs[1]*(q - 1.0)
        pout.coeffs[2] = pin.coeffs[1]
    end

    return nothing
end

function compute_fixed_higher_coeffs!(d::AMFunctor, s::Int)

    last_buffer_is1 = true

    for j = 0:s
        # initialize polynomial
        fill!(d.poly_buffer1.coeffs, 0.0)
        d.poly_buffer1.coeffs[1] = 1.0
        last_buffer_is1 = true

        # compute polynomial as a product
        for i = 0:s
            if i !== j
                if last_buffer_is1
                    mul_poly!(d.poly_buffer2, d.poly_buffer1, i, s)
                else
                    mul_poly!(d.poly_buffer1, d.poly_buffer2, i, s)
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
        d.coeffs[j + 1] *= ((-1)^j)/(factorial(j)*factorial(d.method_step - j))
    end

    return nothing
end

function seed_statics!(Θ::Matrix{SVector{N,Int}}, Θs::Matrix{SVector{N,Int}}) where N
    for i = 1:(s+1)
        Θ[1,i] = seed_gradient(i, Val{N}())
        Θs[1,i] = seed_gradient(i, Val{N}())
    end
    return nothing
end

function compute_adaptive_coeffs!(d::AMFunctor, h::Float64, t::Float64, s::Int)

    current_step = h

    # compute Beta
    d.β[1] = 1.0
    for i = 2:s
        βfrac = @inbounds (d.t[1] - d.t[1 + i])/(d.t[2] - d.t[2 + i])
        @inbounds d.β[i] = βfrac*d.β[i - 1]
    end

    # compute statics for Θ and Θstar
    seed_statics!(d.Θ, d.Θstar)
    for j = 1:s
        for i = 1:s
            d.Θ[j+1,i+1] = d.Θ[j,i+1] - d.Θstar[j,i]
        end
        d.Θstar[j,n] = d.β[j]*d.Θ[j,n]
    end
    #d.Θn = # to do
    #d.Θk = # to do

    # compute c
    for q = 1:s
        d.c[1,q] = 1/q
        d.c[2,q] = 1/(q*(q+1))
    end
    for j = 0:s, q = 1:(s - j)
        d.c[j,q] = d.c[j-1,q] + d.c[j-1,q+1]*current_step/(d.t[1] - d.t[j+1])
    end
    for j = 0:s
        d.g[j] = d.c[j,1]
    end

    # compute coefficients
    for q = 1:s
        d.coeffs[q] = current_step*d.g[d.method_step+1]*d.Θk[q]
    end
    for i = 1:s, q = 1:s
        d.coeffs[q] += current_step*d.g[i]*d.Θn[i][q]
    end

    return nothing
end

function compute_coefficients!(d::AMFunctor, h::Float64, t::Float64, s::Int, adaptive_count::Int)
    if d.is_fixed && iszero(adaptive_count)
        if s == 0
            d.coeff[1] = 1.0
        elseif s == 1
            d.coeff[1] = 0.5
            d.coeff[2] = 0.5
        elseif s == 2
            d.coeff[1] = 5.0/12.0
            d.coeff[2] = 2.0/3.0
            d.coeff[3] = -1.0/12.0
        elseif s == 3
            d.coeff[1] = 9.0/24.0
            d.coeff[2] = 19.0/24.0
            d.coeff[3] = -5.0/24.0
            d.coeff[4] = 1.0/24.0
        elseif s == 4
            d.coeff[1] = 251.0/720.0
            d.coeff[2] = 646.0/720.0
            d.coeff[3] = -264.0/720.0
            d.coeff[4] = 106.0/720.0
            d.coeff[5] = -19.0/720.0
        else
            compute_fixed_higher_coeffs!(d, s)
        end
    else
        compute_adaptive_coeffs!(d, h, t, s)
    end

    return nothing
end

function (d::AMFunctor)(contract::ContractorStorage{T}, result::StepResult{T},
                        adaptive_count::Int) where T<:Number

    h = contract.hj_computed
    t = contract.times[1]
    s = min(contract.step_count, d.method_step)

    # compute coefficients
    compute_coefficients!(d,h,t,s,adaptive_count)

    # compute Rk from union
    set_tf!(d.f̃, contract.Xj_apriori, contract.P, t)
    if iszero(adaptive_count)
        pushfirst!(d.fk_apriori, d.f̃[s + 1])
        d.Rk_temp = d.fk_apriori[2]
        for i = 3:d.s
            @__dot__ d.Rk_temp = d.Rk_temp ∪ d.fk_apriori[i]
        end
    end
    compute_Rk!(d, h)

    # compute real valued rhs sum
    @__dot__ Dk = result.xⱼ + (d.coeffs[s + 1]*h^(s + 2))*d.f̃[s + 1]
    if iszero(adaptive_count)
        eval_cycle!(d.f!, d.fval, contract.xval, contract.pval, t)
    else
        d.f!(d.fval[1], contract.xval, contract.pval, t)
    end
    for i = 1:(s + 1)
        @__dot__  Dk += h*coeff[i]*d.fval[i]
    end

    # evaluate Jacobian of rhs and sum thereof
    μ!(d.μX, contract.Xj_0, contract.xval, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)
    if iszero(adaptive_count)
        eval_cycle!(d.Jx!, d.Jxsto, d.μX, d.ρP, t)
        eval_cycle!(d.Jp!, d.Jpsto, d.μX, d.ρP, t)
    else
        d.Jx!(d.Jxsto[1], d.μX, d.ρP, t)
        d.Jp!(d.Jpsto[1], d.μX, d.ρP, t)
    end

    @__dot__ d.Y0 = mid(d.Jxsto[1])
    @__dot__ d.JxAff = d.Jxsto[1] - d.Y0

    d.sum_x = ((I - d.Jxsto[2])*contract.A[1])*contract.Δ[1]
    for i = 3:s
        d.sum_x += (hⱼ*d.coeff[i])*(d.Jxsto[i]*contract.A[2])*contract.Δ[2]
    end
    @__dot__ d.sum_p = hⱼ*d.coeff[1]*d.Jpsto[i]
    for i = 2:s
        @__dot__ d.sum_p += hⱼ*d.coeff[i]*d.Jpsto[i]
    end

    # compute X bounds and new value
    mul!(d.Jxvec, d.JxAff, d.Xj_delta)
    mul!(d.Jpvec, d.sum_p, contract.rP)
    @__dot__ contract.X_computed = contract.xval + d.Jxvec + d.sum_x + d.Jpvec
    @__dot__ contract.X_computed = contract.X_computed ∩ contract.Xj_0
    @__dot__ contract.xval_computed = mid(contract.X_computed)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = contract.A[1]
    mul!(d.Jxmid, Jf!.Jxsto, contract.A[2].Q)
    @__dot__ contract.B = mid(d.Jxmid)
    calculateQ!(Aⱼ₊₁, contract.B, nx)
    calculateQinv!(Aⱼ₊₁)

    # update Δⱼ₊₁
    d.Y = I - Y0*contract.A[2].Q
    d.precond = lu!(d.Y)
    d.YJxAff = d.precond\d.JxAff
    mul!(d.YJxΔx, d.YJxAff, Xj_delta)
    d.Ysumx = d.precond\d.sum_x
    d.YsumP = d.precond\d.sum_p
    mul!(d.YJpΔp, d.YsumP, contract.rP)
    @__dot__ d.Δⱼ₊₁ = d.YJxΔx + d.Ysumx + d.YJpΔp

    # store new time and step
    if iszero(adaptive_count)
        pushfirst!(contract.times, t)
        pushfirst!(contract.steps, hⱼ)
        pushfirst!(contract.Δ, d.Δⱼ₊₁)
    else
        contract.times[1] = t
        contract.steps[1] = hⱼ
        @__dot__ contract.Δ[1] = d.Δⱼ₊₁
    end

    return nothing
end

function state_contractor(m::AdamsMoulton, f, Jx!, Jp!, nx, np, style, s, h)
    AMFunctor(f, Jx!, Jp!, nx, np, Val(m.k), style, s)
end
state_contractor_k(m::AdamsMoulton) = m.k + 1
state_contractor_γ(m::AdamsMoulton) = 0.0
state_contractor_steps(m::AdamsMoulton) = m.k
