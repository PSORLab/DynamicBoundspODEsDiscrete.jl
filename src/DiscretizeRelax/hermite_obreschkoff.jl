"""
$(TYPEDEF)

A structure that stores the cofficient of the (p,q)-Hermite-Obreschkoff method.

$(TYPEDFIELDS)
"""
struct HermiteObreschkoff{P,Q,K} <: AbstractStateContractorName
    "Cpq[i=1:p] index starting at i = 1 rather than 0"
    cpq::SVector{Float64,P}
    "Cqp[i=1:q] index starting at i = 1 rather than 0"
    cqp::SVector{Float64,Q}
    "gamma for method"
    γ::Float64
end
function HermiteObreschkoff(p::Int, q::Int)
    temp_cpq = 1.0
    temp_cqp = 1.0
    cpq = Float64[temp_cpq]
    cqp = Float64[temp_cqp]
    for i in 1:p
        temp_cpq = temp_cpq*(p - i + 1.0)/(p + q - i + 1)
        push!(cpq, temp_cpq)
    end
    for i in 1:q
        temp_cqp = temp_cqp*(q - i + 1.0)/(q + p - i + 1)
        push!(cqp, temp_cqp)
    end
    k = p + q + 1
    HermiteObreschkoff{p,q,k}(SVector{p}(cpq), SVector{q}(cqp),γ)
end

mutable struct HOFunctor{F <: Function, P, Q, K, T <: Real, S <: Real, NY} <: AbstractStateContractor
    ho::HermiteObreschkoff{P, Q, K}
    lon::LohnersFunctor{F, Q+1, T, S, NY}
    implicit_r::TaylorFunctor!{F, K, T, T}
    implicit_J::JacTaylorFunctor!{F, P, T, S, NY}
end
function HOFunctor(f!::F, nx::Int, np::Int, p::Val{P}, q::Val{Q},
                                   k::Val{K}, style, s) where {P,Q,K}
    set_tf! = TaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(T), zero(T))
    jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    HOFunctor{F, P, Q, K, T, S, nx+np}(set_tf!, real_tf!, jac_tf!)
end

function state_contractor(m::HermiteObreschkoff{P,Q,K}, f, nx, np, style, s) where {P,Q,K}
    HermiteObreschkoffFunctor(f, nx, np, Val{P}(), Val{Q}(), Val{K}(), style, s)
end
state_contractor_k(m::HermiteObreschkoff{P,Q,K}) where {P,Q,K} = K
state_contractor_γ(m::HermiteObreschkoff) = m.γ
state_contractor_steps(m::HermiteObreschkoff) = 2

"""
$(TYPEDSIGNATURES)

Implements the a parametric version of Nedialkov's Hermite=Obreschkoff
method (based on Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 49) full details to be included
in a forthcoming paper.
"""
function (d::HOFunctor{F,P,Q,K,T,S,NY})(hⱼ::Float64, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP,
                                        pval, t) where {F <: Function, K, S <: Real,
                                                        T <: Real, NY}

    # Compute lohner function step
    implicitJf! = x.implicit_J
    explicitJf! = x.lon.jac_tf!
    explicitrf! = x.lon.real_tf!
    p = P
    q = Q

    zⱼ₊₁ = explicitJf!.Rⱼ₊₁

    # perform a lohners method tightening
    d.lon(hⱼ, X̃ⱼ, Xⱼ, xval, A, Δ, P, rP, pval, t)

    Xⱼ₊₁ = explicitJf!.Xⱼ₊₁
    x̂0ⱼ₊₁ = mid.(Xⱼ₊₁)

    # compute real value sum of taylor series (implicit)
    fpⱼ₊₁ = zeros(nx)
    d.implicit_r(d.implicit_r.f̃, x̂0ⱼ₊₁, p, t)
    for i=1:p
        @__dot__ fpⱼ₊₁ += (hⱼ^i)*(ho.cpq[i])*d.implicit_r.f̃[i+1]
    end

    #
    fqⱼ₊₁ = zeros(nx)
    for i=1:q
        @__dot__ fqⱼ₊₁ += (ho.cpq[i])*explicitrf!.f̃[i+1]       # hⱼ^i included prior
    end
    gⱼ₊₁ = xval - x̂0ⱼ₊₁ + fpⱼ₊₁ + fqⱼ₊₁ + x.γ*zⱼ₊₁

    # compute sum of explicit Jacobian with ho weights
    for i = 1:p
        if i == 1
            for j = 1:nx
                explicitJf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ explicitJf!.Jxsto += (ho.cpq[i]*hⱼ^(i-1))*explicitJf!.Jx[i]
        end
        @__dot__ explicitJf!.Jpsto += (ho.cpq[i]*hⱼ^(i-1))*explicitJf!.Jp[i]
    end

    # compute set-valued extension of Jacobian of Taylor series (implicit)
    set_JxJp!(implicitJf!, Xⱼ₊₁, P, t)
    for i = 1:q
        if i == 1
            for j = 1:nx
                implicitJf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ implicitJf!.Jxsto += (ho.cqp[i]*hⱼ^(i-1))*implicitJf!.Jx[i]
        end
        @__dot__ implicitJf!.Jpsto += (ho.cqp[i]*hⱼ^(i-1))*implicitJf!.Jp[i]
    end

    Shat = mid.(implicitJf!.Jxsto)
    B0 = (inv(Shat)*explicitJf!.Jxsto)*A[2].Q
    C = I - Shat*implicitJf!.Jxsto
    VJ = Xⱼ₊₁ - x̂0ⱼ₊₁
    YJ1 = (x̂0ⱼ₊₁ + B0*Δⱼ[2] + C*VJ + inv(Shat)*gⱼ₊₁ +) .∩ Xⱼ₊₁
    mB = mid(B0)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = A[1]
    Jf!.B .= mid.(Jf!.Jxsto*A[2].Q)
    calculateQ!(Aⱼ₊₁, Jf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    mYJ1 = mid.(YJ1)
    Jf!.Δⱼ₊₁ = (Aⱼ₊₁.inv*B0)* + (Aⱼ₊₁.inv*C)* + (Aⱼ₊₁.inv*Shat) + Aⱼ₊₁.inv*(x̂0ⱼ₊₁ - mYJ1)

    pushfirst!(Δⱼ, Jf!.Δⱼ₊₁)

    RELAXATION_NOT_CALLED
end

get_Δ(lf) = lf.jac_tf!.Δⱼ₊₁
function set_x!(out::Vector{Float64}, lf::LohnersFunctor)
    out .= lf.jac_tf!.xⱼ₊₁
    nothing
end
function set_X!(out::Vector{S}, lf::LohnersFunctor) where S
    out .= lf.jac_tf!.Xⱼ₊₁
    nothing
end
