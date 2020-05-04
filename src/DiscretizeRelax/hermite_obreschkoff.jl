"""
$(TYPEDEF)

A structure that stores the cofficient of the (P,Q)-Hermite-Obreschkoff method.
(Offset due to method being zero indexed and Julia begin one indexed).
$(TYPEDFIELDS)
"""
struct HermiteObreschkoff{P,Q,K} <: AbstractStateContractorName
    "Cpq[i=1:p] index starting at i = 1 rather than 0"
    cpq::Vector{Float64}
    "Cqp[i=1:q] index starting at i = 1 rather than 0"
    cqp::Vector{Float64}
    "gamma for method"
    γ::Float64
    "Explicit order Hermite-Obreschkoff"
    p::Int
    "Implicit order Hermite-Obreschkoff"
    q::Int
    "Total order Hermite-Obreschkoff"
    k::Int
end
function HermiteObreschkoff(p::Val{P}, q::Val{Q}) where {P,Q}
    temp_cpq = 1.0
    temp_cqp = 1.0
    cpq = zeros(P+1)
    cqp = zeros(Q+1)
    cpq[1] = temp_cpq
    cqp[1] = temp_cqp
    for i in 1:P
        temp_cpq *= (P - i + 1.0)/(P + Q - i + 1)
        cpq[i+1] = temp_cpq
    end
    γ = 1.0
    for i in 1:Q
        temp_cqp *= (Q - i + 1.0)/(Q + P - i + 1)
        cqp[i+1] = temp_cqp
        γ *= i/(P+i)
    end
    K = P + Q + 1
    HermiteObreschkoff{P,Q,K}(cpq, cqp, γ, P, Q, K)
end
HermiteObreschkoff(p::Int, q::Int) = HermiteObreschkoff(Val(p), Val(q))

mutable struct HermiteObreschkoffFunctor{F <: Function, P, Q, K, Q1, T <: Real, S <: Real, NY} <: AbstractStateContractor
    hermite_obreschkoff::HermiteObreschkoff{P, Q, K}
    lon::LohnersFunctor{F, K, T, S, NY}
    implicit_r::TaylorFunctor!{F, Q1, T, T}
    implicit_J::JacTaylorFunctor!{F, Q1, T, S, NY}
    η::Interval{Float64}
    μX::Vector{S}
    ρP::Vector{S}
end
function HermiteObreschkoffFunctor(f!::F, nx::Int, np::Int, p::Val{P}, q::Val{Q},
                                   k::Val{K}, s::S, t::T) where {F,P,Q,K,S,T}
    hermite_obreschkoff = HermiteObreschkoff(p,q)
    lon = LohnersFunctor(f!, nx, np, Val(K-1), s, t)
    implicit_r = TaylorFunctor!(f!, nx, np, Val(Q), zero(T), zero(T))
    implicit_J = JacTaylorFunctor!(f!, nx, np, Val(Q), zero(S), zero(T))
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    HermiteObreschkoffFunctor{F, P, Q, K, Q+1, T, S, nx+np}(hermite_obreschkoff, lon,
                                                       implicit_r, implicit_J,
                                                       Interval{Float64}(0.0,1.0), μX, ρP)
end

function state_contractor(m::HermiteObreschkoff{P,Q,K}, f, Jx!, Jp!, nx, np, style, s) where {P,Q,K}
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
function (d::HermiteObreschkoffFunctor{F,Pp,Q,K,T,S,NY})(hbuffer, tbuffer, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP,
                                                           pval, fk) where {F,Pp,Q,K,T,S,NY}

    Δⱼlast = deepcopy(Δⱼ)
    Alast = deepcopy(A)

    # Compute lohner function step
    implicitJf! = d.implicit_J
    explicitJf! = d.lon.jac_tf!
    explicitrf! = d.lon.real_tf!
    p = d.hermite_obreschkoff.p
    q = d.hermite_obreschkoff.q
    cqp = d.hermite_obreschkoff.cqp
    cpq = d.hermite_obreschkoff.cpq
    hⱼ = hbuffer[1]
    t = tbuffer[1]
    nx = d.lon.set_tf!.nx

    # perform a lohners method tightening
    d.lon(hbuffer, tbuffer, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP, pval, fk)

    zⱼ₊₁ = (hⱼ^(p+q+1))*fk

    Xⱼ₊₁ = explicitJf!.Xⱼ₊₁
    x̂0ⱼ₊₁ = mid.(Xⱼ₊₁)

    # compute real value sum of taylor series (implicit)

    copyto!(d.implicit_r.X̃ⱼ₀, 1, xval, 1, nx)
    copyto!(d.implicit_r.X̃ⱼ, 1, xval, 1, nx)

    fqⱼ₊₁ = zeros(nx)
    d.implicit_r(d.implicit_r.f̃, x̂0ⱼ₊₁, pval, t)
    for i=2:(q+1)
        @__dot__ fqⱼ₊₁ -= ((-hⱼ)^(i-1))*cqp[i]*d.implicit_r.f̃[i]
    end

    #
    fpⱼ₊₁ = zeros(nx)
    for i=2:(p+1)
        @__dot__ fpⱼ₊₁ += (cpq[i])*explicitrf!.f̃[i]       # hⱼ^(i-1) performed in Lohners
    end
    gⱼ₊₁ = xval - x̂0ⱼ₊₁ + fpⱼ₊₁ + fqⱼ₊₁ + d.hermite_obreschkoff.γ*zⱼ₊₁

    # compute sum of explicit Jacobian with ho weights
    fill!(explicitJf!.Jxsto, zero(S))
    fill!(explicitJf!.Jpsto, zero(S))
    for i = 1:p+1
        if i == 1
            for j = 1:nx
                explicitJf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ explicitJf!.Jxsto += (cpq[i]*hⱼ^(i-1))*explicitJf!.Jx[i]
            @__dot__ explicitJf!.Jpsto += (cpq[i]*hⱼ^(i-1))*explicitJf!.Jp[i]
        end
    end

    # compute set-valued extension of Jacobian of Taylor series (implicit)
    μ!(d.μX, Xⱼ₊₁, xval, d.η)
    ρ!(d.ρP, P, pval, d.η)
    set_JxJp!(implicitJf!, d.μX, d.ρP, t)
    for i = 1:q+1
        if i == 1
            for j = 1:nx
                implicitJf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ implicitJf!.Jxsto += ((-hⱼ)^(i-1))*cqp[i]*implicitJf!.Jx[i]
            @__dot__ implicitJf!.Jpsto += ((-hⱼ)^(i-1))*cqp[i]*implicitJf!.Jp[i]
        end
    end

    Shat = mid.(implicitJf!.Jxsto)
    B0 = (inv(Shat)*explicitJf!.Jxsto)*Alast[1].Q
    C = I - inv(Shat)*implicitJf!.Jxsto
    VJ = Xⱼ₊₁ - x̂0ⱼ₊₁
    PsumJ = explicitJf!.Jpsto - implicitJf!.Jpsto
    Pterm = (inv(Shat)*PsumJ)*rP
    pre_intersect = x̂0ⱼ₊₁ + B0*Δⱼlast[1] + C*VJ + inv(Shat)*gⱼ₊₁ + Pterm
    YJ1 = pre_intersect .∩ Xⱼ₊₁
    implicitJf!.Xⱼ₊₁ .= YJ1
    mB = mid.(B0)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = Alast[1]
    implicitJf!.B .= mid.(implicitJf!.Jxsto*Alast[1].Q)
    calculateQ!(Aⱼ₊₁, implicitJf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    mYJ1 = mid.(YJ1)
    implicitJf!.xⱼ₊₁ .= mYJ1
    R = Δⱼ[2]
    PsumJ = explicitJf!.Jpsto - implicitJf!.Jpsto
    term = (Aⱼ₊₁.inv*PsumJ)*rP
    implicitJf!.Δⱼ₊₁ = (Aⱼ₊₁.inv*B0)*Δⱼlast[1] + (Aⱼ₊₁.inv*C)*VJ+ (Aⱼ₊₁.inv*inv(Shat))*gⱼ₊₁ + Aⱼ₊₁.inv*(x̂0ⱼ₊₁ - mYJ1) + term

    pushfirst!(Δⱼ, implicitJf!.Δⱼ₊₁)

    RELAXATION_NOT_CALLED
end

function get_Δ(f::HermiteObreschkoffFunctor)
    f.implicit_J.Δⱼ₊₁
end
function set_x!(out::Vector{Float64}, f::HermiteObreschkoffFunctor)
    out .= f.implicit_J.xⱼ₊₁
    nothing
end
function set_X!(out::Vector{S}, f::HermiteObreschkoffFunctor) where S
    out .= f.implicit_J.Xⱼ₊₁
    nothing
end

has_jacobians(d::HermiteObreschkoffFunctor) = true

function extract_jacobians!(d::HermiteObreschkoffFunctor, ∂f∂x::Vector{Matrix{T}},
                            ∂f∂p::Vector{Matrix{T}}) where {T <: Real}
    for i = 1:d.lon.set_tf!.k+1
        ∂f∂x[i] .= d.lon.jac_tf!.Jx[i]
        ∂f∂p[i] .= d.lon.jac_tf!.Jp[i]
    end
    nothing
end

function get_jacobians!(d::HermiteObreschkoffFunctor, ∂f∂x::Vector{Matrix{T}},
                        ∂f∂p::Vector{Matrix{T}}, Xⱼ, P, t) where {T <: Real}
    set_JxJp!(d.lon.jac_tf!, Xⱼ, P, t[1])
    extract_jacobians!(d, ∂f∂x, ∂f∂p)
    nothing
end
