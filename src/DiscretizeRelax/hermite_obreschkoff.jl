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

    println(" ------ START FUNCTOR ------")
    println("  *** inputs *** ")
    println(" X̃ⱼ = $(X̃ⱼ), Xⱼ = $(Xⱼ), xval = $(xval), P = $(P), rP = $(rP), pval = $(pval)")
    println(" A = $(A),  Δⱼ = $(Δⱼ)")
    println("  *** calcs *** ")
    # Compute lohner function step
    implicitJf! = d.implicit_J
    explicitJf! = d.lon.jac_tf!
    explicitrf! = d.lon.real_tf!
    p = d.hermite_obreschkoff.p
    q = d.hermite_obreschkoff.q
    cqp = d.hermite_obreschkoff.cqp
    cpq = d.hermite_obreschkoff.cpq
    println("hermite obresrchkoff cqp coefficient: $(cqp)")
    println("hermite obresrchkoff cpq coefficient: $(cpq)")
    println("p = $p")
    println("q = $q")
    hⱼ = hbuffer[1]
    t = tbuffer[1]
    nx = d.lon.set_tf!.nx

    # perform a lohners method tightening
    d.lon(hbuffer, tbuffer, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP, pval, fk)

    println("post Lohners Δⱼ: $(Δⱼ)")

    println("input fk: $(fk)")
    zⱼ₊₁ = (hⱼ^(p+q+1))*fk
    println("input zⱼ₊₁: $(zⱼ₊₁)")

    Xⱼ₊₁ = explicitJf!.Xⱼ₊₁
    println("input Xⱼ₊₁: $(Xⱼ₊₁)") #LOOKS GOOD
    x̂0ⱼ₊₁ = mid.(Xⱼ₊₁)
    println("mid of input Xⱼ₊₁: $(x̂0ⱼ₊₁)") #LOOKS GOOD

    # compute real value sum of taylor series (implicit)

    copyto!(d.implicit_r.X̃ⱼ₀, 1, xval, 1, nx)
    copyto!(d.implicit_r.X̃ⱼ, 1, xval, 1, nx)

    fqⱼ₊₁ = zeros(nx)
    d.implicit_r(d.implicit_r.f̃, x̂0ⱼ₊₁, pval, t)
    println(" implicit real taylor coefficients: $(d.implicit_r.f̃)")
    for i=2:(q+1)
        println("i: $i")
        @__dot__ fqⱼ₊₁ -= (hⱼ^(i-1))*(cqp[i])*d.implicit_r.f̃[i]
        println("fqⱼ₊₁ = $(fqⱼ₊₁)")
    end

    #
    fpⱼ₊₁ = zeros(nx)
    println(" explicit real taylor coefficients: $(explicitrf!.f̃)")
    for i=2:(p+1)
        println("i: $i")
        @__dot__ fpⱼ₊₁ += (cpq[i])*explicitrf!.f̃[i]       # hⱼ^(i-1) performed in Lohners
        println("fpⱼ₊₁ = $(fpⱼ₊₁)")
    end
    println("gamma value: $(d.hermite_obreschkoff.γ)")
    gⱼ₊₁ = xval - x̂0ⱼ₊₁ + fpⱼ₊₁ + fqⱼ₊₁ + d.hermite_obreschkoff.γ*zⱼ₊₁
    println("gⱼ₊₁ value : $(gⱼ₊₁)")

    println("input sum of explicit Jx: $(explicitJf!.Jxsto)")
    println("input sum of explicit Jp: $(explicitJf!.Jpsto)")
    # compute sum of explicit Jacobian with ho weights
    fill!(explicitJf!.Jxsto, zero(S))
    fill!(explicitJf!.Jpsto, zero(S))
    println("sum of explicit Jx: $(explicitJf!.Jxsto)")
    println("sum of explicit Jp: $(explicitJf!.Jpsto)")
    for i = 1:(p+1)
        if i == 1
            for j = 1:nx
                explicitJf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ explicitJf!.Jxsto += (cpq[i]*hⱼ^(i-1))*explicitJf!.Jx[i]
            @__dot__ explicitJf!.Jpsto += (cpq[i]*hⱼ^(i-1))*explicitJf!.Jp[i]
        end
    end
    println("sum of explicit Jx: $(explicitJf!.Jxsto)")
    println("sum of explicit Jp: $(explicitJf!.Jpsto)")

    # compute set-valued extension of Jacobian of Taylor series (implicit)
    μ!(d.μX, Xⱼ₊₁, xval, d.η)
    ρ!(d.ρP, P, pval, d.η)
    set_JxJp!(implicitJf!, d.μX, d.ρP, t)
    for i = 1:(q+1)
        if i == 1
            for j = 1:nx
                implicitJf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ implicitJf!.Jxsto += (cqp[i]*hⱼ^(i-1))*implicitJf!.Jx[i]
            @__dot__ implicitJf!.Jpsto += (cqp[i]*hⱼ^(i-1))*implicitJf!.Jp[i]
        end
    end
    println("sum of implicit Jx: $(implicitJf!.Jxsto)")
    println("sum of implicit Jp: $(implicitJf!.Jpsto)")

    Shat = mid.(implicitJf!.Jxsto)
    println("Shat: $(Shat)")
    println("A[2].Q: $(A[2].Q)")
    B0 = (inv(Shat)*explicitJf!.Jxsto)*Alast[1].Q
    println("B0: $(B0)")
    C = I - Shat*implicitJf!.Jxsto
    println("C: $C")
    VJ = Xⱼ₊₁ - x̂0ⱼ₊₁
    println("Δⱼlast: $(Δⱼlast)")
    println("x̂0ⱼ₊₁ = $(x̂0ⱼ₊₁), B0*Δⱼlast[1] = $(B0*Δⱼlast[1]), C*VJ = $(C*VJ), $(inv(Shat)*gⱼ₊₁)")
    pre_intersect = (x̂0ⱼ₊₁ + B0*Δⱼlast[1] + C*VJ + inv(Shat)*gⱼ₊₁)
    println("pre_intersect: $(pre_intersect), Xⱼ₊₁ = $(Xⱼ₊₁)") # Xj+1 is good
    YJ1 = pre_intersect .∩ Xⱼ₊₁
    implicitJf!.Xⱼ₊₁ .= YJ1
    println("YJ1: $(YJ1)")
    mB = mid.(B0)
    println("mB: $(mB)")

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = Alast[1]
    implicitJf!.B .= mid.(implicitJf!.Jxsto*Alast[1].Q)
    calculateQ!(Aⱼ₊₁, implicitJf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    mYJ1 = mid.(YJ1)
    R = Δⱼ[2]
    V = VJ
    implicitJf!.Δⱼ₊₁ = (Aⱼ₊₁.inv*B0)*R + (Aⱼ₊₁.inv*C)*V + (Aⱼ₊₁.inv*Shat)*gⱼ₊₁ + Aⱼ₊₁.inv*(x̂0ⱼ₊₁ - mYJ1)

    pushfirst!(Δⱼ, implicitJf!.Δⱼ₊₁)

    RELAXATION_NOT_CALLED
end

get_Δ(lf::HermiteObreschkoffFunctor) = lf.implicit_J.Δⱼ₊₁
function set_x!(out::Vector{Float64}, lf::HermiteObreschkoffFunctor)
    out .= lf.implicit_J.xⱼ₊₁
    nothing
end
function set_X!(out::Vector{S}, lf::HermiteObreschkoffFunctor) where S
    out .= lf.implicit_J.Xⱼ₊₁
    nothing
end

has_jacobians(d::HermiteObreschkoffFunctor) = true

function extract_jacobians!(d::HermiteObreschkoffFunctor, ∂f∂x::Vector{Matrix{T}},
                            ∂f∂p::Vector{Matrix{T}}) where {T <: Real}
    for i=1:(d.lon.set_tf!.k+1)
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
