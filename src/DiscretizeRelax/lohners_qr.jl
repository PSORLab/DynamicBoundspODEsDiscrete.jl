"""
$(TYPEDEF)
"""
mutable struct LohnersFunctor{F <: Function, K, T <: Real, S <: Real, NY} <: AbstractStateContractor
    set_tf!::TaylorFunctor!{F, K, T, S}
    real_tf!::TaylorFunctor!{F, K, T, T}
    jac_tf!::JacTaylorFunctor!{F, K, T, S, NY}
    η::Float64
    μX::Vector{S}
    ρP::Vector{S}
end
function LohnersFunctor(f!::F, nx::Int, np::Int, k::Val{K}, s::S, t::T) where {F, K, S <: Number, T <: Number}
    set_tf! = TaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(T), zero(T))
    jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    LohnersFunctor{F, K+1, T, S, nx+np}(set_tf!, real_tf!, jac_tf!, 0.5, μX, ρP)
end

struct LohnerContractor{K} <: AbstractStateContractorName end
LohnerContractor(k::Int) = LohnerContractor{k}()
function state_contractor(m::LohnerContractor{K}, f, nx, np, style, s) where K
    LohnersFunctor(f, nx, np, Val{K}(), style, s)
end
state_contractor_k(m::LohnerContractor{K}) where K = K
state_contractor_γ(m::LohnerContractor) = 1.0
state_contractor_steps(m::LohnerContractor) = 2

function μ!(out::Vector{Interval{Float64}}, xⱼ::Vector{Interval{Float64}}, x̂ⱼ::Vector{Float64}, η::Float64)
    out .= xⱼ
    nothing
end
function μ!(out::Vector{MC{N,T}}, xⱼ::Vector{MC{N,T}}, x̂ⱼ::Vector{Float64}, η::Float64) where {N, T<:RelaxTag}
    @__dot__ out = x̂ⱼ + η*(xⱼ - x̂ⱼ)
    nothing
end

function ρ!(out::Vector{Interval{Float64}}, p::Vector{Interval{Float64}}, p̂::Vector{Float64}, η::Float64)
    out .= p
    nothing
end
function ρ!(out::Vector{MC{N,T}}, p::Vector{MC{N,T}}, p̂::Vector{Float64}, η::Float64) where {N, T<:RelaxTag}
    @__dot__ out = p̂ + η*(p - p̂)
    nothing
end

"""
$(TYPEDSIGNATURES)

An implementation of the parametric Lohner's method described in the paper in (1)
based on the non-parametric version given in (2).

1. [Sahlodin, Ali M., and Benoit Chachuat. "Discretize-then-relax approach for
convex/concave relaxations of the solutions of parametric ODEs." Applied Numerical
Mathematics 61.7 (2011): 803-820.](https://www.sciencedirect.com/science/article/abs/pii/S0168927411000316)
2. [R.J. Lohner, Computation of guaranteed enclosures for the solutions of
ordinary initial and boundary value problems, in: J.R. Cash, I. Gladwell (Eds.),
Computational Ordinary Differential Equations, vol. 1, Clarendon Press, 1992,
pp. 425–436.](http://www.goldsztejn.com/old-papers/Lohner-1992.pdf)
"""
function (d::LohnersFunctor{F,K,S,T,NY})(hⱼ::Float64, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP, pval, t) where {F <: Function, K, S <: Real, T <: Real, NY}

    set_tf! = d.set_tf!
    real_tf! = d.real_tf!
    Jf! = d.jac_tf!
    nx = set_tf!.nx
    k = set_tf!.k

    copyto!(set_tf!.X̃ⱼ₀, 1, X̃ⱼ, 1, nx)
    copyto!(set_tf!.X̃ⱼ, 1, X̃ⱼ, 1, nx)
    copyto!(real_tf!.X̃ⱼ₀, 1, xval, 1, nx)
    copyto!(real_tf!.X̃ⱼ, 1, xval, 1, nx)

    set_tf!(set_tf!.f̃, set_tf!.X̃ⱼ, P, t)
    real_tf!(real_tf!.f̃, real_tf!.X̃ⱼ₀, pval, t)
    hjk = hⱼ^k
    @__dot__ Jf!.Rⱼ₊₁ = hjk*set_tf!.f̃[k+1]
    @__dot__ Jf!.mRⱼ₊₁ = mid(Jf!.Rⱼ₊₁)
    @__dot__ Jf!.xⱼ₊₁ = xval + Jf!.mRⱼ₊₁

    for i = 2:k
        real_tf!.f̃[i] .*= (hⱼ^(i-1))
        @__dot__ Jf!.xⱼ₊₁ += real_tf!.f̃[i]
    end

    # compute extensions of taylor cofficients for rhs
    μ!(d.μX, Xⱼ, xval, d.η)
    ρ!(d.ρP, P, pval, d.η)
    set_JxJp!(Jf!, d.μX, d.ρP, t)
    for i = 1:k
        if i == 1
            for j = 1:nx
                Jf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ Jf!.Jxsto += (hⱼ^(i-1))*Jf!.Jx[i]
        end
        @__dot__ Jf!.Jpsto += (hⱼ^(i-1))*Jf!.Jp[i]
    end

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = A[1]
    Jf!.B .= mid.(Jf!.Jxsto*A[2].Q)
    calculateQ!(Aⱼ₊₁, Jf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    # update X and Delta
    mul!(Jf!.M2, Jf!.Jxsto, A[2].Q)
    mul!(Jf!.M1, Jf!.M2, Δⱼ[1])                     # (Jf!.Jxsto*A[2].Q)*Δⱼ[1]

    mul!(Jf!.M4, Jf!.Jpsto, rP)                     # Jf!.Jpsto*rP
    @__dot__ Jf!.M1a = Jf!.Rⱼ₊₁ - Jf!.mRⱼ₊₁
    @__dot__ Jf!.Xⱼ₊₁ = Jf!.xⱼ₊₁ + Jf!.M1 + Jf!.M4 + Jf!.M1a

    mul!(Jf!.M2a, Aⱼ₊₁.inv, Jf!.M2)
    mul!(Jf!.M1, Jf!.M2a, Δⱼ[1])                    # (Aⱼ₊₁.inv*(Jf!.Jxsto*A[2].Q))*Δⱼ[1]

    mul!(Jf!.M3, Aⱼ₊₁.inv, Jf!.Jpsto)
    mul!(Jf!.M4, Jf!.M3, rP)                        # (Aⱼ₊₁.inv*Jf!.Jpsto)*rP

    mul!(Jf!.M1b, Aⱼ₊₁.inv, Jf!.M1a)                # Aⱼ₊₁.inv*Jf!.M1a
    @__dot__ Jf!.Δⱼ₊₁ = Jf!.M1 + Jf!.M4 + Jf!.M1b

    pushfirst!(Δⱼ,Jf!.Δⱼ₊₁)

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

has_jacobians(d::LohnersFunctor) = true

function extract_jacobians!(d::LohnersFunctor{F,K,S,T,NY}, ∂f∂x::Vector{Matrix{T}},
                            ∂f∂p::Vector{Matrix{T}}, Xⱼ, P, t) where {F <: Function, K, S <: Real, T <: Real, NY}
    for i=1:(d.set_tf!.k+1)
        ∂f∂x[i] .= d.jac_tf!.Jx[i]
        ∂f∂p[i] .= d.jac_tf!.Jp[i]
    end
    nothing
end

function get_jacobians!(d::LohnersFunctor{F,K,S,T,NY}, ∂f∂x::Vector{Matrix{T}},
                        ∂f∂p::Vector{Matrix{T}}, Xⱼ, P, t) where {F <: Function, K, S <: Real, T <: Real, NY}
    set_JxJp!(d.jac_tf!, Xⱼ, P, t)
    extract_jacobians!(d, ∂f∂x, ∂f∂p, Xⱼ, P, t)
    nothing
end
