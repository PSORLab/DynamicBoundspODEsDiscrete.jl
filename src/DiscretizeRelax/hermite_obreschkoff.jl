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
    Jf! = x.implicit_J
    expJf! = x.jac_tf!
    p =
    q =

    zⱼ₊₁ = x.lon.jac_tf!.Rⱼ₊₁

    d.lon(hⱼ, X̃ⱼ, Xⱼ, xval, A, Δ, P, rP, pval, t)

    Xⱼ₊₁ = d.lon.jac_tf!.Xⱼ₊₁
    x̂0ⱼ₊₁ = mid.(Xⱼ₊₁)

    fpⱼ₊₁ = zeros(nx)
    d.implicit_r(d.implicit_r.f̃, x̂0ⱼ₊₁, p, t)
    for i=1:p
        fpⱼ₊₁ += (hⱼ^i)*(ho.cpq[i])*d.implicit_r.f̃[i+1]
    end
    fqⱼ₊₁ = zeros(nx)
    for i=1:q
        fqⱼ₊₁ += (ho.cpq[i])*d.lon.real_tf!.f̃[i+1]       # hⱼ^i included prior
    end

    gⱼ₊₁ = xval - x̂0ⱼ₊₁ + fpⱼ₊₁ + fqⱼ₊₁ + x.γ*zⱼ₊₁


    set_JxJp!(Jf!, Xⱼ, P, t)
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
