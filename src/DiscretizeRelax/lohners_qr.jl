"""
$(TYPEDEF)
"""
mutable struct LohnersFunctor{F <: Function, T <: Real, S <: Real}
    set_tf!::TaylorFunctor!{F,T,S}
    real_tf!::TaylorFunctor!{F,T,T}
    jac_tf!::JacTaylorFunctor!{F,T,S}
end
function LohnersFunctor(f!::F, nx::Int, np::Int, k::Int, s::S, t::T) where {F, S <: Real, T <: Real}
    set_tf! = TaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(T), zero(T))
    jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    LohnersFunctor{F,T,S}(set_tf!, real_tf!, jac_tf!)
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
function (x::LohnersFunctor{F,S,T})(hⱼ::Float64, X̃ⱼ, Xⱼ, xⱼ, A, Δⱼ, P, rP, p) where {F <: Function, S <: Real, T <: Real}


    #rP = P .- p
    # abbreviate field access

    set_tf! = x.set_tf!
    real_tf! = x.real_tf!
    jac_tf! = x.jac_tf!
    nx = set_tf!.nx; np = set_tf!.np; k = set_tf!.k
    sf̃ₜ = set_tf!.f̃ₜ; sf̃ = set_tf!.f̃; sX̃ⱼ₀ = set_tf!.X̃ⱼ₀; sX̃ⱼ = set_tf!.X̃ⱼ
    rf̃ₜ = real_tf!.f̃ₜ; rf̃ = real_tf!.f̃; rX̃ⱼ₀ = real_tf!.X̃ⱼ₀;  rX̃ⱼ = real_tf!.X̃ⱼ
    M1 = jac_tf!.M1;    M2 = jac_tf!.M2;  M3 = jac_tf!.M3
    M2Y = jac_tf!.M2Y

    copyto!(sX̃ⱼ₀, 1, Xⱼ, 1, nx)
    copyto!(sX̃ⱼ, 1, Xⱼ, 1, nx)
    copyto!(rX̃ⱼ₀, 1, xⱼ, 1, nx)
    copyto!(rX̃ⱼ, 1, xⱼ, 1, nx)

    set_tf!(sf̃ₜ, sX̃ⱼ, P)
    #=
    coeff_to_matrix!(sf̃, sf̃ₜ, nx, k)
    @__dot__  jac_tf!.Rⱼ₊₁ = (hⱼ^k)*sf̃[:,k+1]
    @__dot__  jac_tf!.mRⱼ₊₁ = mid(jac_tf!.Rⱼ₊₁)
    =#

    #=
    real_tf!(rf̃ₜ , rX̃ⱼ₀, p)
    coeff_to_matrix!(rf̃, rf̃ₜ, nx, k)
    @__dot__ jac_tf!.xⱼ₊₁ = xⱼ + jac_tf!.mRⱼ₊₁
    for i=2:k
        @__dot__ jac_tf!.xⱼ₊₁ += (hⱼ^(i-1))*rf̃[:,i]
    end

    # compute extensions of taylor cofficients for rhs
    set_JxJp!(jac_tf!, Xⱼ, P)
    fill!(jac_tf!.Jxsto, zero(S))
    jac_tf!.Jxsto[diagind(jac_tf!.Jxsto)] .= one(S)
    for i in 2:k
        jac_tf!.Jxsto .+= hⱼ^(i-1)*jac_tf!.Jx[i]
        jac_tf!.Jpsto .+= hⱼ^(i-1)*jac_tf!.Jp[i]
    end

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = A[1]
    jac_tf!.B .= mid.(jac_tf!.Jxsto*A[2].Q)
    calculateQ!(Aⱼ₊₁, jac_tf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    # update X and Delta
    #term1 = jac_tf!.Jxsto*A[2].Q)*Δⱼ[1]
    #copyto!(jac_tf!.M1, 1, term1, 1, nx)
    term1 = jac_tf!.Jxsto*A[2].Q)*Δⱼ[1]
    #copy!(jac_tf!.M1, jac_tf!.Jxsto*A[2].Q)*Δⱼ[1])
    term2 = jac_tf!.Jpsto*rP
    @__dot__ jac_tf!.Xⱼ₊₁ = jac_tf!.xⱼ₊₁ + term1 + term2 + jac_tf!.Rⱼ₊₁ - jac_tf!.mRⱼ₊₁
    term3 = (Aⱼ₊₁.inv*(jac_tf!.Jxsto*A[2].Q))*Δⱼ[1]
    term4 = (Aⱼ₊₁.inv*jac_tf!.Jpsto)*rP
    term5 = Aⱼ₊₁.inv*(jac_tf!.Rⱼ₊₁ - jac_tf!.mRⱼ₊₁)
    @__dot__ jac_tf!.Δⱼ₊₁ = term3 + term4 + term5

    pushfirst!(Δⱼ,jac_tf!.Δⱼ₊₁)
    =#

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


#μⱼ(xⱼ, x̂ⱼ, η) = x̂ⱼ + η*(xⱼ - x̂ⱼ)
#ρ(p) = p̂ + η*(p - p̂)
