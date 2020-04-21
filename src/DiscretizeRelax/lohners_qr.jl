"""
$(TYPEDEF)
"""
mutable struct LohnersFunctor{F <: Function, K, T <: Real, S <: Real, NY}
    set_tf!::TaylorFunctor!{F, K, T, S}
    real_tf!::TaylorFunctor!{F, K, T, T}
    jac_tf!::JacTaylorFunctor!{F, K, T, S, NY}
end
function LohnersFunctor(f!::F, nx::Int, np::Int, k::Val{K}, s::S, t::T) where {F, K, S <: Number, T <: Number}
    set_tf! = TaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(T), zero(T))
    jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    LohnersFunctor{F, K+1, T, S, nx+np}(set_tf!, real_tf!, jac_tf!)
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
function (x::LohnersFunctor{F,K,S,T,NY})(hⱼ::Float64, X̃ⱼ, Xⱼ, xⱼ, A, Δⱼ, P, rP, p, t) where {F <: Function, K, S <: Real, T <: Real, NY}


    #rP = P .- p
    # abbreviate field access

    set_tf! = x.set_tf!
    real_tf! = x.real_tf!
    jac_tf! = x.jac_tf!
    nx = set_tf!.nx;
    np = set_tf!.np;
    k = set_tf!.k
    sf̃ = set_tf!.f̃;
    sX̃ⱼ₀ = set_tf!.X̃ⱼ₀;
    sX̃ⱼ = set_tf!.X̃ⱼ
    rf̃ = real_tf!.f̃;
    rX̃ⱼ₀ = real_tf!.X̃ⱼ₀;
    rX̃ⱼ = real_tf!.X̃ⱼ
    M1 = jac_tf!.M1;
    M2 = jac_tf!.M2;
    M3 = jac_tf!.M3
    M2Y = jac_tf!.M2Y

    copyto!(sX̃ⱼ₀, 1, Xⱼ, 1, nx)
    copyto!(sX̃ⱼ, 1, Xⱼ, 1, nx)
    copyto!(rX̃ⱼ₀, 1, xⱼ, 1, nx)
    copyto!(rX̃ⱼ, 1, xⱼ, 1, nx)

    set_tf!(sf̃, sX̃ⱼ, P, t)
    real_tf!(rf̃, rX̃ⱼ₀, p, t)
    hjk = hⱼ^k
    for i in eachindex(jac_tf!.Rⱼ₊₁)
        jac_tf!.Rⱼ₊₁[i] = hjk*sf̃[k+1][i]
        jac_tf!.mRⱼ₊₁[i] = mid.(jac_tf!.Rⱼ₊₁[i])
        jac_tf!.xⱼ₊₁[i] = xⱼ[i] + jac_tf!.mRⱼ₊₁[i]
    end

    for i=2:k
        hji = hⱼ^(i-1)
        for j in eachindex(jac_tf!.xⱼ₊₁)
            jac_tf!.xⱼ₊₁[j] += hji*rf̃[i][j]
        end
    end
    # compute extensions of taylor cofficients for rhs
    set_JxJp!(jac_tf!, Xⱼ, P, t)

    for i=1:nx
        jac_tf!.Jxsto[i,i] = one(S)
    end
    for i=2:k
        hji = hⱼ^(i-1)
        for j in eachindex(jac_tf!.Jxsto)
            jac_tf!.Jxsto[j] += hji*jac_tf!.Jx[i][j]
        end
    end
    for i=2:k
        hji = hⱼ^(i-1)
        for j in eachindex(jac_tf!.Jpsto)
            jac_tf!.Jpsto[j] += hji*jac_tf!.Jp[i][j]
        end
    end

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = A[1]
    jac_tf!.B .= mid.(jac_tf!.Jxsto*A[2].Q)
    calculateQ!(Aⱼ₊₁, jac_tf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    # update X and Delta
    #term1 = jac_tf!.Jxsto*A[2].Q)*Δⱼ[1]
    #copyto!(jac_tf!.M1, 1, term1, 1, nx)
    term1 = (jac_tf!.Jxsto*A[2].Q)*Δⱼ[1]
    #copy!(jac_tf!.M1, jac_tf!.Jxsto*A[2].Q)*Δⱼ[1])
    term2 = jac_tf!.Jpsto*rP
    @__dot__ jac_tf!.Xⱼ₊₁ = jac_tf!.xⱼ₊₁ + term1 + term2 + jac_tf!.Rⱼ₊₁ - jac_tf!.mRⱼ₊₁
    term3 = (Aⱼ₊₁.inv*(jac_tf!.Jxsto*A[2].Q))*Δⱼ[1]
    term4 = (Aⱼ₊₁.inv*jac_tf!.Jpsto)*rP
    term5 = Aⱼ₊₁.inv*(jac_tf!.Rⱼ₊₁ - jac_tf!.mRⱼ₊₁)
    @__dot__ jac_tf!.Δⱼ₊₁ = term3 + term4 + term5

    pushfirst!(Δⱼ,jac_tf!.Δⱼ₊₁)

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
