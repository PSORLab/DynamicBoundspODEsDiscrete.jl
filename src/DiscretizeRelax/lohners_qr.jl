"""
$(TYPEDEF)
"""
struct LohnersFunctor{F <: Function, T <: Real, S <: Real}
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
function (x::LohnersFunctor{F,S,T})(hⱼ::Float64, X̃ⱼ, Xⱼ, xⱼ, A, Δⱼ, P, rP) where {F <: Function, S <: Real, T <: Real}

    printstruct = PrintCount()
    printstruct("Δⱼ[1] = $(Δⱼ[1])")
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

    printstruct("sX̃ⱼ: $(sX̃ⱼ)")
    printstruct("P: $(P)")

    set_tf!(sf̃ₜ, sX̃ⱼ, P)
    printstruct("sf̃ₜ: $(sf̃ₜ)")
    coeff_to_matrix!(sf̃, sf̃ₜ, nx, k)
    printstruct("sf̃: $(sf̃)")
    hjk = (hⱼ^k)
    for i in 1:nx
        jac_tf!.Rⱼ₊₁[i] = hjk*sf̃[i,k]
        jac_tf!.mRⱼ₊₁[i] = mid(jac_tf!.Rⱼ₊₁[i])
    end
    printstruct("jac_tf!.Rⱼ₊₁ = $(jac_tf!.Rⱼ₊₁)")
    printstruct("jac_tf!.mRⱼ₊₁ = $(jac_tf!.mRⱼ₊₁)")

    real_tf!(rf̃ₜ , rX̃ⱼ₀, mid.(P))
    coeff_to_matrix!(rf̃, rf̃ₜ, nx, k)
    for j in 1:nx
        jac_tf!.vⱼ₊₁[j] = rf̃[j,1]
    end
    for i=2:(k+1)
        for j in 1:nx
            jac_tf!.vⱼ₊₁[j] += (hⱼ^i)*rf̃[j,k]
        end
    end

    jacobian_taylor_coeffs!(jac_tf!, Xⱼ, P)
    extract_JxJp!(jac_tf!.Jx, jac_tf!.Jp, jac_tf!.result, jac_tf!.tjac, nx, np, k)

    hji = 1.0
    for i in 1:k
        jac_tf!.Jx[i] .*= hji
        jac_tf!.Jp[i] .*= hji
        jac_tf!.Jxsto .+= jac_tf!.Jx[i]
        jac_tf!.Jpsto .+= jac_tf!.Jp[i]
        hji = hji*hⱼ
    end

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = A[1]
    Aⱼ = A[2]

    M2Y .= jac_tf!.Jxsto*Aⱼ.Q
    jac_tf!.B .= mid.(M2Y)
    calculateQ!(Aⱼ₊₁, jac_tf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    @. jac_tf!.Xⱼ₊₁ = jac_tf!.vⱼ₊₁ + jac_tf!.Rⱼ₊₁
    @. jac_tf!.xⱼ₊₁ = jac_tf!.vⱼ₊₁ + jac_tf!.mRⱼ₊₁
    jac_tf!.Rⱼ₊₁ .-= jac_tf!.mRⱼ₊₁

    #jac_tf!.Δⱼ₊₁ .= Aⱼ₊₁.inverse*jac_tf!.Rⱼ₊₁
    mul!(M1, Aⱼ₊₁.inv, jac_tf!.Rⱼ₊₁);
    jac_tf!.Δⱼ₊₁ .= M1

    #jac_tf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Y)*Δⱼ
    mul!(M2, Aⱼ₊₁.inv, M2Y)
    printstruct("M2 = $(M2)")
    printstruct("Δⱼ[1] = $(Δⱼ[1])")
    mul!(M1, M2, Δⱼ[1])
    printstruct("M1 = $(M1)")
    jac_tf!.Δⱼ₊₁ .+= M1
    printstruct("jac_tf!.Δⱼ₊₁ = $(jac_tf!.Δⱼ₊₁)")

    #jac_tf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Jpsto)*rP
    mul!(M3, Aⱼ₊₁.inv, jac_tf!.Jpsto)
    printstruct("M3 = $(M3)")
    mul!(M1, M3, rP)
    printstruct("M1 = $(M1)")
    jac_tf!.Δⱼ₊₁ .+= M1
    printstruct("jac_tf!.Δⱼ₊₁ = $(jac_tf!.Δⱼ₊₁)")

    #jac_tf!.Xⱼ₊₁ .+= Y*Δⱼ
    mul!(M1, M2Y,  Δⱼ[1])
    printstruct("M1 = $(M1)")
    jac_tf!.Xⱼ₊₁ .= M1
    printstruct("jac_tf!.Xⱼ₊₁ = $(jac_tf!.Xⱼ₊₁)")

    # jac_tf!.Xⱼ₊₁ .+= Jpsto*rP
    mul!(M1, jac_tf!.Jpsto, rP)
    jac_tf!.Xⱼ₊₁ .+= M1
    printstruct("jac_tf!.Xⱼ₊₁ = $(jac_tf!.Xⱼ₊₁)")

    pushfirst!(Δⱼ,jac_tf!.Δⱼ₊₁)

    RELAXATION_NOT_CALLED
end

get_Δ(lf) = lf.jac_tf!.Δⱼ₊₁
function set_x!(out::Vector{Float64}, lf::LohnersFunctor)
    println("x out: $(lf.jac_tf!.xⱼ₊₁)")
    out .= lf.jac_tf!.xⱼ₊₁
    nothing
end
function set_X!(out::Vector{S}, lf::LohnersFunctor) where S
    println("X out: $(lf.jac_tf!.Xⱼ₊₁)")
    out .= lf.jac_tf!.Xⱼ₊₁
    nothing
end


#μⱼ(xⱼ, x̂ⱼ, η) = x̂ⱼ + η*(xⱼ - x̂ⱼ)
#ρ(p) = p̂ + η*(p - p̂)
