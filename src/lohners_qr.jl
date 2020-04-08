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
function parametric_lohners!(set_tf!::TaylorFunctor!{F,S,T},
                             real_tf!::TaylorFunctor!{F,S,S},
                             jac_tf!::JacTaylorFunctor!{F,S,D}, hⱼ, Ỹⱼ, Yⱼ,
                             A::QRStack,
                             yⱼ, Δⱼ) where {F <: Function, T <: Real,
                                                      S <: Real, D <: Real}
    nx = set_tf!.nx
    np = set_tf!.np
    k = set_tf!.s

    sf̃ₜ = set_tf!.f̃ₜ
    sf̃ = set_tf!.f̃
    sỸⱼ₀ = set_tf!.Ỹⱼ₀
    sỸⱼ = set_tf!.Ỹⱼ

    rf̃ₜ = real_tf!.f̃ₜ
    rf̃ = real_tf!.f̃
    rỸⱼ₀ = real_tf!.Ỹⱼ₀
    rỸⱼ = real_tf!.Ỹⱼ

    rP = jac_tf!.rP
    M1 = jac_tf!.M1
    M2 = jac_tf!.M2
    M2Y = jac_tf!.M2Y

    copyto!(sỸⱼ₀, 1, Yⱼ, 1, nx + np)
    copyto!(sỸⱼ, 1, Yⱼ, 1, nx + np)
    copyto!(rỸⱼ₀, 1, yⱼ, 1, nx + np)
    copyto!(rỸⱼ, 1, yⱼ, 1, nx + np)

    set_tf!(sf̃ₜ, sỸⱼ)
    coeff_to_matrix!(sf̃, sf̃ₜ, nx, k)
    hjk = (hⱼ^k)
    for i in 1:nx
        jac_tf!.Rⱼ₊₁[i] = hjk*sf̃[i,k]
        jac_tf!.mRⱼ₊₁[i] = mid(jac_tf!.Rⱼ₊₁[i])
    end

    real_tf!(rf̃ₜ , rỸⱼ₀)
    coeff_to_matrix!(rf̃, rf̃ₜ, nx, k)
    for j in 1:nx
        jac_tf!.vⱼ₊₁[j] = rf̃[j,1]
    end
    for i=2:(k+1)
        for j in 1:nx
            jac_tf!.vⱼ₊₁[j] += (hⱼ^i)*rf̃[j,k]
        end
    end

    jacobian_taylor_coeffs!(jac_tf!, Yⱼ)
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

    @. jac_tf!.Yⱼ₊₁ = jac_tf!.vⱼ₊₁ + jac_tf!.Rⱼ₊₁
    @. jac_tf!.yⱼ₊₁ = jac_tf!.vⱼ₊₁ + jac_tf!.mRⱼ₊₁
    jac_tf!.Rⱼ₊₁ .-= jac_tf!.mRⱼ₊₁
    mul!(M1, Aⱼ₊₁.inv, jac_tf!.Rⱼ₊₁);                          jac_tf!.Δⱼ₊₁ .= M1    #jac_tf!.Δⱼ₊₁ .= Aⱼ₊₁.inverse*jac_tf!.Rⱼ₊₁
    mul!(M2, Aⱼ₊₁.inv, M2Y);              mul!(M1, M2, Δⱼ);    jac_tf!.Δⱼ₊₁ .+= M1   #jac_tf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Y)*Δⱼ
    mul!(M2, Aⱼ₊₁.inv, jac_tf!.Jpsto);    mul!(M1, M2, rP);    jac_tf!.Δⱼ₊₁ .+= M1   #jac_tf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Jpsto)*rP
    mul!(M1, M2Y, Δⱼ);                                         jac_tf!.Yⱼ₊₁ .= M1    #jac_tf!.Yⱼ₊₁ .+= Y*Δⱼ
    mul!(M1, jac_tf!.Jpsto, rP);                               jac_tf!.Yⱼ₊₁ .+= M1   # jac_tf!.Yⱼ₊₁ .+= Jpsto*rP
    nothing
end
