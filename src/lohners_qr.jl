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
function parametric_lohners!(stf!::TaylorFunctor!{F,S,T},
                             rtf!::TaylorFunctor!{F,S,S},
                             dtf!::JacTaylorFunctor!{F,S,D},
                             hⱼ, Ỹⱼ, Yⱼ, yⱼ, Aⱼ₊₁, Aⱼ, Δⱼ,
                             result, Jx, Jp) where {F <: Function, T <: Real,
                             S <: Real, D <: Real}

    nx = stf!.nx
    np = stf!.np
    k = stf!.s

    sf̃ₜ = stf!.f̃ₜ
    sf̃ = stf!.f̃
    sỸⱼ₀ = stf!.Ỹⱼ₀
    sỸⱼ = stf!.Ỹⱼ

    rf̃ₜ = rtf!.f̃ₜ
    rf̃ = rtf!.f̃
    rỸⱼ₀ = rtf!.Ỹⱼ₀
    rỸⱼ = rtf!.Ỹⱼ

    rP = dtf!.rP
    M1 = dtf!.M1
    M2 = dtf!.M2
    M2Y = dtf!.M2Y

    copyto!(sỸⱼ₀, 1, Yⱼ, 1, nx + np)
    copyto!(sỸⱼ, 1, Yⱼ, 1, nx + np)
    copyto!(rỸⱼ₀, 1, yⱼ, 1, nx + np)
    copyto!(rỸⱼ, 1, yⱼ, 1, nx + np)

    stf!(sf̃ₜ, sỸⱼ)
    coeff_to_matrix!(sf̃, sf̃ₜ, nx, k)
    hjk = (hⱼ^k)
    for i in 1:nx
        dtf!.Rⱼ₊₁[i] = hjk*sf̃[i,k]
        dtf!.mRⱼ₊₁[i] = mid(dtf!.Rⱼ₊₁[i])
    end

    rtf!(rf̃ₜ , rỸⱼ₀)
    coeff_to_matrix!(rf̃, rf̃ₜ, nx, k)
    for j in 1:nx
        dtf!.vⱼ₊₁[j] = rf̃[j,1]
    end
    for i=2:(k+1)
        for j in 1:nx
            dtf!.vⱼ₊₁[j] += (hⱼ^i)*rf̃[j,k]
        end
    end

    jacobian_taylor_coeffs!(result, dtf!, Yⱼ)
    extract_JxJp!(Jx, Jp, result, dtf!.tjac, nx, np, k)

    hji = 1.0
    for i in 1:k
        Jx[i] .*= hji
        Jp[i] .*= hji
        dtf!.Jxsto .+= Jx[i]
        dtf!.Jpsto .+= Jp[i]
        hji = hji*hⱼ
    end

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    M2Y .= dtf!.Jxsto*Aⱼ.Q
    dtf!.B .= mid.(M2Y)
    calculateQ!(Aⱼ₊₁, dtf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    @. dtf!.Yⱼ₊₁ = dtf!.vⱼ₊₁ + dtf!.Rⱼ₊₁
    @. dtf!.yⱼ₊₁ = dtf!.vⱼ₊₁ + dtf!.mRⱼ₊₁
    dtf!.Rⱼ₊₁ .-= dtf!.mRⱼ₊₁
    mul!(M1, Aⱼ₊₁.inv, dtf!.Rⱼ₊₁);                          dtf!.Δⱼ₊₁ .= M1    #dtf!.Δⱼ₊₁ .= Aⱼ₊₁.inverse*dtf!.Rⱼ₊₁
    mul!(M2, Aⱼ₊₁.inv, M2Y);             mul!(M1, M2, Δⱼ);  dtf!.Δⱼ₊₁ .+= M1   #dtf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Y)*Δⱼ
    mul!(M2, Aⱼ₊₁.inv, dtf!.Jpsto);      mul!(M1, M2, rP);  dtf!.Δⱼ₊₁ .+= M1   #dtf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Jpsto)*rP
    mul!(M1, M2Y, Δⱼ);                                      dtf!.Yⱼ₊₁ .= M1    #dtf!.Yⱼ₊₁ .+= Y*Δⱼ
    mul!(M1, dtf!.Jpsto, rP);                               dtf!.Yⱼ₊₁ .+= M1   # dtf!.Yⱼ₊₁ .+= Jpsto*rP
    copyto!(Aⱼ, Aⱼ₊₁)
    nothing
end
