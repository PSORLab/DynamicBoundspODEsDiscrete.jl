include("C://Users//wilhe//Desktop//Package Development Work//DynamicBoundspODEsPILMS.jl//src//taylor_integrator_utilities.jl")

"""
$(TYPEDEF)

Provides preallocated storage for the QR factorization, Q, and the inverse of Q.

$(TYPEDFIELDS)
"""
mutable struct QRDenseStorage
    "QR Factorization"
    factorization::LinearAlgebra.QR{Float64,Array{Float64,2}}
    "Orthogonal matrix Q"
    Q::Array{Float64,2}
    "Inverse of Q"
    inverse::Array{Float64,2}
end

"""
$(TYPEDSIGNATURES)

A constructor for QRDenseStorage assumes `Q` is of size `nx`-by-`nx` and of
type `Float64`.
"""
function QRDenseStorage(nx::Int)
    A = Float64.(Matrix(I, nx, nx))
    factorization = LinearAlgebra.qrfactUnblocked!(A)
    Q = similar(A)
    inverse = similar(A)
    QRDenseStorage(factorization, Q, inverse)
end

"""
$(TYPEDSIGNATURES)

Copies information in `y` to `x` in place.
"""
function Base.copyto!(x::QRDenseStorage, y::QRDenseStorage)
    x.factorization = y.factorization
    x.Q .= y.Q
    x.inverse .= y.inverse
    nothing
end

"""
$(TYPEDSIGNATURES)

Computes the QR factorization of `A` of size `(nx,nx)` and then stores it to
fields in `qst`.
"""
function calculateQ!(qst::QRDenseStorage, A::Matrix{Float64}, nx::Int)
    qst.factorization = LinearAlgebra.qrfactUnblocked!(A)
    qst.Q .= qst.factorization.Q*Matrix(I,nx,nx)
    nothing
end

"""
$(TYPEDSIGNATURES)

Computes `inv(Q)`` via transpose! and stores this to `qst.inverse`.
"""
function calculateQinv!(qst::QRDenseStorage)
    transpose!(qst.inverse, qst.Q)
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
function parametric_lohners_method!(stf!::TaylorFunctor!{F,S,T},
                                    rtf!::TaylorFunctor!{F,S,S},
                                    dtf!::JacTaylorFunctor!{F,S,D},
                                    hⱼ, Ỹⱼ, Yⱼ, yⱼ, P, p, Aⱼ₊₁, Aⱼ, Δⱼ,
                                    result, tjac,
                                    cfg, Jxsto, Jpsto,
                                    Jx, Jp) where  {F <: Function, T <: Real,
                                                    S <: Real, D <: Real}

    nx = stf!.nx
    np = stf!.np

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

    copyto!(sỸⱼ₀, 1, Yⱼ, 1, nx)
    copyto!(sỸⱼ, 1, Yⱼ, 1, nx)
    copyto!(sỸⱼ₀, 1+nx, P, 1, np)
    copyto!(sỸⱼ, 1+nx, P, 1, np)

    copyto!(rỸⱼ₀, 1, yⱼ, 1, nx)
    copyto!(rỸⱼ, 1, yⱼ, 1, nx)
    copyto!(rỸⱼ₀, 1+nx, p, 1, np)
    copyto!(rỸⱼ, 1+nx, p, 1, np)

    k = stf!.s
    nx = stf!.nx

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

    jacobian_taylor_coeffs!(result, dtf!, Yⱼ, P, cfg)
    extract_JxJp!(Jx, Jp, result, tjac, nx, np, k)

    hji = 1.0
    for i in 1:k
        Jx[i] .*= hji
        Jp[i] .*= hji
        Jxsto .+= Jx[i]
        Jpsto .+= Jp[i]
        hji = hji*hⱼ
    end

    M2Y .= Jxsto*Aⱼ.Q                        # 4 allocs
    dtf!.B .= mid.(M2Y)
    calculateQ!(Aⱼ₊₁, dtf!.B, nx)            # + 3 allocs
    calculateQinv!(Aⱼ₊₁)                     # + 1 alloc

    @. dtf!.Yⱼ₊₁ = dtf!.vⱼ₊₁ + dtf!.Rⱼ₊₁
    @. dtf!.yⱼ₊₁ = dtf!.vⱼ₊₁ + dtf!.mRⱼ₊₁
    dtf!.Rⱼ₊₁ .-= dtf!.mRⱼ₊₁
    mul!(M1, Aⱼ₊₁.inverse, dtf!.Rⱼ₊₁);                     dtf!.Δⱼ₊₁ .= M1    #dtf!.Δⱼ₊₁ .= Aⱼ₊₁.inverse*dtf!.Rⱼ₊₁
    mul!(M2, Aⱼ₊₁.inverse, M2Y);        mul!(M1, M2, Δⱼ);  dtf!.Δⱼ₊₁ .+= M1   #dtf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Y)*Δⱼ
    mul!(M2, Aⱼ₊₁.inverse, Jpsto);      mul!(M1, M2, rP);  dtf!.Δⱼ₊₁ .+= M1   #dtf!.Δⱼ₊₁ .+= (Aⱼ₊₁.inverse*Jpsto)*rP
    mul!(M1, M2Y, Δⱼ);                                     dtf!.Yⱼ₊₁ .= M1    #dtf!.Yⱼ₊₁ .+= Y*Δⱼ
    mul!(M1, Jpsto, rP);                                   dtf!.Yⱼ₊₁ .+= M1   # dtf!.Yⱼ₊₁ .+= Jpsto*rP
    copyto!(Aⱼ, Aⱼ₊₁)
    nothing
end

Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jxsto = zeros(Interval{Float64},2,2)
Jpsto = zeros(Interval{Float64},2,2)

rtf  = TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))
Yⱼ = [Interval{Float64}(-10.0, 20.0); Interval{Float64}(-10.0, 20.0)]
P = [Interval{Float64}(2.0, 3.0); Interval{Float64}(2.0, 3.0)]
yⱼ = mid.(Yⱼ)
Δⱼ = Yⱼ - yⱼ
At = zeros(2,2) + I
Aⱼ =  QRDenseStorage(nx)
Aⱼ₊₁ =  QRDenseStorage(nx)
itf = TaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
dtf = g
hⱼ = 0.001
# TODO: Remember rP is computed outside iteration and stored to JacTaylorFunctor
plohners = parametric_lohners_method!(itf, rtf, dtf, hⱼ, Yⱼ, Yⱼ, yⱼ,
                                     P, p, Aⱼ₊₁, Aⱼ, Δⱼ, result, tjac, cfg,
                                     Jxsto, Jpsto, Jx, Jp)

@btime parametric_lohners_method!($itf, $rtf, $dtf, $hⱼ, $Yⱼ, $Yⱼ, $yⱼ,
                                 $P, $p, $Aⱼ₊₁, $Aⱼ, $Δⱼ, $result, $tjac, $cfg,
                                 $Jxsto, $Jpsto, $Jx, $Jp)
