include("C://Users//wilhe//Desktop//Package Development Work//DynamicBoundspODEsPILMS.jl//src//taylor_integrator_utilities.jl")

"""
$(TYPEDEF)

A structure that stores the cofficient of the (p,q)-Hermite-Obreschkoff method.

$(TYPEDFIELDS)
"""
struct HermiteObreschkoff{P,Q}
    "Cpq[i=1:p] index starting at i = 1 rather than 0"
    cpq::SVector{Float64,P}
    "Cqp[i=1:q] index starting at i = 1 rather than 0"
    cqp::SVector{Float64,Q}
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
    HermiteObreschkoff(SVector{p}(cpq), SVector{q}(cqp))
end


"""
$(TYPEDSIGNATURES)

Implements the a parametric version of Nedialkov's Hermite=Obreschkoff
method (based on Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 49) full details to be included
in a forthcoming paper.
"""
function parametric_hermite_obreschkoff!(stf!::TaylorFunctor!{F,S,T},
                                         rtf!::TaylorFunctor!{F,S,S},
                                         dtf!::JacTaylorFunctor!{F,S,D},
                                         hⱼ, Ỹⱼ, Yⱼ, Y0ⱼ₊₁, yⱼ, P, p, Aⱼ₊₁, Aⱼ, Δⱼ,
                                         result, tjac,
                                         cfg, Jxsto, Jpsto,
                                         Jx, Jp, ho::HermiteObreschkoff{Pho,Qho}) where  {F <: Function, T <: Real,
                                                                                      S <: Real, D <: Real,
                                                                                      Qho, Pho}

    k = stf!.s
    nx = stf!.nx
    np = stf!.nx

    rf̃ₜ = rtf!.f̃ₜ
    rf̃ₜ = rtf!.f̃ₜ
    rỸⱼ₀ = rtf!.Ỹⱼ₀
    rỸⱼ = rtf!.Ỹⱼ
    mY0ⱼ₊₁ = mid.(Y0ⱼ₊₁)
    copyto!(rỸⱼ₀, 1, mY0ⱼ₊₁, 1, nx)
    copyto!(rỸⱼ, 1+nx, p, 1, np)

    rtf!(rf̃ₜ , rỸⱼ₀)
    coeff_to_matrix!(rf̃, rf̃ₜ, nx, k)               # vⱼ₊₁ = f- j+1,i
    for j in 1:nx
        dtf!.vⱼ₊₁[j] = rf̃[j,1]
    end
    for i=2:(k+1)
        for j in 1:nx
            dtf!.vⱼ₊₁[j] += (hⱼ^i)*rf̃[j,k]
        end
    end

    mYⱼ₊₁ = mid.(Y0ⱼ₊₁)
    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Bⱼ₊₁ .= inv(mid.(Jxₖ₊₁))*(Jxsto*Aⱼ.Q)
    Cⱼ₊₁ = inv(mid.(Jxₖ₊₁))*Jxₖ₊₁
    dtf!.B .= mid.(M2Y)
    calculateQ!(Aⱼ₊₁, dtf!.B, nx)
    calculateQinv!(Aⱼ₊₁)
    Yⱼ₊₁ = (mY0ⱼ₊₁ + Bⱼ*Δⱼ + Cⱼ₊₁*(Y0ⱼ₊₁ - mYⱼ₊₁) + δⱼ₊₁ + ) ∩ Y0ⱼ₊₁
    yⱼ₊₁ = mid.(Yⱼ₊₁)
    Δⱼ₊₁ = (Aⱼ₊₁.inv*Bⱼ₊₁)*Δⱼ + (Aⱼ₊₁.inv*Cⱼ₊₁)*vⱼ₊₁ + (mY0ⱼ₊₁ - mYⱼ₊₁)

    nothing
end

Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jxsto = zeros(Interval{Float64},2,2)
Jpsto = zeros(Interval{Float64},2,2)

rtf  = TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))
Yⱼ = [Interval{Float64}(-10.0, 20.0); Interval{Float64}(-10.0, 20.0)]
Y0ⱼ₊₁ = Yⱼ
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
plohners = parametric_hermite_obreschkoff!itf, rtf, dtf, hⱼ, Yⱼ, Yⱼ, Y0ⱼ₊₁, yⱼ,
                                     P, p, Aⱼ₊₁, Aⱼ, Δⱼ, result, tjac, cfg,
                                     Jxsto, Jpsto, Jx, Jp)

@btime parametric_hermite_obreschkoff!($itf, $rtf, $dtf, $hⱼ, $Yⱼ, $Yⱼ, $Y0ⱼ₊₁, $yⱼ,
                                       $P, $p, $Aⱼ₊₁, $Aⱼ, $Δⱼ, $result, $tjac, $cfg,
                                       $Jxsto, $Jpsto, $Jx, $Jp)
