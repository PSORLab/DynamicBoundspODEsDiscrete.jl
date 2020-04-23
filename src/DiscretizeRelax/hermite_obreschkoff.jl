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
function (x::LohnersFunctor{F,K,S,T,NY,})(hⱼ::Float64, X̃ⱼ, Xⱼ, xⱼ, A, Δⱼ, P, rP, p, t) where {F <: Function, K, S <: Real, T <: Real, NY}

    # Compute lohner function step
    out.status_flag = lf(out.hj, out.unique_result.X, out.Xⱼ, out.xⱼ, A, Δ, P, rP, p, t)

    X̂0ₖ₊₁ = out.Xⱼ
    x̂0ₖ₊₁ = mid.(X̂0ₖ₊₁)
    real_tf!(rf̃, x̂0ₖ₊₁, p, t)
    prf̃ = out.xⱼ
    srf̃ = copy(x̂0ₖ₊₁)
    for i=2:k
        cpq = ho.cpq[i]
        cqp = ho.cqp[i]
        for j=1:nx
            prf̃[j] += cpq*x.real_tf!.f̃[i][j]
            srf̃[j] += ((-1)^(i-1))*cqp*rf̃[i][j]
        end
    end
    @__dot__ gₖ₊₁ = out.xⱼ - x̂0ₖ₊₁ + rf + prf̃ - srf̃ + fk + γ*out.fk
    @__dot__ Vₖ = X̂0ₖ₊₁ - x̂0ₖ₊₁
    set_JxJp!(jac_tf!, X̂0ₖ₊₁, P, t)

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

    Rₖ₊₁ =
    δₖ₊₁ = vₖ₊₁ - vₖ + Rₖ₊₁
    mJx = mid(Jxsto)
    Bₖ₊₁ = mJx*(Jxsto*Aₖ)
    Cₖ₊₁ = mJx*Jxsto1
    Xₖ₊₁ = X + Bₖ₊₁*Δₖ + Cₖ₊₁*(X0ₖ₊₁ - x̂0ₖ₊₁) + δₖ₊₁ +
    xₖ₊₁ = mid.(Xₖ₊₁)
    Q = inv(Aⱼ₊₁)
    Δₖ₊₁ = (Q*Bₖ₊₁)*Δₖ + (Q*Cₖ₊₁)*(X0ₖ₊₁ - x̂0ₖ₊₁) + Q*Jpsto*rP - Q*Jpsto1*rP + Q*inv(mid(Jxsto1)*δₖ₊₁
    nothing
end
