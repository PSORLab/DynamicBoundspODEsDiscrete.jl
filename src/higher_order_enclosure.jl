"""
Fast check for the follow condition norm(diam.(Ỹⱼ[1:nx]), Inf)/norm(diam.(Ỹⱼ₀[1:nx]), Inf) > 1.01
"""
function improvement_condition(Ỹⱼ::Vector{Interval{T}}, Ỹⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    Y0norm = 0.0
    Ynorm = 0.0
    for i in 1:nx
        Ynorm = max(diam(Ỹⱼ[i]), Ynorm)
        Y0norm = max(diam(Ỹⱼ₀[i]), Y0norm)
    end
    return (Ynorm/Y0norm) > 1.01
end

function contains(Ỹⱼ::Vector{Interval{T}}, Ỹⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    flag = true
    for i in 1:nx
        if ~(Ỹⱼ[i] ⊆ Ỹⱼ₀[i])
            flag = false
            break
        end
    end
    flag
end

"""

Implements the adaptive higher-order enclosure approach detailed in
Nedyalkov's dissertation (Algorithm 5.1).
"""
function existence_uniqueness(tf!::TaylorFunctor!, Yⱼ::Vector{T},
                              P::Vector{T}, hⱼ::Float64, hmin::Float64, f::Matrix{T},
                              ∂f∂y_in) where {T <: Real}

    k = tf!.s
    nx = tf!.nx
    np = tf!.np
    Vⱼ = tf!.Vⱼ
    f̃ₜ = tf!.f̃ₜ
    f̃ = tf!.f̃
    Ỹⱼ₀ = tf!.Ỹⱼ₀
    Ỹⱼ = tf!.Ỹⱼ
    βⱼⱼ = tf!.βⱼⱼ
    βⱼᵥ = tf!.βⱼᵥ
    βⱼₖ = tf!.βⱼₖ
    Uⱼ = tf!.Uⱼ

    copyto!(Ỹⱼ₀, 1, Yⱼ, 1, nx)
    copyto!(Ỹⱼ, 1, Yⱼ, 1, nx)
    copyto!(Ỹⱼ₀, 1+nx, P, 1, np)
    copyto!(Ỹⱼ, 1+nx, P, 1, np)

    ∂f∂y = tf!.∂f∂y
    hIk = hIk = Interval{Float64}(0.0, hⱼ^k)

    α = 0.8
    ϵ = 1.0
    for i in 1:(k+1)
        for j in eachindex(∂f∂y_in[i])
            ∂f∂y[i][j] = Interval{Float64}(∂f∂y_in[i][j])
        end
    end
    ϵInterval = Interval(-ϵ,ϵ)
    verified  = false

    while ((hⱼ >= hmin) && ~verified) #&& (max_iters > iters)
        #iters += 1
        for j in 1:nx
            Vⱼ[j] = Interval{Float64}(0.0, hⱼ)*f[1,j]
            for i in 2:(k-1)
                Vⱼ[j] += Interval{Float64}(0.0, hⱼ^i)*f[i,j]
            end
        end

        #βⱼⱼ .= (I + Interval{Float64}(0.0, hⱼ^k).*∂f∂y[k])
        βⱼⱼ .= ∂f∂y[k]
        βⱼⱼ .*= hIk
        for i in 1:nx
            βⱼⱼ[i,i] += one(Interval{Float64})
        end

        #βⱼᵥ = f[k,:] .+ ∂f∂y[k]*Vⱼ
        mul!(βⱼᵥ, ∂f∂y[k], Vⱼ)
        for j in 1:nx
            βⱼᵥ[j] += f[k,j]
        end
        mul!(βⱼₖ, βⱼⱼ, βⱼᵥ)

        #βⱼₖ .= βⱼₖ + ϵInterval*abs.(βⱼₖ)
        for j in 1:nx
            Uⱼ[j] = Yⱼ[j] + Vⱼ[j]
            Ỹⱼ₀[j] = Uⱼ[j] + hIk*(βⱼₖ[j] + ϵInterval*abs(βⱼₖ[j]))
        end

        tf!(f̃ₜ, Ỹⱼ₀)
        coeff_to_matrix!(f̃, f̃ₜ, nx, s)
        inβ = true
        for j in 1:nx
            if ~(f̃[j,k] ⊆ βⱼₖ[j])
                inβ = false
                break
            end
        end
        if inβ
            for j in 1:nx
                Ỹⱼ[j] = Uⱼ[j] + hIk*f̃[j,k]
            end
            break
        end
        for j in 1:nx
            Ỹⱼ₀[j] = Uⱼ[j] + hIk*f̃[j,k]
        end
        tf!(f̃ₜ, Ỹⱼ₀)
        coeff_to_matrix!(f̃, f̃ₜ, nx, s)

        reduced = 0
        while ~verified && reduced < 2
            s = 0
            for l = 1:k
                for j in 1:l
                    Vⱼ[j] = Interval{Float64}(0.0, hⱼ)*f[1,j]
                    for i in 2:(k-1)
                        Vⱼ[j] += Interval{Float64}(0.0, hⱼ^i)*f[i,j]
                    end
                end
                for j in 1:nx
                    Ỹⱼ[j]  = Uⱼ[j] + hIk*f̃[j,l]
                end
                if contains(Ỹⱼ, Ỹⱼ₀, nx)
                    verified = true
                    s = l
                    break
                end
            end

            if verified
                improving = true
                while improving
                    tf!(f̃ₜ, Ỹⱼ)
                    coeff_to_matrix!(f̃, f̃ₜ, nx, s)
                    for j in 1:nx
                        Ỹⱼ₀[j]  = Vⱼ[j] + Interval{Float64}(0.0, hⱼ^s)*f̃[j,s]
                    end
                    if improvement_condition(Ỹⱼ, Ỹⱼ₀, nx)
                        copyto!(Ỹⱼ, 1, Ỹⱼ₀, 1, nx)
                    else
                        improving = false
                    end
                end
            else
                hⱼ *= α
                hIk = Interval{Float64}(0.0, hⱼ^k)
                reduced += 1
            end
        end
    end

    flag = hⱼ > hmin
    tf!(f̃ₜ, Ỹⱼ)
    coeff_to_matrix!(f̃, f̃ₜ, nx, k)
    f̃[k] *= hⱼ^k
    return hⱼ, Ỹⱼ, f̃[k], flag
end

hⱼ = 0.001
hmin = 0.00001
Yⱼ = [Interval(0.1, 5.1); Interval(0.1, 8.9)]
P = [Interval(0.1, 5.1); Interval(0.1, 8.9)]
existence_uniqueness(jacobianfunctor, Yⱼ, P, hⱼ, hmin, routIntv, Jx)
#@btime improvement_condition($Yⱼ, $Yⱼ, $nx)
#@btime existence_uniqueness($jacobianfunctor, $Yⱼ, $P, $hⱼ, $hmin, $routIntv, $Jx)
#tv, xv = validated_integration(f!, Interval{Float64}.([3.0, 3.0]), 0.0, 0.3, 4, 1.0e-20, maxsteps=100 )
Q = [Yⱼ; P]
#@btime jacobianfunctor($outIntv, $yInterval)

d = jacobianfunctor
zqwa = d.g!
zqwb = d.t
zqwc = d.xtaylor
zqwd = d.xout
zqwe = d.xaux
zqwr = d.taux
@btime jetcoeffs!($zqwa, $zqwb, $zqwc, $zqwd, $zqwe, $zqwr, $s, $p)
#@code_warntype jetcoeffs!(zqwa, zqwb, zqwc, zqwd, zqwe, zqwr, p)
