"""
$(TYPEDSIGNATURES)

Fast check for to see if the ratio of the L∞ norm is improving in a given iteration
using a hard-code ratio tolerance of 1.01. This is the improvement condition from
Nedialko S. Nedialkov. Computing rigorous bounds on the solution of an initial
value problem for an ordinary differential equation. 1999. Universisty of Toronto,
PhD Dissertation, Algorithm 5.1, page 73-74).
"""
function improvement_condition(X̃ⱼ::Vector{Interval{T}}, X̃ⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    Y0norm = 0.0
    Ynorm = 0.0
    diam1 = 0.0
    diam2 = 0.0
    for i in 1:nx
        diam1 = diam(X̃ⱼ[i])
        diam2 = diam(X̃ⱼ₀[i])
        Ynorm = (diam1 > Ynorm) ? diam1 : Ynorm
        Y0norm = (diam2 > Y0norm) ? diam2 : Y0norm
    end
    println("improvement ratio: $(Ynorm/Y0norm)")
    return (Ynorm/Y0norm) > 1.01
end

"""
$(TYPEDSIGNATURES)

Checks that an interval vector `X̃ⱼ` of length `nx` is contained in `X̃ⱼ₀`.
"""
function contains(X̃ⱼ::Vector{Interval{T}}, X̃ⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    flag = true
    for i in 1:nx
        if ~(X̃ⱼ[i] ⊆ X̃ⱼ₀[i])
            flag = false
            break
        end
    end
    flag
end

"""
$(TYPEDSIGNATURES)

Implements the adaptive higher-order enclosure approach detailed in Nedialkov's
dissertation (Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 73-74).
"""
function existence_uniqueness!(s::StepResult{T}, tf!::TaylorFunctor!{F,K,S,T}, hmin::Float64, P, t) where {F, K, S, T <: Number}
    existence_uniqueness!(s.unique_result, tf!, s.Xⱼ, s.hj, hmin, s.f, s.∂f∂x, s.∂f∂p, P, s.h, t)
    nothing
end
function existence_uniqueness!(out::UniquenessResult{T}, tf!::TaylorFunctor!{F,K,S,T}, Xⱼ::Vector{T},
                               hⱼ::Float64, hmin::Float64, f::Matrix{T},
                               ∂f∂x_in::Vector{Matrix{T}}, ∂f∂p_in::Vector{Matrix{T}},
                               P::Vector{T}, hfixed::Float64, t::Float64) where {F, K, S, T <: Real}

    #println("start existence and uniqueness kernel")
    #println("start of Xⱼ: $(Xⱼ)")
    np = tf!.np
    Vⱼ = tf!.Vⱼ
    f̃ = tf!.f̃
    X̃ⱼ₀ = tf!.X̃ⱼ₀
    X̃ⱼ = tf!.X̃ⱼ
    βⱼⱼ = tf!.βⱼⱼ
    βⱼᵥ = tf!.βⱼᵥ
    βⱼₖ = tf!.βⱼₖ
    Uⱼ = tf!.Uⱼ

    copyto!(X̃ⱼ₀, 1, Xⱼ, 1, tf!.nx)
    copyto!(X̃ⱼ, 1, Xⱼ, 1, tf!.nx)

    ∂f∂x = tf!.∂f∂x
    if hfixed >= 0.0
        hⱼ = hfixed
    end
    hIk = Interval{Float64}(0.0, hⱼ^tf!.k)

    for i=1:(tf!.k+1)
        for j in eachindex(∂f∂x_in[i])
            ∂f∂x[i][j] = Interval{Float64}(∂f∂x_in[i][j])
        end
    end

    ϵInterval = Interval(-1.0, 1.0)
    verified  = false

    if hfixed <= 0.0
        while ((hⱼ >= hmin) && ~verified) #&& (max_iters > iters)
            #iters += 1
            tf!(f, X̃ⱼ, P, t)
            fill!(Vⱼ, Interval{Float64}(0.0))
            @__dot__ Vⱼ = X̃ⱼ
            for j in 2:(tf!.k)
                @__dot__ Vⱼ += Interval{Float64}(0.0, hⱼ^(j-1))*f[j]
            end

            #βⱼⱼ .= (I + Interval{Float64}(0.0, hⱼ^k).*∂f∂y[k])
            βⱼⱼ .= ∂f∂x[tf!.k]
            βⱼⱼ .*= hIk
            for i in 1:tf!.nx
                βⱼⱼ[i,i] += one(Interval{Float64})
            end

            #βⱼᵥ = f[k,:] .+ ∂f∂y[k]*Vⱼ
            mul!(βⱼᵥ, ∂f∂x[tf!.k], Vⱼ)
            βⱼᵥ .+= f[tf!.k]
            mul!(βⱼₖ, βⱼⱼ, βⱼᵥ)
            #println("βⱼₖ: $(βⱼₖ)")

            #βⱼₖ .= βⱼₖ + ϵInterval*abs.(βⱼₖ)
            @__dot__ Uⱼ = Xⱼ + Vⱼ
            @__dot__ X̃ⱼ₀ = Uⱼ + hIk*(βⱼₖ + ϵInterval*abs(βⱼₖ))

            tf!(f̃, X̃ⱼ₀, P, t)
            if contains(f̃[tf!.k], βⱼₖ, tf!.nx)
                @__dot__ X̃ⱼ = Uⱼ + hIk*f̃[tf!.k]
                break
            end
            @__dot__ X̃ⱼ₀ = Uⱼ + hIk*f̃[tf!.k]
            tf!(f̃, X̃ⱼ₀, P, t)

            reduced = 0
            while ~verified && reduced < 2
                s = 0
                for l = 1:tf!.k
                    fill!(Vⱼ, Interval{Float64}(0.0))
                    @__dot__ Vⱼ = X̃ⱼ₀
                    for i in 2:l
                        @__dot__ Vⱼ += Interval{Float64}(0.0, hⱼ^(i-1))*f[i]
                    end
                    @__dot__ X̃ⱼ  = Uⱼ + hIk*f̃[l]
                    if contains(X̃ⱼ, X̃ⱼ₀, tf!.nx)
                        verified = true
                        s = l
                        break
                    end
                end

                if verified
                    improving = true
                    while improving
                        tf!(f̃, X̃ⱼ, P, t)
                        #println("next B f̃: $(f̃)")
                        for j in 2:tf!.k
                            X̃ⱼ₀  = Vⱼ + Interval{Float64}(0.0, hⱼ^(s-1))*f̃[j]
                        end
                        if improvement_condition(X̃ⱼ, X̃ⱼ₀, tf!.nx)
                            copyto!(X̃ⱼ, 1, X̃ⱼ₀, 1, tf!.nx)
                        else
                            improving = false
                        end
                    end
                else
                    hⱼ *= 0.8                              # times alpha value
                    hIk = Interval{Float64}(0.0, hⱼ^tf!.k)
                    reduced += 1
                end
            end
        end
    else
        # compute taylor coefficients
        tf!(f̃, X̃ⱼ, P, t)

        # set X to sum of Taylor cofficients
        fill!(Vⱼ, Interval{Float64}(0.0))
        copyto!(Vⱼ, X̃ⱼ)
        for i=2:tf!.k
            for j in eachindex(f̃[1])
                Vⱼ[j] += f̃[i][j]*Interval{Float64}(0.0, hⱼ^(i-1))
            end
        end
        copy!(X̃ⱼ, Vⱼ)
    end
    flag = hⱼ > hmin
    out.fk .= f̃[tf!.k+1]
    out.fk .*= Interval{Float64}(0.0,hⱼ^tf!.k)

    out.step = hⱼ
    out.confirmed = flag
    out.X .= X̃ⱼ
    nothing
end
