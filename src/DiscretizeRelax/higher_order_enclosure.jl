"""
$(FUNCTIONNAME)

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
    return (Ynorm/Y0norm) > 1.01
end

"""
$(FUNCTIONNAME)

Checks that an interval vector `X̃ⱼ` of length `nx` is contained in `X̃ⱼ₀`.
"""
function contains(X̃ⱼ::Vector{Interval{T}}, X̃ⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    flag = true
    for i = 1:nx
        if ~(X̃ⱼ[i] ⊆ X̃ⱼ₀[i])
            flag = false
            break
        end
    end
    flag
end

"""
$(FUNCTIONNAME)

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
                               hⱼ::Float64, hmin::Float64, f::Vector{Vector{T}},
                               ∂f∂x_in::Vector{Matrix{T}}, ∂f∂p_in::Vector{Matrix{T}},
                               P::Vector{T}, hfixed::Float64, t::Float64) where {F, K, S, T <: Real}

    println(" ")
    println("hⱼ in = $(hⱼ), Xⱼ = $(Xⱼ)")

    np = tf!.np
    Vⱼ = tf!.Vⱼ
    f̃ = tf!.f̃
    #println("f̃: $(f̃)")
    X̃ⱼ₀ = tf!.X̃ⱼ₀
    X̃ⱼ = tf!.X̃ⱼ
    βⱼⱼ = tf!.βⱼⱼ
    βⱼᵥ = tf!.βⱼᵥ
    βⱼₖ = tf!.βⱼₖ
    Uⱼ = tf!.Uⱼ
    k = tf!.k
    #println("k: $(k)")

    unsafe_copyto!(X̃ⱼ₀, 1, Xⱼ, 1, tf!.nx)
    unsafe_copyto!(X̃ⱼ, 1, Xⱼ, 1, tf!.nx)

    #println("X̃ⱼ₀ = $(X̃ⱼ₀)")
    #println("X̃ⱼ = $(X̃ⱼ)")

    ∂f∂x = tf!.∂f∂x
    if hfixed > 0.0
        hⱼ = hfixed
    end

    #println("∂f∂x_asdsaadin: $(∂f∂x_in)")

    hIk = Interval{Float64}(0.0, hⱼ^k)

    for i = 1:k+1
        ∂f∂x[i] .= ∂f∂x_in[i]
    end
    #println("∂f∂x: $(∂f∂x)")

    ϵInterval = Interval(-1.0, 1.0)
    verified  = false

    if hfixed <= 0.0
        #println("pre while: hⱼ = $(hⱼ), hmin = $(hmin), verified = $(verified)")
        while ((hⱼ >= hmin) && ~verified) #&& (max_iters > iters)
            #println("outer while: hⱼ = $(hⱼ), hmin = $(hmin), verified = $(verified)")
            #iters += 1
            tf!(f, Xⱼ, P, t)
            fill!(Vⱼ, zero(S))
            for i = 2:k
                @__dot__ Vⱼ += Interval{Float64}(0.0, hⱼ^(i-1))*f[i]
            end

            #βⱼⱼ .= (I + Interval{Float64}(0.0, hⱼ^k).*∂f∂y[k])
            βⱼⱼ .= ∂f∂x[k+1]
            βⱼⱼ .*= hIk
            βⱼⱼ += I

            #βⱼᵥ = f[k,:] .+ ∂f∂y[k]*Vⱼ
            mul!(βⱼᵥ, ∂f∂x[k+1], Vⱼ)
            βⱼᵥ .+= f[k+1]
            mul!(βⱼₖ, βⱼⱼ, βⱼᵥ)
            @__dot__ βⱼₖ += ϵInterval*abs(βⱼₖ)

            #βⱼₖ .= βⱼₖ + ϵInterval*abs.(βⱼₖ)
            @__dot__ Uⱼ = Xⱼ + Vⱼ
            @__dot__ X̃ⱼ₀ = Uⱼ + hIk*βⱼₖ

            tf!(f̃, X̃ⱼ₀, P, t)
            if contains(f̃[k + 1], βⱼₖ, tf!.nx)
                @__dot__ X̃ⱼ = Uⱼ + hIk*f̃[k+1]
                break
            end
            @__dot__ X̃ⱼ₀ = Uⱼ + hIk*f̃[k+1]
            #println("X̃ⱼ₀: $(X̃ⱼ₀)")
            tf!(f̃, X̃ⱼ₀, P, t)
        #    println("f̃: $(f̃)")

            reduced = 0
            s = 0
            while ~verified && reduced < 50
                for l = 1:k
                    @__dot__ Vⱼ = Xⱼ
                    for i = 1:l-1
                        @__dot__ Vⱼ += Interval{Float64}(0.0, hⱼ^i)*f[i+1]
                    end
                    @__dot__ X̃ⱼ = Vⱼ + Interval{Float64}(0.0, hⱼ^l)*f̃[l+1]
                    println("check 1 X̃ⱼ = $(X̃ⱼ), X̃ⱼ₀ = $(X̃ⱼ₀)")
                    if contains(X̃ⱼ, X̃ⱼ₀, tf!.nx)
                        verified = true
                        s = l
                        break
                    end
                end

                if verified
                    #println("if branch verified")
                    improving = true
                    while improving
                        tf!(f̃, X̃ⱼ, P, t)
                        #println("next B f̃: $(f̃)")
                        X̃ⱼ₀  = Vⱼ + Interval{Float64}(0.0, hⱼ^s)*f̃[s+1]
                        println("X̃ⱼ = $(X̃ⱼ), X̃ⱼ₀ = $(X̃ⱼ₀)")
                        if improvement_condition(X̃ⱼ, X̃ⱼ₀, tf!.nx)
                            #println("copy to...")
                            copyto!(X̃ⱼ, 1, X̃ⱼ₀, 1, tf!.nx)
                        else
                            #println("not improving")
                            improving = false
                        end
                    end
                else
                    #println("else branch verified")
                    hⱼ *= 0.7
                    hIk = Interval{Float64}(0.0, hⱼ^tf!.k)
                    reduced += 1
                    println("hⱼ = $(hⱼ), reduced = $(reduced)")
                end
            end
        end
    else
        # compute taylor coefficients
        tf!(f̃, X̃ⱼ, P, t)

        # set X to sum of Taylor cofficients
        fill!(Vⱼ, zero(T))
        copyto!(Vⱼ, X̃ⱼ)
        for i = 2:k
            for j in eachindex(f̃[1])
                Vⱼ[j] += f̃[i][j]*Interval{Float64}(0.0, hⱼ^(i-1))
            end
        end
        copy!(X̃ⱼ, Vⱼ)
    end
    #println("predicted step hⱼ: $(hⱼ), X̃ⱼ = $(X̃ⱼ)")
    #println("hmin: $(hmin)")
    flag = hⱼ > hmin
    out.fk .= f̃[k+1]
    #out.fk .*= Interval{Float64}(0.0,hⱼ^k)

    out.step = hⱼ
    out.confirmed = flag
    out.X .= X̃ⱼ
    println("hⱼ = $(hⱼ), X̃ⱼ = $(X̃ⱼ)")
    nothing
end

#=
"""
$(TYPEDSIGNATURES)
Implements a golden section search for step size
"""
function existence_uniquenes(s::StepResult{T}, tf!::TaylorFunctor!{F,K,S,T}, hmin::Float64, P, t) where {F, K, S, T <: Number}

    np = tf!.np
    Vⱼ = tf!.Vⱼ
    f̃ = tf!.f̃
    X̃ⱼ₀ = tf!.X̃ⱼ₀
    X̃ⱼ = tf!.X̃ⱼ
    k = tf!.k
    iteration_limit = 50

    unsafe_copyto!(X̃ⱼ₀, 1, Xⱼ, 1, tf!.nx)
    unsafe_copyto!(X̃ⱼ, 1, Xⱼ, 1, tf!.nx)

    hⱼ = (hfixed > 0.0) ? hfixed : hⱼ
    hIk = Interval{Float64}(0.0, hⱼ^k)

    # compute taylor coefficient sum
    tf!(f̃, X̃ⱼ, P, t)
    fill!(Vⱼ, zero(T))
    copyto!(Vⱼ, X̃ⱼ)
    for i = 2:k+1
        Vⱼ .+= f̃[i]*Interval{Float64}(0.0, hⱼ^(i-1))
    end
    copyto!(X̃ⱼ, Vⱼ)

    count = 0
    confirmed_flag = contains(X̃ⱼ, X̃ⱼ₀, nx)
    convergence_flag = improvement_condition(X̃ⱼ, X̃ⱼ₀, nx)

    if hfixed <= 0.0 && !improvement_flag
        while confirmed_flag && convergence_flag && count < iteration_limit
            count += 1
            confirmed_flag = contains(X̃ⱼ, X̃ⱼ₀, nx)
            convergence_flag = improvement_condition(X̃ⱼ, X̃ⱼ₀, nx)
        end
    end
    flag = hⱼ > hmin
    out.fk .= f̃[k+1]
    out.step = hⱼ
    out.confirmed = confirmed_flag
    out.X .= X̃ⱼ

    nothing
end
=#
