"""
$(TYPEDSIGNATURES)

Fast check for to see if the ratio of the L∞ norm is improving in a given iteration
using a hard-code ratio tolerance of 1.01. This is the improvement condition from
Nedialko S. Nedialkov. Computing rigorous bounds on the solution of an initial
value problem for an ordinary differential equation. 1999. Universisty of Toronto,
PhD Dissertation, Algorithm 5.1, page 73-74).
"""
function improvement_condition(Ỹⱼ::Vector{Interval{T}}, Ỹⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    Y0norm = 0.0
    Ynorm = 0.0
    diam1 = 0.0
    diam2 = 0.0
    for i in 1:nx
        diam1 = diam(Ỹⱼ[i])
        diam2 = diam(Ỹⱼ₀[i])
        Ynorm = (diam1 > Ynorm) ? diam1 : Ynorm
        Y0norm = (diam2 > Y0norm) ? diam2 : Y0norm
    end
    return (Ynorm/Y0norm) > 1.01
end

"""
$(TYPEDSIGNATURES)

Checks that an interval vector `Ỹⱼ` of length `nx` is contained in `Ỹⱼ₀`.
"""
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
$(TYPEDSIGNATURES)

Implements the adaptive higher-order enclosure approach detailed in Nedialkov's
dissertation (Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 73-74).
"""
function existence_uniqueness!(s::StepResult, tf!::TaylorFunctor!, hmin::Float64)
    existence_uniqueness!(s.unique_result, tf!, s.Yⱼ, s.hj, hmin, s.f, s.∂f∂x, s.∂f∂p)
    nothing
end
function existence_uniqueness!(out::UniquenessResult, tf!::TaylorFunctor!, Yⱼ::Vector{T},
                               hⱼ::Float64, hmin::Float64, f::Matrix{T},
                               ∂f∂x_in::Vector{Matrix{T}}, ∂f∂p_in::Vector{Matrix{T}}) where {T <: Real}

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

    copyto!(Ỹⱼ₀, 1, Yⱼ, 1, nx+np)
    copyto!(Ỹⱼ, 1, Yⱼ, 1, nx+np)

    println("Ỹⱼ₀ = $(Ỹⱼ₀)")
    println("Ỹⱼ = $(Ỹⱼ)")

    ∂f∂x = tf!.∂f∂x
    hIk = Interval{Float64}(0.0, hⱼ^k)
    println("∂f∂x = $(∂f∂x)")
    println("hIk = $(hIk)")

    inβ = true
    α = 0.8
    ϵ = 1.0
    for i in 1:(k+1)
        for j in eachindex(∂f∂x_in[i])
            ∂f∂x[i][j] = Interval{Float64}(∂f∂x_in[i][j])
        end
    end
    ϵInterval = Interval(-ϵ,ϵ)
    verified  = false

    while ((hⱼ >= hmin) && ~verified) #&& (max_iters > iters)
        #iters += 1
        fill!(Vⱼ, Interval{Float64}(0.0))
        for i in 1:nx
            for j in 1:(k-1)
                Vⱼ[i] += Interval{Float64}(0.0, hⱼ^j)*f[i,j]
            end
        end

        #βⱼⱼ .= (I + Interval{Float64}(0.0, hⱼ^k).*∂f∂y[k])
        βⱼⱼ .= ∂f∂x[k]
        βⱼⱼ .*= hIk
        for i in 1:nx
            βⱼⱼ[i,i] += one(Interval{Float64})
        end

        #βⱼᵥ = f[k,:] .+ ∂f∂y[k]*Vⱼ
        mul!(βⱼᵥ, ∂f∂x[k], Vⱼ)
        for j in 1:nx
            βⱼᵥ[j] += f[j,k]
        end
        mul!(βⱼₖ, βⱼⱼ, βⱼᵥ)

        #βⱼₖ .= βⱼₖ + ϵInterval*abs.(βⱼₖ)
        for j in 1:nx
            Uⱼ[j] = Yⱼ[j] + Vⱼ[j]
            Ỹⱼ₀[j] = Uⱼ[j] + hIk*(βⱼₖ[j] + ϵInterval*abs(βⱼₖ[j]))
        end

        tf!(f̃ₜ, Ỹⱼ₀)
        coeff_to_matrix!(f̃, f̃ₜ, nx, k)
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
        coeff_to_matrix!(f̃, f̃ₜ, nx, k)

        reduced = 0
        while ~verified && reduced < 2
            s = 0
            for l = 1:k
                fill!(Vⱼ, Interval{Float64}(0.0))
                for j in 1:nx
                    for i in 1:l
                        Vⱼ[j] += Interval{Float64}(0.0, hⱼ^i)*f[j,i]
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
    out.fk .= f̃[:,k]
    out.fk .*= hⱼ^k
    out.step = hⱼ
    out.confirmed = flag
    out.Y .= Ỹⱼ

    nothing
end
