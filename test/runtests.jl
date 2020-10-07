#!/usr/bin/env julia
using Test, DynamicBoundspODEsDiscrete, IntervalArithmetic, DynamicBoundsBase
using DynamicBoundspODEsDiscrete.StaticTaylorSeries
using DiffResults: JacobianResult

const DBB = DynamicBoundsBase
const DR = DynamicBoundspODEsDiscrete

@testset "Discretize and Relax" begin

    # test improvement condition for existence & uniqueness
    Yold = [Interval(1.0, 3.0); Interval(2.0, 4.0); Interval(1.0, 3.0)]
    Ynew = [Interval(1.5, 2.0); Interval(3.0, 3.5); Interval(0.5, 3.5)]
    nx_ic = 2
    @test DR.improvement_condition(Yold, Ynew, nx_ic)

    # construct storage for QR factorizations
    storage = DR.QRDenseStorage(nx_ic)
    @test storage.factorization.Q[1, 1] == -1.0
    @test storage.factorization.Q[2, 2] == 1.0
    @test storage.factorization.R[1, 1] == -1.0
    @test storage.factorization.R[2, 2] == 1.0

    DR.calculateQ!(storage, [1.0 3.0; 2.0 1.0], nx_ic)
    @test isapprox(storage.Q[1, 1], -0.447214, atol = 1E-3)
    @test isapprox(storage.Q[1, 2], -0.894427, atol = 1E-3)
    @test isapprox(storage.Q[2, 1], -0.894427, atol = 1E-3)
    @test isapprox(storage.Q[2, 2], 0.4472135, atol = 1E-3)

    # results in symmetric matrix
    DR.calculateQinv!(storage)
    @test storage.inv[1, 1] == storage.Q[1, 1]
    @test storage.inv[1, 2] == storage.Q[1, 2]
    @test storage.inv[2, 1] == storage.Q[2, 1]
    @test storage.inv[2, 2] == storage.Q[2, 2]

    stack = DR.qr_stack(nx_ic, 3)
    DR.reinitialize!(stack)
    @test stack[1].Q[1, 1] == 1.0
    @test stack[1].Q[1, 2] == 0.0
    @test stack[1].Q[2, 1] == 0.0
    @test stack[1].Q[2, 2] == 1.0

    # eval_cycle! test
    buffs = DR.CircularBuffer(zeros(2, 2), 3)

    function J!(out, x, p, t)
        out[1, 1] = x
        nothing
    end

    DR.eval_cycle!(J!, buffs, 1.0, 0.0, 0.0)
    DR.eval_cycle!(J!, buffs, 2.0, 0.0, 0.0)
    DR.eval_cycle!(J!, buffs, 3.0, 0.0, 0.0)
    @test buffs.buffer[1][1, 1] == 3.0
    @test buffs.buffer[2][1, 1] == 2.0
    @test buffs.buffer[3][1, 1] == 1.0

    function f!(dx, x, p, t)
        dx[1] = x[1]^2 + p[2]
        dx[2] = x[2] + p[1]^2
        nothing
    end
    np = 2
    nx = 2
    k = 3
    x = [0.1; 1.0]
    p = [0.2; 0.1]
    jtf! = DR.JacTaylorFunctor!(f!, nx, np, Val(k), Interval{Float64}(0.0), 0.0)

    xIntv = Interval{Float64}.(x)
    pIntv = Interval{Float64}.(p)
    yIntv = [xIntv; pIntv]
    DR.jacobian_taylor_coeffs!(jtf!, xIntv, pIntv, 0.0)

    jac = JacobianResult(jtf!.out, yIntv).derivs[1]
    tjac = zeros(Interval{Float64}, nx + np, nx * (k + 1))
    Jx = Matrix{Interval{Float64}}[
        zeros(Interval{Float64}, nx, nx) for i = 1:(k+1)
    ]
    Jp = Matrix{Interval{Float64}}[
        zeros(Interval{Float64}, nx, np) for i = 1:(k+1)
    ]

    DR.set_JxJp!(jtf!, xIntv, pIntv, 0.0)
    @test isapprox(jtf!.Jp[2][2, 1].lo, 0.4, atol = 1E-3)
    @test isapprox(jtf!.Jp[2][1, 2].lo, 1.0, atol = 1E-3)
    @test isapprox(jtf!.Jp[4][2, 1].lo, 0.0666666, atol = 1E-3)
    @test isapprox(jtf!.Jp[4][1, 2].lo, 0.079999, atol = 1E-3)
    @test isapprox(jtf!.Jx[2][1, 1].lo, 0.2, atol = 1E-3)
    @test isapprox(jtf!.Jx[2][2, 2].lo, 1.0, atol = 1E-3)
    @test isapprox(jtf!.Jx[4][1, 1].lo, 0.030666, atol = 1E-3)
    @test isapprox(jtf!.Jx[4][2, 2].lo, 0.1666, atol = 1E-3)

    # make/evaluate interval valued Taylor cofficient functor
    itf! = DR.TaylorFunctor!(
        f!,
        nx,
        np,
        Val(k),
        zero(Interval{Float64}),
        zero(Float64),
    )
    outIntv = Vector{Interval{Float64}}[zeros(Interval{Float64}, 2) for i = 1:4]
    itf!(outIntv, xIntv, pIntv, 0.0)
    @test isapprox(outIntv[1][1].lo, 0.10001, atol = 1E-3)
    @test isapprox(outIntv[2][2].lo, 1.0399999999999998, atol = 1E-3)
    @test isapprox(outIntv[3][1].lo, 0.011, atol = 1E-3)
    @test isapprox(outIntv[4][2].lo, 0.173334, atol = 1E-3)
    @test isapprox(outIntv[1][2].hi, 1.0, atol = 1E-3)
    @test isapprox(outIntv[2][1].hi, 0.1100000000000000, atol = 1E-3)
    @test isapprox(outIntv[3][2].hi, 0.52, atol = 1E-3)
    @test isapprox(outIntv[4][1].hi, 0.004766666666666669, atol = 1E-3)

    # make/evaluate real valued Taylor cofficient functor
    rtf! = DR.TaylorFunctor!(f!, nx, np, Val(k), zero(Float64), zero(Float64))
    out = Vector{Float64}[zeros(Float64, 2) for i = 1:4]
    rtf!(out, x, p, 0.0)
    @test isapprox(out[1][1], 0.10001, atol = 1E-3)
    @test isapprox(out[2][1], 0.11000000000000001, atol = 1E-3)
    @test isapprox(out[3][1], 0.011, atol = 1E-3)
    @test isapprox(out[4][1], 0.004766666666666668, atol = 1E-3)
    @test isapprox(out[1][2], 1.0, atol = 1E-3)
    @test isapprox(out[2][2], 1.04, atol = 1E-3)
    @test isapprox(out[3][2], 0.52, atol = 1E-3)
    @test isapprox(out[4][2], 0.17333333333333334, atol = 1E-3)

    # higher order existence tests
    hⱼ = 0.001
    hmin = 0.00001
    function euf!(out, x, p, t)
        out[1, 1] = -x[1]^2
        nothing
    end

    jtf_exist_unique! =
        DR.JacTaylorFunctor!(euf!, 1, 1, Val(k), Interval{Float64}(0.0), 0.0)
    xIntv_plus = xIntv .+ Interval(0, 1)
    DR.jacobian_taylor_coeffs!(jtf_exist_unique!, xIntv_plus, pIntv, 0.0)
    @test isapprox(
        jtf_exist_unique!.result.value[4].lo,
        -1.464100000000001,
        atol = 1E-5,
    )
    @test isapprox(
        jtf_exist_unique!.result.value[4].hi,
        -9.999999999999999e-5,
        atol = 1E-5,
    )
    @test isapprox(
        jtf_exist_unique!.result.derivs[1][3, 1].lo,
        0.03,
        atol = 1E-5,
    )
    @test isapprox(
        jtf_exist_unique!.result.derivs[1][3, 1].hi,
        3.6300000000000012,
        atol = 1E-5,
    )

    coeff_out = zeros(Interval{Float64}, 1, k)
    DR.set_JxJp!(jtf_exist_unique!, xIntv_plus, pIntv, 0.0)
    @test isapprox(jtf_exist_unique!.Jx[4][1, 1].lo, -5.32401, atol = 1E-5)
    @test isapprox(jtf_exist_unique!.Jx[4][1, 1].hi, -0.00399999, atol = 1E-5)
    @test jtf_exist_unique!.Jp[1][1, 1].lo ==
          jtf_exist_unique!.Jp[1][1, 1].hi ==
          0.0

    #u_result = DR.UniquenessResult(1,1)
    #DR.existence_uniqueness!(u_result, itf_exist_unique!, eufY, hⱼ, hmin,
    #                         coeff_out_exist_unique!, Jx, Jp)

    #@test u_result.step == 0.001
    #@test u_result.confirmed

    Y = [1.1 3.2; 4.0 -1.0]
    y = [1.1; 3.2]
    A = [1.1 3.2; 4.0 -1.0]
    B = [1.1 3.2; 4.0 -1.0]
    b = [1.1; -1.0]
    mul_split!(Y, A, B, 2)
    @test isapprox(Y[2,2], 13.8, atol=1E-5)
    mul_split!(y, A, b, 2)
    @test isapprox(Y[1,2], 0.3200000000000003, atol=1E-5)
end

if !(VERSION < v"1.1" && testfile == "intervals.jl")
    using TaylorSeries, IntervalArithmetic

    function test_vs_Taylor1(x, y)
        flag = true
        for i = 0:2
            if x[i] !== y[i]
                flag = false
                break
            end
        end
        flag
    end

    @testset "Tests for STaylor1 expansions" begin

        @test STaylor1 <: AbstractSeries
        @test STaylor1{1,Float64} <: AbstractSeries{Float64}
        @test STaylor1([1.0, 2.0]) == STaylor1((1.0, 2.0))
        @test STaylor1(STaylor1((1.0, 2.0))) == STaylor1((1.0, 2.0))
        @test STaylor1(1.0, Val(2)) == STaylor1((1.0, 0.0, 0.0))

        @test +STaylor1([1.0, 2.0, 3.0]) == STaylor1([1.0, 2.0, 3.0])
        @test -STaylor1([1.0, 2.0, 3.0]) == -STaylor1([1.0, 2.0, 3.0])
        @test STaylor1([1.0, 2.0, 3.0]) + STaylor1([3.0, 2.0, 3.0]) ==
              STaylor1([4.0, 4.0, 6.0])
        @test STaylor1([1.0, 2.0, 3.0]) - STaylor1([3.0, 2.0, 4.0]) ==
              STaylor1([-2.0, 0.0, -1.0])
        @test STaylor1([1.0, 2.0, 3.0]) + 2.0 == STaylor1([3.0, 2.0, 3.0])
        @test STaylor1([1.0, 2.0, 3.0]) - 2.0 == STaylor1([-1.0, 2.0, 3.0])
        @test 2.0 + STaylor1([1.0, 2.0, 3.0]) == STaylor1([3.0, 2.0, 3.0])
        @test 2.0 - STaylor1([1.0, 2.0, 3.0]) == STaylor1([1.0, -2.0, -3.0])

        @test zero(STaylor1([1.0, 2.0, 3.0])) == STaylor1([0.0, 0.0, 0.0])
        @test one(STaylor1([1.0, 2.0, 3.0])) == STaylor1([1.0, 0.0, 0.0])

        @test isinf(STaylor1([Inf, 2.0, 3.0])) &&
              ~isinf(STaylor1([0.0, 0.0, 0.0]))
        @test isnan(STaylor1([NaN, 2.0, 3.0])) &&
              ~isnan(STaylor1([1.0, 0.0, 0.0]))
        @test iszero(STaylor1([0.0, 0.0, 0.0])) &&
              ~iszero(STaylor1([0.0, 1.0, 0.0]))

        @test length(STaylor1([0.0, 0.0, 0.0])) == 3
        @test size(STaylor1([0.0, 0.0, 0.0])) == 3
        @test firstindex(STaylor1([0.0, 0.0, 0.0])) == 0
        @test lastindex(STaylor1([0.0, 0.0, 0.0])) == 2

        st1 = STaylor1([1.0, 2.0, 3.0])
        @test st1(2.0) == 41.0
        @test st1() == 1.00
        st2 = typeof(st1)[st1; st1]
        @test st2(2.0)[1] == st2(2.0)[2] == 41.0
        @test st2()[1] == st2()[2] == 1.0
        @test StaticTaylorSeries.evaluate(st1, 2.0) == 41.0
        @test StaticTaylorSeries.evaluate(st1) == 1.00
        @test StaticTaylorSeries.evaluate(st2, 2.0)[1] ==
              StaticTaylorSeries.evaluate(st2, 2.0)[2] ==
              41.0
        @test StaticTaylorSeries.evaluate(st2)[1] ==
              StaticTaylorSeries.evaluate(st2)[2] ==
              1.0

        # check that STaylor1 and Taylor yeild same result
        t1 = STaylor1([1.1, 2.1, 3.1])
        t2 = Taylor1([1.1, 2.1, 3.1])
        for f in (exp, abs, log, sin, cos, sinh, cosh, mod2pi, sqrt)
            @test test_vs_Taylor1(f(t1), f(t2))
        end

        t1_mod = mod(t1, 2.0)
        t2_mod = mod(t2, 2.0)
        @test isapprox(t1_mod[0], t2_mod[0], atol = 1E-10)
        @test isapprox(t1_mod[1], t2_mod[1], atol = 1E-10)
        @test isapprox(t1_mod[2], t2_mod[2], atol = 1E-10)

        t1_rem = rem(t1, 2.0)
        t2_rem = rem(t2, 2.0)
        @test isapprox(t1_rem[0], t2_rem[0], atol = 1E-10)
        @test isapprox(t1_rem[1], t2_rem[1], atol = 1E-10)
        @test isapprox(t1_rem[2], t2_rem[2], atol = 1E-10)

        t1a = STaylor1([2.1, 2.1, 3.1])
        t2a = Taylor1([2.1, 2.1, 3.1])

        for test_tup in (
            (/, t1, t1a, t2, t2a),
            (*, t1, t1a, t2, t2a),
            (/, t1, 1.3, t2, 1.3),
            (*, t1, 1.3, t2, 1.3),
            (+, t1, 1.3, t2, 1.3),
            (-, t1, 1.3, t2, 1.3),
            (*, 1.3, t1, 1.3, t2),
            (+, 1.3, t1, 1.3, t2),
            (-, 1.3, t1, 1.3, t2),
            (*, 1.3, t1, 1.3, t2),
            (^, t1, 0, t2, 0),
            (^, t1, 1, t2, 1),
            (^, t1, 2, t2, 2),
            (^, t1, 3, t2, 3),
            (^, t1, 4, t2, 4),
            (/, 1.3, t1, 1.3, t2),
            (^, t1, -1, t2, -1),
            (^, t1, -2, t2, -2),
            (^, t1, -3, t2, -3),
            (^, t1, 0.6, t2, 0.6),
            (^, t1, 1 / 2, t2, 1 / 2),
        )
            temp1 = test_tup[1](test_tup[2], test_tup[3])
            temp2 = test_tup[1](test_tup[4], test_tup[5])
            @test isapprox(temp1[0], temp2[0], atol = 1E-10)
            @test isapprox(temp1[1], temp2[1], atol = 1E-10)
            @test isapprox(temp1[2], temp2[2], atol = 1E-10)
        end

        @test isapprox(
            StaticTaylorSeries.square(t1)[0],
            (t2^2)[0],
            atol = 1E-10,
        )
        @test isapprox(
            StaticTaylorSeries.square(t1)[1],
            (t2^2)[1],
            atol = 1E-10,
        )
        @test isapprox(
            StaticTaylorSeries.square(t1)[2],
            (t2^2)[2],
            atol = 1E-10,
        )

        a = STaylor1([0.0, 1.2, 2.3, 4.5, 0.0])
        @test findfirst(a) == 1
        @test findlast(a) == 3

        eval_staylor = StaticTaylorSeries.evaluate(a, Interval(1.0, 2.0))
        @test isapprox(eval_staylor.lo, 7.99999, atol = 1E-4)
        @test isapprox(eval_staylor.hi, 47.599999999999994, atol = 1E-4)

        a = STaylor1([5.0, 1.2, 2.3, 4.5, 0.0])
        @test isapprox(deg2rad(a)[0], 0.087266, atol = 1E-5)
        @test isapprox(deg2rad(a)[2], 0.040142, atol = 1E-5)
        @test isapprox(rad2deg(a)[0], 286.4788975, atol = 1E-5)
        @test isapprox(rad2deg(a)[2], 131.7802928, atol = 1E-5)
        @test real(a) == STaylor1([5.0, 1.2, 2.3, 4.5, 0.0])
        @test imag(a) == STaylor1([0.0, 0.0, 0.0, 0.0, 0.0])
        @test adjoint(a) == STaylor1([5.0, 1.2, 2.3, 4.5, 0.0])
        @test conj(a) == STaylor1([5.0, 1.2, 2.3, 4.5, 0.0])
        @test a == abs(a)
        @test a == abs(-a)

        @test convert(
            STaylor1{3,Float64},
            STaylor1{3,Float64}((1.1, 2.2, 3.3)),
        ) == STaylor1{3,Float64}((1.1, 2.2, 3.3))
        @test convert(STaylor1{3,Float64}, 1) == STaylor1(1.0, Val(3))
        @test convert(STaylor1{3,Float64}, 1.2) == STaylor1(1.2, Val(3))

        #ta(a) = STaylor1(1.0, Val(15))
        @test promote(1.0, STaylor1(1.0, Val(15)))[1] == STaylor1(1.0, Val(16))
        @test promote(0, STaylor1(1.0, Val(15)))[1] == STaylor1(0.0, Val(16))
        @test eltype(promote(STaylor1(1, Val(15)), 2)[2]) == Int
        @test eltype(promote(STaylor1(1.0, Val(15)), 1.1)[2]) == Float64
        @test eltype(promote(0, STaylor1(1.0, Val(15)))[1]) == Float64
    end
end


@testset "Lohner's Method Testset" begin

    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]^2
        nothing
    end
    tspan = (0.0, tend)
    pL = [-1.0]
    pU = [1.0]
    prob = DBB.ODERelaxProb(f!, tspan, x0, pL, pU)

    integrator = DiscretizeRelax(
        prob,
        DR.LohnerContractor{7}(),
        h = 1 / ticks,
        repeat_limit = 1,
        skip_step2 = false,
        step_limit = steps,
        relax = false,
    )

    ratio = rand(1)
    pstar = pL .* ratio .+ pU .* (1.0 .- ratio)
    DBB.setall!(integrator, DBB.ParameterValue(), [0.0])
    DBB.relax!(integrator)

    lo_vec = getfield.(getindex.(integrator.storage[:], 1), :lo)
    hi_vec = getfield.(getindex.(integrator.storage[:], 1), :hi)

    @test isapprox(lo_vec[7], 5.802638399798364, atol = 1E-5)
    @test isapprox(hi_vec[7], 5.885672409877501, atol = 1E-5)

    integrator = DiscretizeRelax(
        prob,
        DR.LohnerContractor{7}(),
        repeat_limit = 1,
        skip_step2 = false,
        step_limit = steps,
        relax = false,
    )

    ratio = rand(1)
    pstar = pL .* ratio .+ pU .* (1.0 .- ratio)
    DBB.setall!(integrator, DBB.ParameterValue(), [0.0])
    DBB.relax!(integrator)

    lo_vec = getfield.(getindex.(integrator.storage[:], 1), :lo)
    hi_vec = getfield.(getindex.(integrator.storage[:], 1), :hi)

    @test isapprox(lo_vec[6], 3.3824180351195783, atol = 1E-5)
    @test isapprox(hi_vec[6], 3.5704139370767916, atol = 1E-5)
end

@testset "Hermite-Obreshkoff Testset" begin

    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]^2
        nothing
    end
    tspan = (0.0, tend)
    pL = [-1.0]
    pU = [1.0]
    prob = DBB.ODERelaxProb(f!, tspan, x0, pL, pU)

    integrator = DiscretizeRelax(prob, DR.HermiteObreschkoff(3, 3),
                                 h = 1/ticks, repeat_limit = 1,
                                 skip_step2 = false,
                                 step_limit = steps,
                                 relax = false)

    ratio = rand(1)
    pstar = pL .* ratio .+ pU .* (1.0 .- ratio)
    DBB.setall!(integrator, DBB.ParameterValue(), [0.0])
    DBB.relax!(integrator)

    lo_vec = getfield.(getindex.(integrator.storage[:], 1), :lo)
    hi_vec = getfield.(getindex.(integrator.storage[:], 1), :hi)

    @test isapprox(lo_vec[6], 6.150098100006023, atol = 1E-5)
    @test isapprox(hi_vec[6], 6.263901898905692, atol = 1E-5)
end

@testset "Wilhelm 2019 Integrator Testset" begin
end

@testset "Discretize and Relax - Access Functions" begin

    use_relax = false
    lohners_type = 1
    prob_num = 1
    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [1.2; 1.1]
    function f!(dx, x, p, t)
        dx[1] = p[1] * x[1] * (one(typeof(p[1])) - x[2])
        dx[2] = p[1] * x[2] * (x[1] - one(typeof(p[1])))
        nothing
    end
    tspan = (0.0, tend)
    pL = [2.95]
    pU = [3.05]

    prob = DBB.ODERelaxProb(f!, tspan, x0, pL, pU)

    integrator = DiscretizeRelax(
        prob,
        DynamicBoundspODEsDiscrete.LohnerContractor{7}(),
        h = 1 / ticks,
        repeat_limit = 1,
        skip_step2 = false,
        step_limit = steps,
        relax = use_relax,
    )

    @test DBB.supports(integrator, DBB.IntegratorName())
    @test !DBB.supports(integrator, DBB.Gradient())
    @test DBB.supports(integrator, DBB.Subgradient())
    @test DBB.supports(integrator, DBB.Bound())
    @test DBB.supports(integrator, DBB.Relaxation())
    @test DBB.supports(integrator, DBB.IsNumeric())
    @test DBB.supports(integrator, DBB.IsSolutionSet())
    @test DBB.supports(integrator, DBB.TerminationStatus())
    @test DBB.supports(integrator, DBB.Value())
    @test DBB.supports(integrator, DBB.ParameterValue())

    @test DBB.get(integrator, DBB.IntegratorName()) == "Discretize & Relax Integrator"
    @test !DBB.get(integrator, DBB.IsNumeric())
    @test DBB.get(integrator, DBB.IsSolutionSet())
    @test DBB.get(integrator, DBB.TerminationStatus()) == RELAXATION_NOT_CALLED

    DBB.set!(integrator, DBB.SupportSet(Float64[i for i in range(0.0, tend, length = 200)]))

    ratio = rand(1)
    pstar = pL .* ratio .+ pU .* (1.0 .- ratio)
    DBB.setall!(integrator, DBB.ParameterValue(), [0.0])
    DBB.relax!(integrator)

    DBB.setall!(integrator, DBB.ParameterBound{Lower}(), [2.99])
    DBB.setall!(integrator, DBB.ParameterBound{Upper}(), [3.01])

    support_set = DBB.get(integrator, DBB.SupportSet())
    @test support_set.s[3] == 0.02

    out = Matrix{Float64}[]
    for i in support_set.s
        push!(out, zeros(10,1))
    end
    DBB.getall!(out, integrator, DBB.Subgradient{Lower}())
    @test out[1][10,1] == 0.0

    DBB.getall!(out, integrator, DBB.Subgradient{Upper}())
    @test out[1][10,1] == 0.0

    out = copy(support_set.s)
    DBB.getall!(out, integrator, DBB.Bound{Lower}())
    @test isapprox(out[10,1], 1.1507186500504751, atol=1E-8)

    DBB.getall!(out, integrator, DBB.Bound{Upper}())
    @test isapprox(out[10,1], 1.1534467709985823, atol=1E-8)

    #DBB.getall!(out, integrator, DBB.Relaxation{Lower}())
    #@test isapprox(out[10,1], 1.1534467709985823, atol=1E-8)

    #DBB.getall(out, integrator, DBB.Relaxation{Upper}())
    #@test isapprox(out[10,1], 1.1534467709985823, atol=1E-8)

    #=
    out = DBB.getall(integrator, DBB.Subgradient{Lower}())
    @test out[1][10,1] == 0.0

    out = DBB.getall(integrator, DBB.Subgradient{Upper}())
    @test out[1][10,1] == 0.0

    out = DBB.getall(integrator, DBB.Bound{Lower}())
    @test out[10,1] == 0.0

    out = DBB.getall(integrator, DBB.Bound{Upper}())
    @test out[10,1] == 0.0

    out = DBB.getall(integrator, DBB.Relaxation{Lower}())
    @test out[10,1] == 0.0

    out = DBB.getall(integrator, DBB.Relaxation{Upper}())
    @test out[10,1] == 0.0
    =#
end
