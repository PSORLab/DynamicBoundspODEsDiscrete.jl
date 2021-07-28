#!/usr/bin/env julia
using Test, DynamicBoundspODEsDiscrete, McCormick, DynamicBoundsBase,
      DataStructures
using DynamicBoundspODEsDiscrete.StaticTaylorSeries
using DiffResults: JacobianResult

const DBB = DynamicBoundsBase
const DR = DynamicBoundspODEsDiscrete

@testset "Discretize and Relax" begin

    struct unit_test_name <: DR.AbstractStateContractorName
    end

    @test_throws ErrorException DR.state_contractor_k(unit_test_name())
    @test_throws ErrorException DR.state_contractor_γ(unit_test_name())
    @test_throws ErrorException DR.state_contractor_steps(unit_test_name())


    # test improvement condition for existence & uniqueness
    Yold = [Interval(1.0, 3.0); Interval(2.0, 4.0); Interval(1.0, 3.0)]
    Ynew = [Interval(1.5, 2.0); Interval(3.0, 3.5); Interval(0.5, 3.5)]
    nx_ic = 2
    @test DR.improvement_condition(Yold, Ynew, nx_ic)

    function J!(out, x, p, t)
        out[1, 1] = x
        nothing
    end

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

        @test STaylor1 <: DR.StaticTaylorSeries.AbstractSeries
        @test STaylor1{1,Float64} <: DR.StaticTaylorSeries.AbstractSeries{Float64}
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

        @test_throws ArgumentError STaylor1([1.1, 2.1])/STaylor1([0.0, 2.1])

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

        @test view(typeof(STaylor1([1.1, 2.1]))[STaylor1([1.1, 2.1]) STaylor1([1.1, 2.1]); STaylor1([1.1, 2.1]) STaylor1([1.1, 2.1])], :, 1)(0.0) == Float64[1.1; 1.1]
        @test view(typeof(STaylor1([1.1, 2.1]))[STaylor1([1.1, 2.1]) STaylor1([1.1, 2.1]); STaylor1([1.1, 2.1]) STaylor1([1.1, 2.1])], :, 1)() == Float64[1.1; 1.1]

        # check that STaylor1 and Taylor yeild same result
        t1 = STaylor1([1.1, 2.1, 3.1])
        t2 = Taylor1([1.1, 2.1, 3.1])
        for f in (exp, abs, log, sin, cos, sinh, cosh, mod2pi, sqrt, abs2,
                  deg2rad, rad2deg)
            @test test_vs_Taylor1(f(t1), f(t2))
        end

        @test DR.StaticTaylorSeries.get_order(t1) == 2
        @test axes(t1) isa Tuple{}
        @test iterate(t1)[1] == 1.10
        @test iterate(t1)[2] == 1
        @test eachindex(t1) == 0:2
        @test t1[:] == (1.10, 2.10, 3.10)
        @test t1[1:2] == (2.10, 3.10)
        @test t1[0:2:2] == (1.10, 3.10)
        @test rem(t1, 2) == t1
        @test mod(t1, 2) == t1

        @test STaylor1([1.1, 2.1, 3.1], Val(3),  Val(5)) == STaylor1([1.1, 2.1, 3.1, 0.0, 0.0, 0.0])

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
            #(^, t1, -1, t2, -1),
            #(^, t1, -2, t2, -2),
            #(^, t1, -3, t2, -3),
            (^, t1, 0.6, t2, 0.6),
            (^, t1, 1 / 2, t2, 1 / 2),
        )
            temp1 = test_tup[1](test_tup[2], test_tup[3])
            temp2 = test_tup[1](test_tup[4], test_tup[5])
            check1 = isapprox(temp1[0], temp2[0], atol = 1E-10)
            check2 = isapprox(temp1[1], temp2[1], atol = 1E-10)
            check3 = isapprox(temp1[2], temp2[2], atol = 1E-10)
            @test check1
            @test check2
            @test check3
            if !check1 || !check2 || !check3
                println("$test_tup, $temp1, $temp2")
            end
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
        @test_broken promote(1.0, STaylor1(1.0, Val(15)))[1] == STaylor1(1.0, Val(16))
        @test_broken promote(0, STaylor1(1.0, Val(15)))[1] == STaylor1(0.0, Val(16))
        @test_broken eltype(promote(STaylor1(1, Val(15)), 2)[2]) == Int
        @test_broken eltype(promote(STaylor1(1.0, Val(15)), 1.1)[2]) == Float64
        @test_broken eltype(promote(0, STaylor1(1.0, Val(15)))[1]) == Float64

        @test_broken promote_rule(typeof(STaylor1([1.1, 2.1])), typeof(STaylor1([1.1, 2.1]))) == STaylor1{2,Float64}
        @test_broken promote_rule(typeof(STaylor1([1.1, 2.1])), typeof(STaylor1([1, 2]))) == STaylor1{2,Float64}
        @test_broken promote_rule(typeof(STaylor1([1.1, 2.1])), typeof([1.1, 2.1])) == STaylor1{2,Float64}
        @test_broken promote_rule(typeof(STaylor1([1.1, 2.1])), typeof([1, 2])) == STaylor1{2,Float64}
        @test_broken promote_rule(typeof(STaylor1([1.1, 2.1])), typeof(1.1)) == STaylor1{2,Float64}
        @test_broken promote_rule(typeof(STaylor1([1.1, 2.1])), typeof(1)) == STaylor1{2,Float64} #TODO: FAILING

        @test convert(STaylor1{2,Float64}, [1; 2]) == STaylor1(Float64[1, 2])
        @test convert(STaylor1{2,Float64}, [1.1; 2.1]) == STaylor1([1.1, 2.1])
        @test convert(STaylor1{2,Rational{Int64}}, STaylor1(BigFloat[0.5, 0.75])) == STaylor1([0.5, 0.75])

        @test isapprox(StaticTaylorSeries.normalize_taylor(STaylor1([1.1, 2.1]), Interval(1.0, 2.0))[0], 13.7, atol = 1E-8)
        @test isapprox(StaticTaylorSeries._normalize(STaylor1([1.1, 2.1]), Interval(1.0, 2.0), Val(true))[0], 13.7, atol = 1E-8)
        @test isapprox(StaticTaylorSeries._normalize(STaylor1([1.1, 2.1]), Interval(1.0, 2.0), Val(false))[0], 13.7, atol = 1E-8)

        @test_nowarn Base.show(stdout, a)
        @test StaticTaylorSeries.coeffstring(a, 5) == "0.0t^4"
        @test StaticTaylorSeries.coeffstring(a, 2) == "1.2t"

        @test_throws ArgumentError abs(STaylor1([0.0, 2.1]))
    end
end

@testset "Lohner's Method Interval Testset" begin

    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]
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

    @test isapprox(lo_vec[7], 8.417645335842485, atol = 1E-5)
    @test isapprox(hi_vec[7], 8.534116268673994, atol = 1E-5)
end

@testset "Lohner's Method Adaptive Interval Testset" begin

    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1]  - x[1]*x[1]
        nothing
    end
    tspan = (0.0, tend)
    pL = [-1.0]
    pU = [1.0]
    prob = DBB.ODERelaxProb(f!, tspan, x0, pL, pU)

    integrator = DiscretizeRelax(
        prob,
        DR.LohnerContractor(7),
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

    @test isapprox(lo_vec[6], 5.884811279034315, atol = 1E-5)
    @test isapprox(hi_vec[6], 5.965419612738188, atol = 1E-5)

    support_set = DBB.get(integrator, DBB.SupportSet())
    outvec = zeros(length(support_set.s))

    DBB.getall!(outvec, integrator, DBB.Bound{Lower}())
    @test_broken isapprox(outvec[10], 0.9746606911231538, atol=1E-8)

    DBB.getall!(outvec, integrator, DBB.Bound{Upper}())
    @test_broken isapprox(outvec[10], 1.7009858838207106, atol=1E-8)

    DBB.getall!(outvec, integrator, DBB.Relaxation{Lower}())
    @test_broken isapprox(outvec[10], 0.9746606911231538, atol=1E-8)

    DBB.getall!(outvec, integrator, DBB.Relaxation{Upper}())
    @test_broken isapprox(outvec[10], 1.7009858838207106, atol=1E-8)
end

@testset "Lohner's Method MC Testset" begin

    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]
        nothing
    end
    tspan = (0.0, tend)
    pL = [-0.3]
    pU = [0.3]
    prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)

    integrator = DiscretizeRelax(
        prob,
        DynamicBoundspODEsDiscrete.LohnerContractor{7}(),
        h = 1 / ticks,
        repeat_limit = 1,
        skip_step2 = false,
        step_limit = steps,
        relax = true,
    )

    ratio = rand(1)
    pstar = pL .* ratio .+ pU .* (1.0 .- ratio)
    DynamicBoundsBase.setall!(integrator, DynamicBoundsBase.ParameterValue(), [0.0])
    DynamicBoundsBase.relax!(integrator)

    lo_vec = getfield.(getfield.(getindex.(integrator.storage[:], 1), :Intv), :lo)
    hi_vec = getfield.(getfield.(getindex.(integrator.storage[:], 1), :Intv), :hi)

    @test isapprox(lo_vec[6], 8.546433647856642, atol = 1E-5)
    @test isapprox(hi_vec[6], 8.575695993156213, atol = 1E-5)

    support_set = DBB.get(integrator, DBB.SupportSet())
    out = Matrix{Float64}[]
    for i = 1:1
        push!(out, zeros(1,length(support_set.s)))
    end
    # DBB.getall!(out, integrator, DBB.Subgradient{Lower}())
    # @test isapprox(out[1][1,10], 0.08606881472877183, atol=1E-8)

    # DBB.getall!(out, integrator, DBB.Subgradient{Upper}())
    # @test isapprox(out[1][1,10], 0.08606881472877183, atol=1E-8)

    #out = zeros(1, length(support_set.s))
    #DBB.getall!(out, integrator, DBB.Bound{Lower}())
    #@test isapprox(out[1,10], 8.199560023022423, atol=1E-8)

    #DBB.getall!(out, integrator, DBB.Bound{Upper}())
    #@test isapprox(out[1,10], 8.251201311859687, atol=1E-8)

    #DBB.getall!(out, integrator, DBB.Relaxation{Lower}())
    #@test isapprox(out[1,10], 8.225380667441055, atol=1E-8) #

    #DBB.getall!(out, integrator, DBB.Relaxation{Upper}())
    #@test isapprox(out[1,10], 8.225380667441055, atol=1E-8) #

    #out = DBB.getall(integrator, DBB.Subgradient{Lower}())
    #@test isapprox(out[1][1,10], 0.08606881472877183, atol=1E-8)

    #out = DBB.getall(integrator, DBB.Subgradient{Upper}())
    #@test isapprox(out[1][1,10], 0.08606881472877183, atol=1E-8)

    #out = DBB.getall(integrator, DBB.Bound{Lower}())
    #@test isapprox(out[1,10], 8.199560023022423, atol=1E-8)

    #out = DBB.getall(integrator, DBB.Bound{Upper}())
    #@test isapprox(out[1,10], 8.251201311859687, atol=1E-8)

    #out = DBB.getall(integrator, DBB.Relaxation{Lower}())
    #@test isapprox(out[1,10], 8.225380667441055, atol=1E-8)

    #out = DBB.getall(integrator, DBB.Relaxation{Upper}())
    #@test isapprox(out[1,10], 8.225380667441055, atol=1E-8)

    #outvec = zeros(length(support_set.s))

    #DBB.getall!(outvec, integrator, DBB.Bound{Lower}())
    #@test isapprox(outvec[10], 8.199560023022423, atol=1E-8)

    #DBB.getall!(outvec, integrator, DBB.Bound{Upper}())
    #@test isapprox(outvec[10], 8.251201311859687, atol=1E-8)

    #DBB.getall!(outvec, integrator, DBB.Relaxation{Lower}())
    #@test isapprox(outvec[10], 8.225380667441055, atol=1E-8)

    #DBB.getall!(outvec, integrator, DBB.Relaxation{Upper}())
    #@test isapprox(outvec[10], 8.225380667441055, atol=1E-8)
end

@testset "Hermite-Obreshkoff Testset" begin

    ticks = 100.0
    steps = 100.0
    tend = steps / ticks

    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]
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

    @test isapprox(lo_vec[6], 8.510710620631412, atol = 1E-5)
    @test isapprox(hi_vec[6], 8.611419020357289, atol = 1E-5)
end

@testset "Wilhelm 2019 Integrator Testset" begin
    Y = Interval(0,5)
    X1 = Interval(1,2)
    X2 = Interval(-10,10)
    X3 = Interval(0,5)
    flag1 = DR.strict_x_in_y(X1,Y)
    flag2 = DR.strict_x_in_y(X2,Y)
    flag3 = DR.strict_x_in_y(X3,Y)
    @test flag1 == true
    @test flag2 == false
    @test flag3 == false

    A1 = Interval(0)
    A2 = Interval(0,3)
    A3 = Interval(-2,0)
    A4 = Interval(-3,2)
    ind1,B1,C1 = DR.extended_divide(A1)
    ind2,B2,C2 = DR.extended_divide(A2)
    ind3,B3,C3 = DR.extended_divide(A3)
    ind4,B4,C4 = DR.extended_divide(A4)
    @test ind1 == 0
    @test ind2 == 1
    @test ind3 == 2
    @test ind4 == 3
    @test B1 == Interval(-Inf,Inf)
    @test 0.33333 - 1E-4 <= B2.lo <= 0.33333 + 1E-4
    @test B2.hi == Inf
    @test B3 == Interval(-Inf,-0.5)
    @test B4.lo == -Inf
    @test -0.33333 - 1E-4 <= B4.hi <= -0.33333 + 1E-4
    @test C1 == Interval(-Inf,Inf)
    @test C2 == Interval(Inf,Inf)
    @test C3 == Interval(-Inf,-Inf)
    @test C4 == Interval(0.5,Inf)

    N =  Interval(-5,5)
    X = Interval(-5,5)
    Mii = Interval(-5,5)
    B = Interval(-5,5)
    rtol = 1E-4
    indx1,box11,box12 = DR.extended_process(N,X,Mii,B,rtol)
    Miib = Interval(0,5)
    S1b = Interval(1,5)
    S2b = Interval(1,5)
    Bb = Interval(1,5)
    indx2,box21,box22 = DR.extended_process(N,X,Mii,B,rtol)
    Miic = Interval(-5,0)
    S1c = Interval(1,5)
    S2c = Interval(1,5)
    Bc = Interval(1,5)
    indx3,box31,box32 = DR.extended_process(N,X,Mii,B,rtol)
    Miia = Interval(1,5)
    S1a = Interval(1,5)
    S2a = Interval(1,5)
    Ba = Interval(1,5)
    indx6,box61,box62 = DR.extended_process(N,X,Mii,B,rtol)
    Miid = Interval(0,0)
    S1d = Interval(1,5)
    S2d = Interval(1,5)
    Bd = Interval(1,5)
    indx8,box81,box82 = DR.extended_process(N,X,Mii,B,rtol)

    @test indx1 == 0
    @test box11 == Interval(-Inf,Inf)
    @test box12 == Interval(-5,5)

    @test indx2 == 0
    @test box21.hi > -Inf
    @test box22 == Interval(-5,5)

    @test indx3 == 0
    @test box31.lo < Inf
    @test box32 == Interval(-5,5)

    @test indx6 == 0
    @test box62.lo == -5.0
    @test box61.hi == Inf

    @test indx8 == 0
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
    @test DBB.supports(integrator, DBB.SupportSet())

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
    #@test support_set.s[3] == 0.02

    #=
    out = Matrix{Float64}[]
    for i in 1:1
        push!(out, zeros(1,length(support_set.s)))
    end
    DBB.getall!(out, integrator, DBB.Subgradient{Lower}())
    @test out[1][1,10] == 0.0

    DBB.getall!(out, integrator, DBB.Subgradient{Upper}())
    @test out[1][1,10] == 0.0

    out = zeros(1,length(support_set.s))
    DBB.getall!(out, integrator, DBB.Bound{Lower}())
    @test isapprox(out[1,10], 1.1507186500504751, atol=1E-8)

    DBB.getall!(out, integrator, DBB.Bound{Upper}())
    @test isapprox(out[1,10], 1.1534467709985823, atol=1E-8)

    DBB.getall!(out, integrator, DBB.Relaxation{Lower}())
    @test isapprox(out[1,10], 1.1507186500504751, atol=1E-8)

    DBB.getall!(out, integrator, DBB.Relaxation{Upper}())
    @test isapprox(out[1,10], 1.1534467709985823, atol=1E-8)

    out = DBB.getall(integrator, DBB.Subgradient{Lower}())
    @test out[1][1,10] == 0.0

    out = DBB.getall(integrator, DBB.Subgradient{Upper}())
    @test out[1][1,10] == 0.0

    out = DBB.getall(integrator, DBB.Bound{Lower}())
    @test isapprox(out[1,10], 1.1507186500504751, atol=1E-8)

    out = DBB.getall(integrator, DBB.Bound{Upper}())
    @test isapprox(out[1,10], 1.1534467709985823, atol=1E-8)

    out = DBB.getall(integrator, DBB.Relaxation{Lower}())
    @test isapprox(out[1,10], 1.1507186500504751, atol=1E-8)

    out = DBB.getall(integrator, DBB.Relaxation{Upper}())
    @test isapprox(out[1,10], 1.1534467709985823, atol=1E-8)
    =#
end
