using Revise

using DynamicBoundspODEsPILMS, LinearAlgebra, IntervalArithmetic, StaticArrays, TaylorSeries,
      TaylorIntegration, ForwardDiff, McCormick, BenchmarkTools, DocStringExtensions

using DiffResults: JacobianResult, MutableDiffResult
using ForwardDiff: Partials, JacobianConfig, vector_mode_dual_eval, value, vector_mode_jacobian!

import Base.copyto!

function f!(dx,x,p,t)
    dx[1] = x[1]^2 + p[2]
    dx[2] = x[2] + p[1]^2
    nothing
end
np = 2
nx = 2
k = 3
x = [0.1; 1.0]
p = [0.2; 0.1]
jtf! = DynamicBoundspODEsPILMS.JacTaylorFunctor!(f!, nx, np, k, Interval{Float64}(0.0), 0.0)


xIntv = Interval{Float64}.(x)
pIntv = Interval{Float64}.(p)
yIntv = [xIntv; pIntv]
DynamicBoundspODEsPILMS.jacobian_taylor_coeffs!(jtf!, yIntv)

jac = JacobianResult(jtf!.out, yIntv).derivs[1]
tjac = zeros(Interval{Float64}, 4, 8)
Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
DynamicBoundspODEsPILMS.extract_JxJp!(Jx, Jp, jtf!.result, tjac, nx, np, k)

itf! = DynamicBoundspODEsPILMS.TaylorFunctor!(f!, nx, np, k, zero(Interval{Float64}), zero(Float64))
outIntv = zeros(Interval{Float64},8)
itf!(outIntv, yIntv)

y = [x; p]
rtf!  = DynamicBoundspODEsPILMS.TaylorFunctor!(f!, nx, np, k, zero(Float64), zero(Float64))
out = zeros(8)
rtf!(out, y)

coeff_out = zeros(Interval{Float64},2,4)
DynamicBoundspODEsPILMS.coeff_to_matrix!(coeff_out, jtf!.out, nx, k)

hⱼ = 0.001
hmin = 0.00001
function euf!(out, x, p, t)
    out[1,1] = -x[1]
    nothing
end
eufY = [Interval{Float64}(0.5,1.5); Interval(0.0)]
itf_exist_unique! = DynamicBoundspODEsPILMS.TaylorFunctor!(euf!, 1, 1, k, zero(Interval{Float64}), zero(Float64))
jtf_exist_unique! = DynamicBoundspODEsPILMS.JacTaylorFunctor!(euf!, 1, 1, k, Interval{Float64}(0.0), 0.0)
DynamicBoundspODEsPILMS.jacobian_taylor_coeffs!(jtf_exist_unique!, eufY)
coeff_out = zeros(Interval{Float64},1,k)
DynamicBoundspODEsPILMS.coeff_to_matrix!(coeff_out, jtf!.out, 1, k)
Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},1,1) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},1,1) for i in 1:4]
tjac = zeros(Interval{Float64}, 2, 4)
outIntv_exist_unique! = zeros(Interval{Float64},4)
itf_exist_unique!(outIntv_exist_unique!, eufY)
coeff_out_exist_unique! = zeros(Interval{Float64},1,k+1)
DynamicBoundspODEsPILMS.coeff_to_matrix!(coeff_out_exist_unique!, outIntv_exist_unique!, 1, k)
DynamicBoundspODEsPILMS.extract_JxJp!(Jx, Jp, jtf_exist_unique!.result, tjac, 1, 1, k)
unique_result = DynamicBoundspODEsPILMS.UniquenessResult(1,1)
DynamicBoundspODEsPILMS.existence_uniqueness!(unique_result, itf_exist_unique!,
                                              eufY, hⱼ, hmin, coeff_out_exist_unique!, Jx, Jp)

bool1 = unique_result.step == 0.001
bool2 = unique_result.confirmed
bool3a = isapprox(unique_result.Ỹⱼ[1].lo, -1.50001E-6, atol=1E-10)
bool3b = isapprox(unique_result.Ỹⱼ[1].hi, 0.00150001, atol=1E-6)
bool4 = isapprox(unique_result.f̃k[1].lo, -7.50001E-16, atol = 1E-19)
bool5 = isapprox(unique_result.f̃k[1].hi, 7.50001E-13, atol = 1E-16)

#=
plohners = DynamicBoundspODEsPILMS.parametric_lohners!(itf!, rtf!, dtf, hⱼ, Ycat, Ycat,
                                                       A, yjcat, Δⱼ)

jetcoeffs!(zqwa, zqwb, zqwc, zqwd, zqwe, zqwr, p)
y = Interval{Float64}.([x; p])
out = g.out
cfg = ForwardDiff.JacobianConfig(nothing, out, y)

# extact is good... actual jacobians look odd...

hⱼ = 0.001
hmin = 0.00001
Yⱼ = [Interval(0.1, 5.1); Interval(0.1, 8.9)]
P = [Interval(0.1, 5.1); Interval(0.1, 8.9)]
routIntv = copy(Yⱼ)

out = zeros(Interval{Float64},8)
coeff_out = zeros(Interval{Float64},2,4)
Ycat = [Yⱼ; P]
itf!(out, Ycat)

@btime itf!($out, $Ycat)
DynamicBoundspODEsPILMS.coeff_to_matrix!(coeff_out, out, nx, k)
@btime DynamicBoundspODEsPILMS.coeff_to_matrix!($coeff_out, $out, $nx, $k)

DynamicBoundspODEsPILoutIntvMS.existence_uniqueness(itf!, Ycat, hⱼ, hmin, coeff_out, Jx)
#@btime improvement_condition($Yⱼ, $Yⱼ, $nx)
@btime DynamicBoundspODEsPILMS.existence_uniqueness($itf!, $Ycat, $hⱼ, $hmin, $coeff_out, $Jx)
#tv, xv = validated_integration(f!, Interval{Float64}.([3.0, 3.0]), 0.0, 0.3, 4, 1.0e-20, maxsteps=100 )
Q = [Yⱼ; P]
#@btime jacobianfunctor($outIntv, $yInterval)

d = g
zqwa = d.g!
zqwb = d.t
zqwc = d.xtaylor
zqwd = d.xout
zqwe = d.xaux
zqwr = d.taux
#@btime jetcoeffs!($zqwa, $zqwb, $zqwc, $zqwd, $zqwe, $zqwr, $s, $p)
#@code_warntype jetcoeffs!(zqwa, zqwb, zqwc, zqwd, zqwe, zqwr, p)


Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jxsto = zeros(Interval{Float64},2,2)
Jpsto = zeros(Interval{Float64},2,2)

Yⱼ = [Interval{Float64}(-10.0, 20.0); Interval{Float64}(-10.0, 20.0)]
P = [Interval{Float64}(2.0, 3.0); Interval{Float64}(2.0, 3.0)]
Ycat = [Yⱼ; P]
yⱼ = mid.(Yⱼ)
Δⱼ = Yⱼ - yⱼ
At = zeros(2,2) + I
#Aⱼ =  DynamicBoundspODEsPILMS.QRDenseStorage(nx)
#Aⱼ₊₁ =  DynamicBoundspODEsPILMS.QRDenseStorage(nx)
A =  DynamicBoundspODEsPILMS.QRStack(nx, 2)
dtf = g
hⱼ = 0.001
yjcat = vcat(yⱼ,p)
# TODO: Remember rP is computed outside iteration and stored to JacTaylorFunctor
plohners = DynamicBoundspODEsPILMS.parametric_lohners!(itf!, rtf!, dtf, hⱼ, Ycat, Ycat,
                                                       A, yjcat, Δⱼ)

@btime DynamicBoundspODEsPILMS.parametric_lohners!($itf!, $rtf!, $dtf, $hⱼ, $Ycat, $Ycat,
                                                    $A, $yjcat, $Δⱼ)
=#
