# Test that the transition density
# in a non-linear, non-homogenous, non-constant diffusivity model
# estimated by forward simulation
# agrees with the density obtained from the linearisation
# reweighted with importance weights using guided proposals.
using Test
using Bridge, BridgeSDEInference, StaticArrays, Distributions
using Statistics, Random, LinearAlgebra

const BSI = BridgeSDEInference
using BridgeSDEInference: ℝ

const 𝕏 = SVector
using GaussianDistributions

struct TargetSDE <: Bridge.ContinuousTimeProcess{Float64}
end
struct LinearSDE{T}  <: Bridge.ContinuousTimeProcess{Float64}
    σ::T
end
Bridge.b(s, x, P::TargetSDE) = Bridge.B(s, P)*x + Bridge.β(s, P) + 𝕏(0.0, 0.5sin(x[2]))
Bridge.b(s, x, P::LinearSDE) = Bridge.B(s, P)*x + Bridge.β(s, P)
Bridge.B(s, P::Union{LinearSDE,TargetSDE}) = @SMatrix [-0.1 0.1; 0.0 -0.1]
Bridge.β(s, P::Union{LinearSDE,TargetSDE}) = 𝕏(0.0, 0.5sin(s/4))

Bridge.σ(s, x, P::TargetSDE) = @SMatrix [.0 0.0; 0.0 2.0]
Bridge.σ(s, x, P::LinearSDE) = P.σ
Bridge.σ(s, P::LinearSDE) = P.σ
Bridge.a(s, P::LinearSDE) = P.σ*P.σ'

Bridge.constdiff(::TargetSDE) = true
Bridge.constdiff(::LinearSDE) = true

binind(r, x) = searchsortedfirst(r, x) - 1

simid = 1 # 1, 2

#function test_measchange()
    Random.seed!(1)
    resfactor = 1.0
    timescale = [1.0, 10.0][simid]
    T = round(timescale*4*pi, digits=2)
    P = TargetSDE()
    v = 𝕏(1.0, pi/2)

    x0 = 𝕏(0.0, -pi/2)

    t = 0:0.01*resfactor:T
    t = Bridge.tofs.(t, 0, T)
    W = Bridge.samplepath(t, 𝕏(0.0, 0.0))

    Wnr =  Wiener{typeof(𝕏(1.0, 1.0))}()

    Σ = SMatrix{2,2}(1e-6, 0.0, 0.0, 1.)
    L = SMatrix{2,2}(1.0, 0.0, 1.0, 0.)
    Noise = Gaussian(𝕏(0.0, 0.0), Σ)

    sample!(W, Wnr)
    X = solve(Euler(), x0, W, P)
    v1 = X.yy[end]
    X.yy[end] = zero(v1)
    solve!(Euler(), X, x0, W, P)
    @test v1 ≈ X.yy[end]



    K = 70


    k = 1
    N = 100000

    # Forward simulation

    vs = typeof(x0)[]
    for i in 1:N
        sample!(W, Wnr)
        solve!(Euler(), X, x0, W, P)
        push!(vs, X.yy[end])
    end



    counts = zeros(K+2)
    obs = [L * v + rand(Noise) for v in vs]

    R = maximum(abs.(first.(obs)))
    vrange = range(-R,R, length=K+1)
    vints = [(vrange[i], vrange[i+1]) for i in 1:K]


    [counts[binind(vrange, o[1])+1] += 1 for o in obs]
    counts /= length(vs)

    wcounts = zeros(K)


    VProp = Gaussian(mean(obs), cov(obs))

    fpt = fill(NaN, 1)

    P̃ = LinearSDE(Bridge.σ(T, 𝕏(0.0, 0.0), P)) # use a law with large variance
    Pᵒ = BridgeSDEInference.GuidPropBridge(eltype(x0), t, P, P̃, L, v, Σ; changePt = BSI.SimpleChangePt(length(t)÷2))

    # Guided proposals

    for i in 1:N
        v = rand(VProp)
        while !(binind(vrange, v[1]) in 1:K)
            v = rand(VProp)
        end
        # other possibility: change proposal each step
    #    P̃ = LinearSDE(Bridge.σ(T, v, P))
        Pᵒ = BridgeSDEInference.GuidPropBridge(eltype(x0), t, P, P̃, L, v, Σ;  changePt = BSI.SimpleChangePt(length(t)÷2))

        sample!(W, Wnr)
        solve!(Euler(), X, x0, W, Pᵒ)
        ll = BSI.pathLogLikhd(BridgeSDEInference.PartObs(), [X], [Pᵒ], 1:1, fpt, skipFPT=true)
        ll += BSI.lobslikelihood(Pᵒ, x0)
        ll -= logpdf(VProp, v)
        wcounts[binind(vrange, v[1])] += exp(ll)/N
    end
    bias = wcounts - counts[2:end-1]

    @testset "Statistical correctness of guided proposals" begin
    #    @test norm(bias) < 0.1
    end
    vrange, wcounts, counts
#end

simname = "L11T$(round(Int, T)).png"
