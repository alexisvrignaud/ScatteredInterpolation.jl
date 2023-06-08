using Distances
using GLMakie
using Random
using CUDA

n_points = 75*75
X = rand(0:25.0,n_points)
Y = rand(0:25.0,n_points)

window_size = 10

samples_pos = [[x, y] for (x,y) in zip(X,Y)]
samples_pos = mapreduce(permutedims, vcat, samples_pos)' #Transforms to a Matrix 2xN

@time pairwise(Euclidean(), samples_pos)

println("end")

@doc "
    Multiquadratic(ɛ = 1)

Define a Multiquadratic Radial Basis Function

```math
ϕ(r) = \\sqrt{1 + (ɛr)^2}
```
" 
rbfMultiquadratic(r) = (sqrt(1 + (rbf.ɛ*r)^2))

struct RBFInterpolant{T1, T2, F, M} <: RadialBasisInterpolant where {T1 <: AbstractArray, T2 <: AbstractMatrix{<:Real}}

    w::T1
    points::T2
    rbf::F
    metric::M
end

function interpolate(points::AbstractArray{<:Real,2},
                     samples::AbstractArray{<:Number,N};
                     returnRBFmatrix::Bool = false,
                     smooth::Union{S, AbstractVector{S}} = false) where {N} where {S<:Number}

    #hinder smooth from being set to true and interpreted as the value 1 
    @assert smooth != true "set the smoothing value as a number or vector of numbers"

    # Compute pairwise distances, apply the Radial Basis Function
    # and optional smoothing (ridge regression)
    println("pairwise")
    println("metric: ", metric)
    A = pairwise(Euclidean(), points;dims=2)
    
    A = evaluateRBF!(A, rbfMultiquadratic, smooth)

    # Solve for the weights
    itp = solveForWeights(A, points, samples, rbfMultiquadratic, metric)

    # Create and return an interpolation object
    if returnRBFmatrix    # Return matrix A
        return itp, A
    else
        return itp
    end

end