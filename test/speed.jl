# using ScatteredInterpolation
include("../src/ScatteredInterpolation.jl")
using GLMakie
using Random

freq = 20
δ = 0.1
sizex = 20
sizey = 20

f(x,y) = (cos.(x./freq) + sin.(3*x./freq)).*sin.(y./freq)

#Create initial grid
x_mat = vec([ x for x=1:sizex, y = 1:sizey])
y_mat = vec([ y for x=1:sizex, y = 1:sizey])

#Displace grid by random amounts to create the "scattered effect"
δx_mat = x_mat .+ 2*δ.*(rand(length(x_mat)).-1)
δy_mat = y_mat .+ 2*δ.*(rand(length(y_mat)).-1)

#Remove some positions to simulate "itf_ok" real world situations
delete_percentage = 0.15
n_positions = length(δx_mat)
delete_number = Int(floor(delete_percentage*n_positions))
delete_pos = sort(randperm(n_positions)[1:delete_number])
deleteat!(δx_mat,delete_pos)
deleteat!(δy_mat,delete_pos)

#Compute function values at these positions
δz_mat = f.(δx_mat,δy_mat)
z_mat = f.(x_mat,y_mat)

#Interpolation
# println("Interpolator creation")
# samples = δz_mat
# samples_pos = [[x, y] for (x,y) in zip(δx_mat,δy_mat)]
# samples_pos = mapreduce(permutedims, vcat, samples_pos)' #Transforms to a Matrix 2xN
# itp = interpolate(Multiquadratic(), samples_pos, samples);

# println("Interpolation application")
# eval_points = [[x, y] for (x,y) in zip(x_mat,y_mat)]
# eval_points = mapreduce(permutedims, vcat, eval_points)'
# @time "Interpolation application" interpolated_values = evaluate(itp, eval_points)

# diff =  interpolated_values .- z_mat
# print("max difference is: ", maximum(diff))

function interp_slice(samples)
    samples = δz_mat
    samples_pos = [[x, y] for (x,y) in zip(δx_mat,δy_mat)]
    samples_pos = mapreduce(permutedims, vcat, samples_pos)' #Transforms to a Matrix 2xN
    @time "Multiquadratic" itp = interpolate(Multiquadratic(), samples_pos, samples);

    eval_points = [[x, y] for (x,y) in zip(x_mat,y_mat)]
    eval_points = mapreduce(permutedims, vcat, eval_points)'
    interpolated_values = evaluate(itp, eval_points)

    diff_values =  interpolated_values .- z_mat
    return vec(interpolated_values), vec(diff_values)
end

@time interpolated_values, diff_values = interp_slice(δz_mat)

print(typeof(interpolated_values))

#########
# PLOTS #
#########

#Display function values as color in a scatter plot
# fig    = Figure()
# ax1     = Axis(fig[1, 1])
# Colorbar(fig[1, 2], limits=(0,maximum(δz_mat)))
# ax2 = Axis(fig[1, 3])
# ax3 = Axis(fig[1, 4])
# Colorbar(fig[1, 5], limits=(0,maximum(diff_values)))
# scatter!(ax1,δx_mat, δy_mat, color=:blue)
# scatter!(ax1,x_mat, y_mat, color=z_mat)
# scatter!(ax2,x_mat, y_mat, color=interpolated_values)
# scatter!(ax3,x_mat, y_mat, color=diff_values)
# fig