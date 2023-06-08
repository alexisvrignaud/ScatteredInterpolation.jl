# using ScatteredInterpolation
include("../src/ScatteredInterpolation.jl")
using GLMakie
using Random
using CUDA

freq = 10
δ = 0.2
sizex = 5
sizey = 5

f(x,y) = (cos.(x./freq) + sin.(2.2*x./freq)).*sin.(y./freq)

#Create initial grid
x_mat = vec([ x for x=1:sizex, y = 1:sizey])
y_mat = vec([ y for x=1:sizex, y = 1:sizey])

#Displace grid by random amounts to create the "scattered effect"
δx_mat = x_mat .+ 2*δ.*(rand(length(x_mat)).-1)
δy_mat = y_mat .+ 2*δ.*(rand(length(y_mat)).-1)

#Remove some positions to simulate "itf_ok" real world situations
delete_percentage = 0.5
n_positions = length(δx_mat)
delete_number = Int(floor(delete_percentage*n_positions))
delete_pos = sort(randperm(n_positions)[1:delete_number])
deleteat!(δx_mat,delete_pos)
deleteat!(δy_mat,delete_pos)

#Compute function values at these positions
δz_mat = f.(δx_mat,δy_mat)
z_mat = f.(x_mat,y_mat)

###############
# Interpolation

println("OK ##########################")

GPU = false

samples_pos = [[x, y] for (x,y) in zip(δx_mat,δy_mat)]
samples_pos = mapreduce(permutedims, vcat, samples_pos)' #Transforms to a Matrix 2xN

eval_points = [[x, y] for (x,y) in zip(x_mat,y_mat)]
eval_points = mapreduce(permutedims, vcat, eval_points)'

function interp_slice(samples_val, samples_pos, eval_pos)
    if GPU
        samples_pos = cu(samples_pos)
        samples_val = cu(samples_val)
        eval_pos = cu(eval_pos)
    end

    @time "Multiquadratic" itp = ScatteredInterpolation.interpolate(ScatteredInterpolation.Multiquadratic(), samples_pos, samples_val);

    interpolated_values = ScatteredInterpolation.evaluate(itp, eval_pos)

    diff_values =  interpolated_values .- z_mat
    return vec(interpolated_values), vec(diff_values)
end

using PyCall
scipy=pyimport("scipy")
qhull = scipy.spatial.qhull

function un_nan(x)
    if isnan(x)
        return 0.0
    else
        return x
    end
end

function ju_griddata(samples_val, samples_pos, eval_pos; method="linear")
    #cubic method is 2x slower for a 200x200 grid

    interpolated_values = scipy.interpolate.griddata(samples_pos',samples_val, eval_pos', method=method)
    diff_values =  interpolated_values .- z_mat

    # diff_values[diff_values .== NaN] .= 0.0
    diff_values = un_nan.(diff_values)

    s = sum(diff_values)
    
    println(s)
    println(isnan(s) || !isfinite(s))
    println( isnan(s) )
    println( !isfinite(s))

    return interpolated_values, diff_values
end

####################
####################
##### CUSTOM #######
####################

function CustomInterp(samples_val, samples_pos, eval_pos)
    println(size(samples_pos))
    tri = scipy.spatial.Delaunay(samples_pos')

    simplices = tri.simplices
    println(simplices)
    simplexes = tri.find_simplex(eval_pos')


    # for simplex in simplexes
    #     println(simplex)
    # end


    println(simplexes)
    println(size(simplexes))

    return tri
end

@time "Delaunay" tri = CustomInterp(δz_mat, samples_pos, eval_points)

# ##################
# # INTERPOLATIONS #

# @time "Python griddata" interpolated_values_py, diff_values_py = ju_griddata(δz_mat, samples_pos, eval_points)
# @time "Python griddata" interpolated_values_py, diff_values_py = ju_griddata(δz_mat, samples_pos, eval_points, method="cubic")
# @time "Julia Scattered Multiquadratic" interpolated_values, diff_values = interp_slice(δz_mat, samples_pos, eval_points)


# #########
# # PLOTS #
# #########

#Display function values as color in a scatter plot
fig    = Figure()
ax1     = Axis(fig[1, 1], title="Original scattered data") #Original scattered data
Colorbar(fig[1, 2], limits=(minimum(δz_mat),maximum(δz_mat)))
ax2 = Axis(fig[1, 3], title="Interpolated value ScatteredInterpolation") # Interpolated value julia
ax3 = Axis(fig[1, 4], title="Difference ground truth ScatteredInterpolation") # Difference ground truth
Colorbar(fig[1, 5], limits=(minimum(diff_values),maximum(diff_values)))
ax4 = Axis(fig[1, 6], title="Interpolated value Python griddata") # Interpolated value julia
ax5 = Axis(fig[1, 7], title="Difference ground truth Python griddata") # Difference ground truth
Colorbar(fig[1, 8], limits=(minimum(diff_values_py),maximum(diff_values_py)))

scatter!(ax1,δx_mat, δy_mat, color=δz_mat)
scatter!(ax1,x_mat, y_mat, color=z_mat)
# scatter!(ax2,x_mat, y_mat, color=interpolated_values)
# scatter!(ax3,x_mat, y_mat, color=diff_values)
# scatter!(ax4,x_mat, y_mat, color=interpolated_values_py)
# scatter!(ax5,x_mat, y_mat, color=diff_values_py)
fig




# println(tri.simplices);
# println(tri.neighbors);