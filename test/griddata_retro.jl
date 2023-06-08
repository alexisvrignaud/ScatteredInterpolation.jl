using GLMakie
using Random
using CUDA
using Revise

freq = 10
δ = 0.2
sizex = 8
sizey = 8

f(x,y) = (cos.(x./freq) + sin.(2.2*x./freq)).*sin.(y./freq)

#Create initial grid
x_mat = vec([ x for x=1:sizex, y = 1:sizey])
y_mat = vec([ y for x=1:sizex, y = 1:sizey])

#Displace grid by random amounts to create the "scattered effect"
δx_mat = x_mat .+ 2*δ.*(rand(length(x_mat)).-1)
δy_mat = y_mat .+ 2*δ.*(rand(length(y_mat)).-1)

#Remove some positions to simulate "itf_ok" real world situations
delete_percentage = 0
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
GPU = false

samples_pos = [[x, y] for (x,y) in zip(δx_mat,δy_mat)]
samples_pos = mapreduce(permutedims, vcat, samples_pos)' #Transforms to a Matrix 2xN

eval_points = [[x, y] for (x,y) in zip(x_mat,y_mat)]
eval_points = mapreduce(permutedims, vcat, eval_points)'

# Python call reference

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


    for i in size(samples_pos)[2]
        println(i)
    end


    println(simplexes)
    println(size(simplexes))

    return tri
end

# @time "Delaunay" tri = CustomInterp(δz_mat, samples_pos, eval_points)

# ##################
# # INTERPOLATIONS #

# @time "Python griddata" interpolated_values_py, diff_values_py = ju_griddata(δz_mat, samples_pos, eval_points)
# @time "Python griddata" interpolated_values_py, diff_values_py = ju_griddata(δz_mat, samples_pos, eval_points, method="cubic")
# @time "Julia Scattered Multiquadratic" interpolated_values, diff_values = interp_slice(δz_mat, samples_pos, eval_points)


# #########
# # PLOTS #
# #########

# #Display function values as color in a scatter plot
# fig    = Figure()
# ax1     = Axis(fig[1, 1], title="Original scattered data") #Original scattered data
# Colorbar(fig[1, 2], limits=(minimum(δz_mat),maximum(δz_mat)))
# ax2 = Axis(fig[1, 3], title="Interpolated value custom") # Interpolated value julia
# ax3 = Axis(fig[1, 4], title="Difference ground truth custom") # Difference ground truth
# Colorbar(fig[1, 5], limits=(minimum(diff_values),maximum(diff_values)))
# ax4 = Axis(fig[1, 6], title="Interpolated value Python griddata") # Interpolated value julia
# ax5 = Axis(fig[1, 7], title="Difference ground truth Python griddata") # Difference ground truth
# Colorbar(fig[1, 8], limits=(minimum(diff_values_py),maximum(diff_values_py)))

# scatter!(ax1,δx_mat, δy_mat, color=δz_mat)
# scatter!(ax1,x_mat, y_mat, color=z_mat)
# scatter!(ax2,x_mat, y_mat, color=interpolated_values)
# scatter!(ax3,x_mat, y_mat, color=diff_values)
# scatter!(ax4,x_mat, y_mat, color=interpolated_values_py)
# scatter!(ax5,x_mat, y_mat, color=diff_values_py)
# fig

struct DelaunayStruct
    #Input data
    x_samples_positions :: Vector{AbstractFloat}
    y_samples_positions :: Vector{AbstractFloat}
    samples_values :: Vector{AbstractFloat}

    #Delaunay triangulation
    simplices :: Matrix{AbstractFloat}
    valid_simplices ::Vector{Bool}

    #Evaluation 
    x_eval_positions :: Vector{AbstractFloat}
    y_eval_positions :: Vector{AbstractFloat}

    eval_matrix :: Array{AbstractFloat} #Should be a sparse array: 3 non zero values per row

end

function ComputeDelaunay(x_vec, y_vec, samples_value; maxtriangleratio=2)
    samples_pos = [[x, y] for (x,y) in zip(x_vec,y_vec)]
    samples_pos = mapreduce(permutedims, vcat, samples_pos) #Transforms to a Matrix 2xN

    #Qhull options should be thoroughly investigated. Side triangles are too flat and will degrade the data
    tri = scipy.spatial.Delaunay(samples_pos) #, qhull_options="Qbb Qc Qz Q2 Qw Qm"
    simplices = tri.simplices .+1 #From 0-indexed Python array to 1-indexed Julia

    if maxtriangleratio >0
        #Delete "flat outer triangles"
        valid_simplices = ones(Bool, size(simplices)[1])

        #Get list of exterior triangles
        convex_hull = tri.convex_hull .+1 #From 0-indexed Python array to 1-indexed Julia
        convex_hull_list = sort(unique(vec(convex_hull)))
    
        for i in 1:size(simplices)[1]
            simplice = simplices[i,:]
            #Check if outer triangles
            is_outer = simplice[1] in convex_hull_list || simplice[2] in convex_hull_list || simplice[3] in convex_hull_list
            if is_outer
                x_triangle = [x_vec[simplice[1]], x_vec[simplice[2]], x_vec[simplice[3]]]
                y_triangle = [y_vec[simplice[1]], y_vec[simplice[2]], y_vec[simplice[3]]]

                #Check span in both directions to exclude too "thin" triangles that are likely to interpolate using too far data
                x_span = abs(maximum(x_triangle)-minimum(x_triangle))
                y_span = abs(maximum(y_triangle)-minimum(y_triangle))
                form_ratio = max(x_span, y_span)/min(x_span, y_span)

                if form_ratio > maxtriangleratio
                    valid_simplices[i] = false
                end
            end
        end
    end


    #Vertex k
    vertex_k = 3
    neighbors = indices[indptr[vertex_k]:indptr[vertex_k+1]]
    print(neighbors)

    return DelaunayStruct(x_vec,y_vec,samples_value, )

end

function DisplayDelaunayTriangulation(x_vec, y_vec; maxtriangleratio=2, simplexfocus= 2)

    samples_pos = [[x, y] for (x,y) in zip(x_vec,y_vec)]
    samples_pos = mapreduce(permutedims, vcat, samples_pos) #Transforms to a Matrix 2xN

    # Figure Delaunay
    fig    = Figure()
    ax1     = Axis(fig[1, 1], title="Positions", aspect=AxisAspect(1)) #Original scattered data

    scatter!(ax1,x_vec, y_vec, color=:blue)

    #Qhull options should be thoroughly investigated. Side triangles are too flat and will degrade the data
    tri = scipy.spatial.Delaunay(samples_pos) #, qhull_options="Qbb Qc Qz Q2 Qw Qm"
    simplices = tri.simplices .+1 #From 0-indexed Python array to 1-indexed Julia

    if maxtriangleratio >0
        #Delete "flat outer triangles"
        valid_simplices = ones(Bool, size(simplices)[1])

        #Get list of exterior triangles
        convex_hull = tri.convex_hull .+1 #From 0-indexed Python array to 1-indexed Julia
        convex_hull_list = sort(unique(vec(convex_hull)))
    
        for i in 1:size(simplices)[1]
            simplice = simplices[i,:]
            #Check if outer triangles
            is_outer = simplice[1] in convex_hull_list || simplice[2] in convex_hull_list || simplice[3] in convex_hull_list
            if is_outer
                x_triangle = [x_vec[simplice[1]], x_vec[simplice[2]], x_vec[simplice[3]]]
                y_triangle = [y_vec[simplice[1]], y_vec[simplice[2]], y_vec[simplice[3]]]

                #Check span in both directions to exclude too "thin" triangles that are likely to interpolate using too far data
                x_span = abs(maximum(x_triangle)-minimum(x_triangle))
                y_span = abs(maximum(y_triangle)-minimum(y_triangle))
                form_ratio = max(x_span, y_span)/min(x_span, y_span)

                if form_ratio > maxtriangleratio
                    valid_simplices[i] = false
                end
            end
        end
    end

    for i in 1:size(simplices)[1] 
        simplice = simplices[i,:]

        x_triangle = [x_vec[simplice[1]], x_vec[simplice[2]], x_vec[simplice[3]]]
        y_triangle = [y_vec[simplice[1]], y_vec[simplice[2]], y_vec[simplice[3]]]

        if valid_simplices[i]
            _color=:grey
        else
            _color=:red
        end

        lines!(ax1, x_triangle, y_triangle, color=_color)        
    end

    #Display hull
    convex_hull = tri.convex_hull .+1 #From 0-indexed Python array to 1-indexed Julia
    for outerline in eachrow(convex_hull)
        x_line = [x_vec[outerline[1]],x_vec[outerline[2]]]
        y_line = [y_vec[outerline[1]],y_vec[outerline[2]]]
        lines!(ax1, x_line, y_line, color=:green)
    end

    #Display neighbors of a specific simplex
    k = 2
    (indptr, indices) = tri.vertex_neighbor_vertices
    indptr = indptr .+ 1
    indices = indices .+ 1
    println(size(x_vec))
    println(size(simplices))
    println(size(indptr))
    println(size(indices))

    for i in 1:size(simplices)[1] 
        simplice = simplices[i,:]

        x_triangle = [x_vec[simplice[1]], x_vec[simplice[2]], x_vec[simplice[3]]]
        y_triangle = [y_vec[simplice[1]], y_vec[simplice[2]], y_vec[simplice[3]]]

        if valid_simplices[i]
            _color=:grey
        else
            _color=:red
        end

        lines!(ax1, x_triangle, y_triangle, color=_color)        
    end

    #Display vertex of interest
    vertex_k = 15
    neighbors = indices[indptr[vertex_k]:indptr[vertex_k+1]]
    print(neighbors)
    x_neighbors = x_vec[neighbors]
    y_neighbors = y_vec[neighbors]

    scatter!(ax1, x_neighbors, y_neighbors, color=:orange)
    scatter!(ax1, x_vec[vertex_k], y_vec[vertex_k], color=:red)     

    fig
end

DisplayDelaunayTriangulation(δx_mat, δy_mat, maxtriangleratio=2)
