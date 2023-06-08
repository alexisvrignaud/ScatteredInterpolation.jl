using GLMakie

nx = 50
ny = 60
nz = 70

A = [1.0 for i=1:nx, j=1:ny]
A_bool = [false for i=1:nx, j=1:ny]

xywindowSize = 12
edgeSize = 3

i,j,k = 1,1,1

a1  = @view A[i:i+xywindowSize, j:j+xywindowSize]

println(typeof(a1))

nx_windows = ceil(nx/xywindowSize)
ny_windows = ceil(nx/xywindowSize)

fig = Figure()
ax = Axis(fig[1,1])

for i in vcat([1], collect((xywindowSize-2*edgeSize):(xywindowSize-2*edgeSize):nx))
    for j in vcat([1], collect((xywindowSize-2*edgeSize):xywindowSize-2*edgeSize:ny))
    # for j in [1]
        println("STEP N")

        i_start_view = i
        i_end_view = i+xywindowSize
        i_start_mask = edgeSize
        i_end_mask = xywindowSize-edgeSize

        j_start_view = j
        j_end_view = j+xywindowSize
        j_start_mask = edgeSize
        j_end_mask = xywindowSize-edgeSize

        if i == 1
            i_start_view = 1
            i_end_view = i+xywindowSize
            i_start_mask = 1
            i_end_mask = xywindowSize-edgeSize
        elseif i+xywindowSize>=nx
            i_start_view = nx - xywindowSize
            i_end_view = nx
            i_start_mask = edgeSize
            i_end_mask = i_end_view - i_start_view
        end

        if j == 1
            j_start_view = 1
            j_end_view = j+xywindowSize
            j_start_mask = 1
            j_end_mask = xywindowSize-edgeSize
        elseif j+xywindowSize>=ny
            j_start_view = ny - xywindowSize
            j_end_view = ny
            j_start_mask = edgeSize
            j_end_mask = j_end_view - j_start_view
        end

        println("i_start_view ", i_start_view)
        println("i_end_view ", i_end_view)
        println("i_start_mask ", i_start_mask)
        println("i_end_mask ", i_end_mask)

        println("j_start_view ", j_start_view)
        println("j_end_view ", j_end_view)
        println("j_start_mask ", j_start_mask)
        println("j_end_mask ", j_end_mask)

        a_view = @view A[i_start_view:i_end_view, j_start_view:j_end_view]
        a_view[i_start_mask:i_end_mask, j_start_mask:j_end_mask] .+= 1
    end
end



heatmap!(ax,A)
fig