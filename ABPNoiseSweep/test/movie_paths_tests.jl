using Random
using StaticArrays
using JLD2

@testset "diagnostic trajectory snapshots" begin
    mod = ABPNoiseSweep
    @test mod.abp_snapshot_indices(8, 3) == [1, 4, 7, 8]
    @test mod.abp_snapshot_indices(7, 3) == [1, 4, 7]
    @test mod.abp_snapshot_indices(1, 20) == [1]
    @test mod.abp_movie_effective_stride(10, 1000, 20) == 50
    @test mod.abp_movie_effective_stride(100, 1000, 20) == 100

    sys = (xs=[SVector(Float64(k), -Float64(k)) for k in 1:8],
           θs=collect(0.0:0.2:1.4), dt=0.1)
    original = deepcopy(sys)
    acc = (movie_paths=Any[], saved_paths=[Any[]],)
    function save_snapshot(step; final=false, capacity=3)
        mod.abp_maybe_save_movie_path!(acc, sys;
            chain_id=1, mcmc_step=step, xT_now=8.0, yT_now=-8.0,
            ymean_now=-4.5, yint_now=-3.15, bias_value_now=8.0, w_path=NaN,
            max_movie_paths=capacity, movie_path_time_thin=3,
            phase=:muca, iteration=2, iteration_step=step-10,
            sampling_step=step-2, replace_last=final)
    end
    for step in (10, 20, 30, 40)
        save_snapshot(step)
    end
    @test length(acc.movie_paths) == 3
    @test last(acc.movie_paths).mcmc_step == 30
    save_snapshot(99; final=true)
    @test length(acc.movie_paths) == 3
    path = last(acc.movie_paths)
    @test path.mcmc_step == 99
    @test path.iteration == 2
    @test path.phase == "muca"
    @test path.sampling_step == 97
    @test path.xs[end] == sys.xs[end]
    @test path.physical_times ≈ [0.0, 0.3, 0.6, 0.7]
    @test isnan(path.unbias_weight_shifted)
    @test sys == original
    sys.xs[1] = SVector(-10.0, -20.0)
    @test path.xs[1] == original.xs[1]

    mod.abp_maybe_save_path!(acc, sys, 1, 8.0, -8.0, -4.5, -3.15, 8.0, 0.25;
        max_saved_paths_per_window_chain=0, saved_path_time_thin=3)
    @test isempty(acc.saved_paths[1])
    mod.abp_maybe_save_path!(acc, sys, 1, 8.0, -8.0, -4.5, -3.15, 8.0, 0.25;
        max_saved_paths_per_window_chain=1, saved_path_time_thin=1,
        chain_id=2, mcmc_step=200)
    @test length(acc.saved_paths[1][1].xs) == length(sys.xs)
    @test acc.saved_paths[1][1].chain_id == 2

    mktempdir() do dir
        filename = joinpath(dir, "snapshots.jld2")
        jldopen(filename, "w") do file
            file["metadata/dt"] = sys.dt
            file["metadata/movie_path_time_thin"] = 3
            mod.abp_write_movie_group!(file, "muca_movie_trajectories", [path])
        end
        jldopen(filename, "r") do file
            outputs = mod.abp_export_movie_csvs(file, dir,
                "muca_movie_trajectories", "muca_movie", "muca")
            @test length(outputs) == 2
            rows = readlines(outputs[2])
            @test length(rows) == 5
            columns = split(last(rows), ',')
            @test parse(Float64, columns[5]) ≈ 0.7
            @test parse(Float64, columns[6]) == 8.0
            @test columns[17] == "muca"
            @test parse(Int, columns[18]) == 2
            @test parse(Int, columns[20]) == 97
            @test file["muca_movie_trajectories/1/x"] == getindex.(path.xs, 1)
        end
    end
end
