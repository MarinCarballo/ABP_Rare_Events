using JLD2, DelimitedFiles

@testset "Component moments survive JLD2 and CSV exports" begin
    acc, _, edges, _ = component_test_accumulator()
    for (wx, wy, weight) in ((2.0, -1.0, 0.5), (4.0, 3.0, 2.0))
        ABPNoiseSweep.abp_add_active_work_crossing!(
            acc, edges, component_test_metrics(wx, wy), weight,
        )
    end
    mktempdir() do directory
        path = joinpath(directory, "components.jld2")
        jldopen(path, "w") do file
            file["metadata/v"] = 0.31
            file["metadata/trajectory_T"] = 40.0
            file["active_work/crossing_y_bins"] = edges
            for weighting in (:biased, :unbiased)
                base = "active_work/$weighting"
                file["$base/mass"] = getproperty(acc, Symbol(:active_work_mass_, weighting))
                ABPNoiseSweep.abp_write_work_component_moments!(file, base, acc, weighting)
            end
            file["chain_statistics/n_chains"] = 1
            base = "chain_statistics/chains/1/active_work/unbiased"
            file["$base/mass"] = acc.active_work_mass_unbiased
            file["$base/mass2"] = acc.active_work_mass2_unbiased
            ABPNoiseSweep.abp_write_work_component_moments!(file, base, acc, :unbiased)
        end
        jldopen(path, "r") do file
            @test file["active_work/unbiased/sum_Wx"] == [0.0, 9.0]
            @test file["active_work/unbiased/sum_Wy"] == [0.0, 5.5]
            @test file["active_work/unbiased/sum_WxWy"] == [0.0, 23.0]
            outputs = ABPNoiseSweep.abp_export_work_component_csvs(file, directory)
            @test length(outputs) == 2
            pooled, header = readdlm(outputs[1], ','; header=true)
            names = vec(String.(header))
            column(name) = findfirst(==(name), names)
            row = findfirst(i -> pooled[i, column("weighting")] == "unbiased" &&
                                pooled[i, column("mass")] > 0, axes(pooled, 1))
            @test pooled[row, column("mean_Wx")] ≈ 9 / 2.5
            @test pooled[row, column("mean_Wy")] ≈ 5.5 / 2.5
            @test pooled[row, column("covariance_Wx_Wy")] ≈ 23 / 2.5 - (9 / 2.5)*(5.5 / 2.5)
            chain, chain_header = readdlm(outputs[2], ','; header=true)
            @test size(chain, 1) == 2
            @test "sum_WxWy" in vec(String.(chain_header))
        end
        # A legacy file must supply no fabricated component values or exports.
        legacy = joinpath(directory, "legacy.jld2")
        jldopen(legacy, "w") do file
            file["active_work/unbiased/mass"] = [1.0, 2.0]
        end
        jldopen(legacy, "r") do file
            @test isempty(ABPNoiseSweep.abp_export_work_component_csvs(file, directory))
        end
    end
end
