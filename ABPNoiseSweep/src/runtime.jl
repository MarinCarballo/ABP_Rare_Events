# Runtime/reporting helpers.

function abp_print_runtime_header()
    println("============================================================")
    println("ABP noise sweep cluster script")
    println("Started at: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    println("Active project: ", Base.active_project())
    println("Julia threads: ", Threads.nthreads())
    println("CPU threads visible to Julia: ", Sys.CPU_THREADS)
    println("Hostname: ", get(ENV, "HOSTNAME", "unknown"))
    println("============================================================")
    if Threads.nthreads() == 1
        @warn "Only one Julia thread is active. Start Julia with --threads=auto or --threads=\$SLURM_CPUS_PER_TASK to use all allocated cores."
    end
end
