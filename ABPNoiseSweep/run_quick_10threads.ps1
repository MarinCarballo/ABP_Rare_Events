$env:JULIA_NUM_THREADS = '10'
$env:ABP_D_VALUES = '0.01'
$env:ABP_N_ITER = '15'
$env:ABP_N_ITER_STEPS_PER_ITER = '20000000'
$env:ABP_N_THERM_MUCA = '1000'
$env:ABP_N_PROD_OBS_TOTAL = '1000000'
$env:ABP_N_THERM_PROD = '1000'
$env:ABP_PROD_STRIDE = '1000'
$env:ABP_ROUNDTRIP_STRIDE = '1000'
$env:ABP_OUTPUT_DIR = 'quick_10threads_test_fix'

julia --threads=10 --project=. scripts/run_noise_sweep.jl
