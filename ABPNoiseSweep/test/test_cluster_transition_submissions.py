"""Exercise both standalone Slurm files with a local Julia shell stub.

Run from the project root (no sbatch and no simulation):
    python -B test/test_cluster_transition_submissions.py

Set ABP_TEST_BASH or BASH to choose a Bash executable. On Windows the default is
the user's installed Git Bash. Temporary outputs stay under this project's
test/ directory; resolved cleanup targets are checked before recursive removal.
"""

import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_CASES = [
    ("lambda2p5", 2.5, PROJECT / "run/submit_production_work_theta_lambda2p5.sh"),
    ("lambda8", 8.0, PROJECT / "run/submit_production_work_theta_lambda8.sh"),
]
EXPECTED_SPEEDS = ([0.200, 0.240, 0.260]
                   + [value / 1000 for value in range(280, 391, 5)]
                   + [0.400, 0.450, 0.500])


def find_bash():
    selected = os.environ.get("ABP_TEST_BASH") or os.environ.get("BASH")
    if selected:
        return selected
    windows_git = Path.home() / "AppData/Local/Programs/Git/bin/bash.exe"
    if windows_git.is_file():
        return str(windows_git)
    selected = shutil.which("bash")
    if not selected:
        raise RuntimeError("Bash not found; set ABP_TEST_BASH to a Bash executable")
    return selected


MOCK_JULIA = r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\0' "$@" > "$TEST_CLUSTER_CAPTURE_ARGS"
for name in "${!ABP_@}" JULIA_NUM_THREADS OPENBLAS_NUM_THREADS OMP_NUM_THREADS; do
    printf '%s=%s\0' "$name" "${!name-}" >> "$TEST_CLUSTER_CAPTURE_ENV"
done
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        exit 0
    fi
done
mkdir -p "$ABP_OUTPUT_DIR/data/abp_endpoint_conditioned_D0p025"
for (( index=0; index<${TEST_CLUSTER_RESULT_COUNT:-1}; index++ )); do
    printf 'fake result from lambda=%s speed=%s\n' "$ABP_POTENTIAL_LAMBDA" "$ABP_V" > "$ABP_OUTPUT_DIR/abp_endpoint_conditioned_stub${index}.jld2"
done
printf 'value\n1\n' > "$ABP_OUTPUT_DIR/data/abp_endpoint_conditioned_D0p025/stub.csv"
'''


class ClusterTransitionSubmissions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()
        cls.test_root = (PROJECT / "test").resolve()
        # Keep fixture paths below Windows MAX_PATH, including the long
        # plotting-ready CSV directory names created by the real wrappers.
        cls.temporary = tempfile.TemporaryDirectory(prefix="cs_", dir=cls.test_root)
        cls.workspace = Path(cls.temporary.name).resolve()
        if cls.workspace.parent != cls.test_root or not cls.workspace.name.startswith("cs_"):
            raise RuntimeError(f"Unexpected temporary workspace: {cls.workspace}")
        cls.fake_project = cls.workspace / "p s"
        (cls.fake_project / "scripts").mkdir(parents=True)
        (cls.fake_project / "scripts/run_fixed_D_speed_landscape.jl").write_bytes(b"# Julia is mocked by this test.\n")
        (cls.fake_project / "Project.toml").write_bytes(b"name = \"StubProject\"\n")
        cls.mock = cls.workspace / "mock_julia.sh"
        cls.mock.write_bytes(MOCK_JULIA.encode("utf-8"))
        cls.mock.chmod(cls.mock.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        cls.serial = 0

    @classmethod
    def tearDownClass(cls):
        # Check the exact resolved target again immediately before tempfile's
        # recursive cleanup. This never follows a computed path outside test/.
        target = cls.workspace.resolve()
        if (target.parent != cls.test_root or target == cls.test_root
                or not target.name.startswith("cs_")):
            raise RuntimeError(f"Refusing cleanup outside the controlled test workspace: {target}")
        cls.temporary.cleanup()

    def run_wrapper(self, script, *, index="0", arguments=("--dry-run",), extra=None, cpus="2"):
        type(self).serial += 1
        serial = type(self).serial
        capture_env = self.workspace / f"capture_{serial}.env"
        capture_args = self.workspace / f"capture_{serial}.args"
        # Do not inherit a previous simulation's settings. Individual tests
        # deliberately add stale values to verify the wrapper's pinned choices.
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith("ABP_") and not name.startswith("SLURM_")
                       and name not in {"JULIA_NUM_THREADS", "JULIA_CMD"}}
        environment.update(
            ABP_PROJECT_DIR=self.fake_project.as_posix(), JULIA_CMD=self.mock.as_posix(),
            SLURM_ARRAY_JOB_ID=f"preflight_{serial}", SLURM_CPUS_PER_TASK=str(cpus),
            TEST_CLUSTER_CAPTURE_ENV=capture_env.as_posix(),
            TEST_CLUSTER_CAPTURE_ARGS=capture_args.as_posix(),
            MSYS2_ARG_CONV_EXCL="*", MSYS_NO_PATHCONV="1",
        )
        if index is not None:
            environment["SLURM_ARRAY_TASK_ID"] = str(index)
        if extra:
            environment.update(extra)
        completed = subprocess.run(
            [self.bash, script.as_posix(), *arguments], cwd=PROJECT, env=environment,
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        captured = {}
        if capture_env.exists():
            captured = dict(part.decode("utf-8").split("=", 1)
                            for part in capture_env.read_bytes().split(b"\0") if part)
        argv = ([part.decode("utf-8") for part in capture_args.read_bytes().split(b"\0") if part]
                if capture_args.exists() else [])
        return completed, captured, argv

    def assert_success(self, completed):
        self.assertEqual(completed.returncode, 0,
                         f"Wrapper failed:\n{completed.stdout}\n{completed.stderr}")

    def test_both_scripts_have_exact_29_speed_grid_and_distinct_logs(self):
        log_paths, job_names = set(), set()
        for tag, _, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                text = script.read_text(encoding="utf-8")
                match = re.search(r"V_VALUES=\((.*?)\)", text, flags=re.DOTALL)
                self.assertIsNotNone(match)
                tokens = re.findall(r"\d+\.\d+", match.group(1))
                self.assertEqual([float(token) for token in tokens], EXPECTED_SPEEDS)
                self.assertEqual(len(tokens), 29)
                self.assertRegex(text, r"(?m)^#SBATCH\s+--array=0-28\s*$")
                self.assertRegex(text, r"(?m)^#SBATCH\s+--cpus-per-task=128\s*$")
                self.assertRegex(text, r"(?m)^#SBATCH\s+--mem=96G\s*$")
                job = re.search(r"(?m)^#SBATCH\s+--job-name=(.+)$", text).group(1).strip()
                self.assertIn(tag, job)
                self.assertNotIn(job, job_names)
                job_names.add(job)
                for kind in ["output", "error"]:
                    path = re.search(rf"(?m)^#SBATCH\s+--{kind}=(.+)$", text).group(1).strip()
                    self.assertIn(tag, path)
                    self.assertIn("%A_%a", path)
                    self.assertNotIn(path, log_paths)
                    log_paths.add(path)

    def test_both_scripts_parse_with_bash(self):
        for _, _, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                result = subprocess.run([self.bash, "-n", script.as_posix()], cwd=PROJECT,
                                        capture_output=True, text=True, timeout=30,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.assert_success(result)

    def test_endpoint_and_critical_tasks_select_correct_speed_and_default_chains(self):
        for tag, potential, script in SCRIPT_CASES:
            for index in [0, 11, 28]:
                with self.subTest(script=script.name, index=index):
                    result, captured, argv = self.run_wrapper(script, index=index)
                    self.assert_success(result)
                    self.assertAlmostEqual(float(captured["ABP_V"]), EXPECTED_SPEEDS[index])
                    self.assertEqual(float(captured["ABP_POTENTIAL_LAMBDA"]), potential)
                    self.assertEqual(captured["JULIA_NUM_THREADS"], "2")
                    self.assertEqual(captured["ABP_N_PROD_CHAINS"], "2")
                    self.assertEqual(captured["OPENBLAS_NUM_THREADS"], "1")
                    self.assertEqual(captured["OMP_NUM_THREADS"], "1")
                    self.assertIn("--threads=2", argv)
                    self.assertIn("--dry-run", argv)
                    self.assertTrue(any(value.endswith("scripts/run_fixed_D_speed_landscape.jl") for value in argv))
                    output = Path(captured["ABP_OUTPUT_DIR"])
                    self.assertIn(tag, output.parent.name)
                    self.assertTrue(output.is_relative_to(self.fake_project))
                    self.assertEqual(list(output.rglob("*.jld2")), [])

    def test_both_scripts_match_defaults_except_lambda_and_output_namespace(self):
        captures = []
        for _, _, script in SCRIPT_CASES:
            result, captured, _ = self.run_wrapper(script, index=11)
            self.assert_success(result)
            captures.append({name: value for name, value in captured.items()
                             if name.startswith("ABP_") and name not in {"ABP_POTENTIAL_LAMBDA", "ABP_OUTPUT_DIR"}})
        self.assertEqual(captures[0], captures[1])
        defaults = captures[0]
        self.assertEqual(defaults["ABP_N_ITER"], "200")
        self.assertEqual(defaults["ABP_N_ITER_STEPS_PER_ITER"], "800000000")
        self.assertEqual(defaults["ABP_N_PROD_OBS_TOTAL"], "55000000000")
        self.assertEqual(defaults["ABP_PATH_OBSERVATION_STRIDE"], "1000")
        self.assertEqual(defaults["ABP_PATH_TIME_STRIDE"], "1")
        self.assertEqual(defaults["ABP_ACTIVE_WORK_OBSERVATION_STRIDE"], "10000")
        self.assertEqual(defaults["ABP_N_PATH_THETA_BINS"], "180")
        self.assertEqual(defaults["ABP_SAVED_PATH_TIME_THIN"], "1")
        self.assertEqual(defaults["ABP_MAX_SAVED_PATHS_PER_WINDOW"], "120")

    def test_stale_physical_and_movie_environment_cannot_change_pinned_setup(self):
        stale = dict(ABP_POTENTIAL_LAMBDA="123", ABP_TRAJECTORY_T="987", ABP_DT="0.5",
                     ABP_INTEGRATOR="crank_nicolson", ABP_NOISE_PROPOSAL="pcn",
                     ABP_SAVE_MOVIE_PATHS="true", ABP_SAVE_MUCA_MOVIE_PATHS="true", ABP_V="999",
                     ABP_OUTPUT_DIR=(self.workspace / "stale output").as_posix())
        for tag, potential, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                result, captured, _ = self.run_wrapper(script, index=0, extra=stale)
                self.assert_success(result)
                self.assertEqual(float(captured["ABP_POTENTIAL_LAMBDA"]), potential)
                self.assertEqual(float(captured["ABP_TRAJECTORY_T"]), 40)
                self.assertEqual(float(captured["ABP_DT"]), 0.01)
                self.assertEqual(captured["ABP_INTEGRATOR"], "euler")
                self.assertEqual(captured["ABP_NOISE_PROPOSAL"], "random_walk")
                self.assertEqual(captured["ABP_SAVE_MOVIE_PATHS"], "false")
                self.assertEqual(captured["ABP_SAVE_MUCA_MOVIE_PATHS"], "false")
                self.assertEqual(float(captured["ABP_V"]), 0.2)
                self.assertIn(tag, Path(captured["ABP_OUTPUT_DIR"]).parent.name)
                self.assertNotEqual(captured["ABP_OUTPUT_DIR"], stale["ABP_OUTPUT_DIR"])

    def test_explicit_workload_and_storage_overrides_are_retained(self):
        overrides = dict(ABP_N_ITER="2", ABP_N_ITER_STEPS_PER_ITER="240", ABP_N_THERM_MUCA="0",
                         ABP_N_PROD_OBS_TOTAL="120", ABP_N_THERM_PROD="0", ABP_N_PROD_CHAINS="1",
                         ABP_PROD_STRIDE="10", ABP_ROUNDTRIP_STRIDE="10",
                         ABP_PATH_OBSERVATION_STRIDE="2", ABP_ACTIVE_WORK_OBSERVATION_STRIDE="4",
                         ABP_SAVED_PATH_TIME_THIN="3", ABP_MAX_SAVED_PATHS_PER_WINDOW="5")
        for _, _, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                result, captured, _ = self.run_wrapper(script, extra=overrides)
                self.assert_success(result)
                for name, expected in overrides.items():
                    self.assertEqual(captured[name], expected, name)

    def test_dry_run_in_middle_forwards_arguments_and_skips_result_collection(self):
        arguments = ("--integrator=euler", "--dry-run", "--integrator=crank_nicolson")
        for _, _, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                result, captured, argv = self.run_wrapper(script, arguments=arguments)
                self.assert_success(result)
                self.assertEqual(captured["ABP_INTEGRATOR"], "crank_nicolson")
                self.assertEqual(argv[-len(arguments):], list(arguments))
                root = Path(captured["ABP_OUTPUT_DIR"]).parent
                self.assertEqual(list(root.rglob("*.jld2")), [])
                self.assertEqual(list(root.rglob("*.csv")), [])

    def test_invalid_array_indices_are_rejected_before_julia(self):
        for _, _, script in SCRIPT_CASES:
            for index in [None, "-1", "29", "abc"]:
                with self.subTest(script=script.name, index=index):
                    result, captured, _ = self.run_wrapper(script, index=index)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(captured, {})

    def test_invalid_chain_counts_and_incompatible_strides_are_rejected(self):
        invalid = [dict(ABP_N_PROD_CHAINS="0"), dict(ABP_N_PROD_CHAINS="3"),
                   dict(ABP_PATH_OBSERVATION_STRIDE="0"),
                   dict(ABP_PATH_OBSERVATION_STRIDE="3", ABP_ACTIVE_WORK_OBSERVATION_STRIDE="4")]
        for _, _, script in SCRIPT_CASES:
            for values in invalid:
                with self.subTest(script=script.name, values=values):
                    result, captured, _ = self.run_wrapper(script, extra=values)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(captured, {})

    def test_lambda_cli_override_is_rejected(self):
        for _, _, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                result, captured, _ = self.run_wrapper(script, arguments=("--potential-lambda=123", "--dry-run"))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(captured, {})

    def test_success_collects_one_result_and_csv_into_lambda_specific_paths(self):
        destinations = []
        for tag, potential, script in SCRIPT_CASES:
            with self.subTest(script=script.name):
                result, captured, _ = self.run_wrapper(script, index=28, arguments=())
                self.assert_success(result)
                output = Path(captured["ABP_OUTPUT_DIR"])
                combined = output.parent / "combined"
                result_path = combined / f"abp_endpoint_conditioned_{tag}_D0p025_v0p500.jld2"
                csv_path = combined / "data" / f"abp_endpoint_conditioned_{tag}_D0p025_v0p500" / "stub.csv"
                self.assertTrue(result_path.is_file())
                self.assertIn(f"lambda={potential}", result_path.read_text(encoding="utf-8"))
                self.assertEqual(csv_path.read_text(encoding="utf-8").splitlines(), ["value", "1"])
                destinations.append(result_path)
        self.assertNotEqual(destinations[0], destinations[1])

    def test_collection_rejects_missing_or_multiple_jld2_results(self):
        for _, _, script in SCRIPT_CASES:
            for count in ["0", "2"]:
                with self.subTest(script=script.name, count=count):
                    result, captured, _ = self.run_wrapper(script, arguments=(), extra=dict(TEST_CLUSTER_RESULT_COUNT=count))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(captured)
                    self.assertIn("expected one JLD2", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
