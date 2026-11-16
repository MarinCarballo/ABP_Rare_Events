"""Notebook cell: fit near_m1 work to v(dx*cos(theta*) + dy*sin(theta*)).

Also fit near_m0 work to v*dx, independently of the angle histograms.
Execute after the notebook's work-analysis cells; no simulation data are changed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from IPython.display import display

FIT_M_SLICE = "near_m1"
FIT_FLAT_M_SLICE = "near_m0"
FIT_USE_JACKKNIFE_SE = True
FIT_RUN_SB = globals().get("FIT_RUN_SB", True)


def fit_linear_work_projection(observed, design, errors=None, *, use_jackknife_se=True):
    """Fit a linear model without an intercept, with conditional parameter errors.

    Positive finite jackknife SEs are absolute WLS errors. When any exist, only
    rows with such errors enter WLS; otherwise use OLS on all finite rows.
    A rank-deficient design does not identify a unique parameter vector.
    """
    observed = np.asarray(observed, dtype=float)
    design = np.asarray(design, dtype=float)
    if design.ndim == 1:
        design = design[:, None]
    if observed.ndim != 1 or design.ndim != 2 or design.shape[0] != len(observed) or design.shape[1] < 1:
        raise ValueError("Expected one observed value per design row and at least one fit parameter")
    errors = np.full(len(observed), np.nan) if errors is None else np.asarray(errors, dtype=float)
    if errors.shape != observed.shape:
        raise ValueError("Fit errors must have the same shape as observed values")
    valid = np.isfinite(observed) & np.isfinite(design).all(axis=1)
    positive_se = np.isfinite(errors) & (errors > 0)
    weighted = bool(use_jackknife_se and np.any(valid & positive_se))
    fit_mask = valid & positive_se if weighted else valid
    n, p = int(fit_mask.sum()), design.shape[1]
    result = dict(coefficients=np.full(p, np.nan), covariance=np.full((p, p), np.nan),
                  parameter_se=np.full(p, np.nan), fit_mask=fit_mask, rank=0,
                  condition_number=np.inf, status="insufficient_data", n_fit=n,
                  dof=n-p, weighted=weighted, chi2=np.nan, reduced_chi2=np.nan,
                  rmse=np.nan, residual_sum_squares=np.nan)
    if n == 0:
        return result
    matrix, target = design[fit_mask], observed[fit_mask]
    if weighted:
        matrix = matrix / errors[fit_mask, None]
        target = target / errors[fit_mask]
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    result["rank"] = rank
    if n < p:
        return result
    if rank < p:
        result["status"] = "rank_deficient"
        return result
    condition = float(singular[0] / singular[-1])
    coefficients = vt.T @ ((u.T @ target) / singular)
    residual = observed[fit_mask] - design[fit_mask] @ coefficients
    rss = float(residual @ residual)
    covariance = (vt.T / singular**2) @ vt
    chi2 = float(np.sum((residual / errors[fit_mask])**2)) if weighted else np.nan
    if not weighted:
        covariance = covariance * (rss / (n-p)) if n > p else np.full((p, p), np.nan)
    result.update(coefficients=coefficients, covariance=covariance,
                  parameter_se=np.sqrt(np.maximum(0, np.diag(covariance))),
                  condition_number=condition, status="weakly_identified" if condition > 1e8 else "ok",
                  chi2=chi2, reduced_chi2=chi2/(n-p) if weighted and n > p else np.nan,
                  rmse=float(np.sqrt(rss/n)), residual_sum_squares=rss)
    return result


def _mark_known_work_copies(rows, modal_rows):
    """Count known copied exports once, only when their work statistics agree."""
    rows = rows.copy()
    representatives = {}
    key = "identical_pdf_representative_case_id"
    if key in modal_rows:
        for case_id, group in modal_rows.groupby("case_id", sort=False):
            known = group[key].dropna().unique()
            if len(known) == 1:
                representatives[case_id] = known[0]
    rows["copy_source_group"] = [representatives.get(case_id, case_id) for case_id in rows["case_id"]]
    fields = ["copy_source_group", "D", "v", "T", "mean_active_work", "chain_jackknife_se"]
    fields += [name for name in ["unbiased_mass", "weight_only_ess", "weighted_std_active_work",
                                "effective_m_lo", "effective_m_hi", "n_chains_total",
                                "theta_rad", "pdf_fingerprint"] if name in rows]
    rows["excluded_identical_copy"] = rows.duplicated(fields, keep="first")
    return rows


def _fit_projection_rows(rows, modal_rows, *, fit_y, use_jackknife_se):
    rows = _mark_known_work_copies(rows, modal_rows)
    candidates = ~rows["excluded_identical_copy"].to_numpy()
    speed = rows["v"].to_numpy(dtype=float)
    if fit_y:
        theta = rows["theta_rad"].to_numpy(dtype=float)
        design = np.column_stack((speed*np.cos(theta), speed*np.sin(theta)))
    else:
        design = speed[:, None]
    result = fit_linear_work_projection(rows["mean_active_work"].to_numpy()[candidates],
                                        design[candidates], rows["chain_jackknife_se"].to_numpy()[candidates],
                                        use_jackknife_se=use_jackknife_se)
    rows["used_in_fit"] = False
    rows.loc[candidates, "used_in_fit"] = result["fit_mask"]
    coefficients, covariance, errors = (result[name] for name in ["coefficients", "covariance", "parameter_se"])
    rows["delta_x"], rows["delta_x_se"] = coefficients[0], errors[0]
    rows["delta_y"], rows["delta_y_se"] = (coefficients[1], errors[1]) if fit_y else (0.0, np.nan)
    rows["fit_status"] = result["status"]
    rows["v_cos_theta_delta_x"] = design[:, 0] * coefficients[0]
    rows["v_sin_theta_delta_y"] = design[:, 1] * coefficients[1] if fit_y else 0.0
    rows["projected_active_work"] = rows["v_cos_theta_delta_x"] + rows["v_sin_theta_delta_y"]
    rows["work_residual"] = rows["mean_active_work"] - rows["projected_active_work"]
    scale = speed**2 * rows["T"].to_numpy(dtype=float)
    rows["projected_active_work_normalized"] = np.divide(
        rows["projected_active_work"].to_numpy(dtype=float), scale,
        out=np.full(len(rows), np.nan), where=scale > 0)
    summary = {name: result[name] for name in ["status", "rank", "condition_number", "n_fit", "dof",
                                              "chi2", "reduced_chi2", "rmse", "residual_sum_squares"]}
    summary.update(delta_x_fit=coefficients[0], delta_x_se=errors[0],
                   n_parameters=design.shape[1], n_matched=len(rows),
                   n_known_copy_rows_excluded=int((~candidates).sum()),
                   method="inverse-variance WLS" if result["weighted"] else "unweighted OLS")
    if fit_y:
        denominator = errors[0]*errors[1]
        summary.update(delta_y_fit=coefficients[1], delta_y_se=errors[1],
                       delta_x_delta_y_covariance=covariance[0, 1],
                       delta_x_delta_y_correlation=float(np.clip(covariance[0, 1]/denominator, -1, 1))
                       if np.isfinite(denominator) and denominator > 0 else np.nan)
    return summary, rows


def fit_active_work_projection(modal_rows, work_summary, m_slices, *, sb_slice="near_m1",
                               flat_slice="near_m0", use_jackknife_se=True, fit_sb=True):
    """Return SB fits/predictions and the independent one-parameter flat fit."""
    for name in [sb_slice, flat_slice]:
        if name not in m_slices:
            raise ValueError(f"{name!r} is not defined in ACTIVE_WORK_M_SLICES")
    sb_results, sb_predictions, flat_results, flat_predictions = [], [], [], []
    if work_summary.empty:
        return tuple(pd.DataFrame() for _ in range(4))
    def measured_slice(name):
        rows = work_summary.loc[(work_summary["m_slice"] == name) & (work_summary["status"] == "ok")].copy()
        if rows["case_id"].duplicated().any():
            raise ValueError(f"Multiple measured rows for the same case_id in {name}")
        return rows
    sb_measured = measured_slice(sb_slice)
    flat_measured = measured_slice(flat_slice)
    if fit_sb and not modal_rows.empty and not sb_measured.empty:
        if modal_rows.duplicated(["case_id", "branch"]).any():
            raise ValueError("Multiple theta* modes per (case_id, branch); select one theta x slice first")
        # Verify physical settings after the case-ID join, retaining run identity.
        measured = sb_measured.rename(columns={name: "measured_"+name for name in ["D", "v", "T"]})
        matched = modal_rows.merge(measured, on="case_id", how="inner", validate="many_to_one")
        for name in ["D", "v", "T"]:
            if not np.allclose(matched[name], matched["measured_"+name], rtol=1e-12, atol=1e-14):
                raise ValueError(f"Theta/work settings disagree for {name} at the same case_id")
        for branch, rows in matched.groupby("branch", sort=True):
            summary, predictions = _fit_projection_rows(rows, modal_rows, fit_y=True,
                                                        use_jackknife_se=use_jackknife_se)
            summary.update(m_slice=sb_slice, branch=branch)
            sb_results.append(summary)
            sb_predictions.append(predictions)
    if not flat_measured.empty:
        summary, predictions = _fit_projection_rows(flat_measured, modal_rows, fit_y=False,
                                                    use_jackknife_se=use_jackknife_se)
        summary.update(m_slice=flat_slice, model="W_flat = v*delta_x")
        flat_results.append(summary)
        flat_predictions.append(predictions)
    return (pd.DataFrame(sb_results), pd.concat(sb_predictions, ignore_index=True) if sb_predictions else pd.DataFrame(),
            pd.DataFrame(flat_results), pd.concat(flat_predictions, ignore_index=True) if flat_predictions else pd.DataFrame())


sb_fit, sb_prediction, FIT_FLAT_DELTA_X, FLAT_PROJECTED_WORK_THEORY = fit_active_work_projection(
    globals().get("SB_WORK_THEORY", pd.DataFrame()), WORK_M_SLICE_SUMMARY, ACTIVE_WORK_M_SLICES,
    sb_slice=FIT_M_SLICE, flat_slice=FIT_FLAT_M_SLICE, use_jackknife_se=FIT_USE_JACKKNIFE_SE, fit_sb=FIT_RUN_SB)
if FIT_RUN_SB:
    FIT_DELTA_XY_BY_BRANCH, PROJECTED_WORK_THEORY = sb_fit, sb_prediction
    # Compatibility with the original pasted cell's result variable.
    FIT_DELTA_Y_BY_BRANCH = FIT_DELTA_XY_BY_BRANCH

print("SB: fit delta_x and delta_y per theta* branch." if FIT_RUN_SB else "Keeping the preceding SB fit unchanged.")
print("Flat: fit delta_x only (theta=0, no free intercept).")
print("Fits use raw full-path work. Errors condition on fixed modal angles and independent retained rows;")
print("known copied exports with identical work statistics count once. Coefficients are effective fit parameters.")
for table, filename in [(sb_fit, "active_work_delta_xy_fit_near_m1_by_branch"),
                         (FIT_FLAT_DELTA_X, "active_work_delta_x_fit_near_m0"),
                         (sb_prediction, "active_work_xy_projection_fit_near_m1_by_case"),
                         (FLAT_PROJECTED_WORK_THEORY, "active_work_flat_projection_fit_near_m0_by_case")]:
    if not table.empty:
        display(table)
        export_table(table, filename)
fit_tables = [(FIT_M_SLICE, sb_fit)] if FIT_RUN_SB else []
for label, table in fit_tables + [(FIT_FLAT_M_SLICE, FIT_FLAT_DELTA_X)]:
    lo, hi = ACTIVE_WORK_M_SLICES[label]
    print(f"{label} fit uses ONLY its configured requested m slice [{lo:g}, {hi:g}); effective saved-bin edges remain in predictions.")
    if table.empty:
        print(f"No matched work data/modes available for {label}.")
    else:
        for row in table.itertuples():
            if row.status != "ok":
                print(f"{label} {getattr(row, 'branch', 'flat')}: {row.status}, rank={row.rank}, condition={row.condition_number:g}")

# Separate raw/normalized comparisons for the SB and flat data selections.
for slice_name, predictions, fit_y in [(FIT_M_SLICE, sb_prediction, True),
                                       (FIT_FLAT_M_SLICE, FLAT_PROJECTED_WORK_THEORY, False)]:
    if predictions.empty:
        continue
    fixed_cols = ["D", "T"] if SWEEP_PARAMETER == "v" else ["v", "T"]
    measured_all = WORK_M_SLICE_SUMMARY.loc[(WORK_M_SLICE_SUMMARY["m_slice"] == slice_name)
                                           & (WORK_M_SLICE_SUMMARY["status"] == "ok")]
    styles = {"negative": ("#7B3294", "v", ":"), "positive": ("#8B1A1A", "^", "-."),
              "central": ("#555555", "o", ":"), "seam": ("#AA3377", "s", ":")}
    for fixed_values, measured in measured_all.groupby(fixed_cols, sort=True):
        title = ", ".join(f"{name}={value:g}" for name, value in zip(fixed_cols, fixed_values))
        matched = predictions[predictions["case_id"].isin(measured["case_id"])].copy()
        if matched.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        columns = [("mean_active_work", "chain_jackknife_se", "projected_active_work"),
                   ("mean_normalized_active_work", "normalized_chain_jackknife_se", "projected_active_work_normalized")]
        measured = measured.sort_values([SWEEP_PARAMETER, "case_id"])
        color = WORK_SLICE_COLORS[slice_name]
        for ax, (value, error, _) in zip(axes, columns):
            xx, yy, ee = (measured[col].to_numpy(dtype=float) for col in [SWEEP_PARAMETER, value, error])
            valid = np.isfinite(xx) & np.isfinite(yy)
            ax.plot(xx[valid], yy[valid], "o", ls="none", color=color, label=f"Measured: {slice_name}")
            with_se = valid & np.isfinite(ee) & (ee >= 0)
            ax.errorbar(xx[with_se], yy[with_se], yerr=BAND_SIGMA*ee[with_se], fmt="none", color=color, capsize=3)
        branches = matched.groupby("branch", sort=True) if fit_y else [("flat", matched)]
        grid = np.sort(measured[SWEEP_PARAMETER].unique())
        for branch, subset in branches:
            subset = subset.loc[~subset["excluded_identical_copy"]].sort_values([SWEEP_PARAMETER, "case_id"])
            if subset.empty or not np.isfinite(subset["projected_active_work"]).any():
                continue
            color, marker, style = styles.get(branch, ("0.25", "s", "--"))
            dx, dy = float(subset["delta_x"].iloc[0]), float(subset["delta_y"].iloc[0])
            label = (rf"{branch} $\theta_*$: $\Delta x={dx:.3g},\ \Delta y={dy:.3g}$" if fit_y
                     else rf"Flat fit: $\Delta x={dx:.5g}$")
            if subset[SWEEP_PARAMETER].duplicated().any():
                xx, curve, style = subset[SWEEP_PARAMETER], subset, "none"
            else:
                xx, curve = grid, subset.set_index(SWEEP_PARAMETER).reindex(grid)
            for ax, (_, _, reference) in zip(axes, columns):
                ax.plot(xx, curve[reference], marker=marker, ls=style, color=color, lw=1.3, ms=5, label=label)
        for ax in axes:
            ax.set_xlabel(SWEEP_PARAMETER)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.legend(fontsize=8)
        axes[0].set_ylabel(r"$\langle W_a(0,T)\rangle$ and fitted work")
        axes[1].set_ylabel(r"Work / $(v^2 T)$")
        model = r"$v(\Delta x\cos\theta_*+\Delta y\sin\theta_*)$" if fit_y else r"$v\Delta x$"
        fig.suptitle(f"Fit to {slice_name} only: {model}; {title}")
        fit_tag = "xy" if fit_y else "delta_x"
        finish_figure(fig, f"active_work_{fit_tag}_fit_{slice_name}_vs_{SWEEP_PARAMETER}_{title}")
