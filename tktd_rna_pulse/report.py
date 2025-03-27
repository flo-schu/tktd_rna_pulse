import os
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm, ttest_1samp
import pandas as pd
import xarray as xr

from pymob.sim.report import Report, reporting
from tktd_rna_pulse.plot import pretty_posterior_plot_multisubstance

def _local_variance(residuals: np.ndarray):
    """This uses aggregated data residuals[id,time].mean(id).
    This uses the average variance of all 3-value pairs of direct neighbors.
    
    """
    local_variance = [np.var(r) for r in zip(
        np.roll(residuals, shift=1)[1:-1],
        np.roll(residuals, shift=0)[1:-1],
        np.roll(residuals, shift=-1)[1:-1]
    )]
    global_variance = np.var(residuals)

    return np.mean(local_variance) / global_variance

def _replicate_variance(residuals: np.ndarray):
    """This uses unaggregated data residuals[id,time]"""
    replicate_variance = [np.var(r[~np.isnan(r)]) for r in residuals.T]
    global_variance = np.var(residuals[~np.isnan(residuals)])

    return np.mean(replicate_variance) / global_variance

def _deviation_probability(residuals: np.ndarray):
    """This uses unaggregated data residuals[id,time]"""
    deviation_tests = [ttest_1samp(r[~np.isnan(r)], popmean=0) for r in residuals.T]

    return np.sum([np.log(test.pvalue) for test in deviation_tests])

def _significant_deviations(residuals: np.ndarray, alpha=0.05):
    """This uses unaggregated data residuals[id,time]"""
    deviation_tests = [ttest_1samp(r[~np.isnan(r)], popmean=0) for r in residuals.T]

    return np.sum([test.pvalue < alpha for test in deviation_tests])

def _autocorrelation(residuals, lag=1):
    """This uses unaggregated data residuals[id,time]"""
    return pd.Series(residuals).autocorr(lag=lag)

class MolecularTKTDReport(Report):
    obs_transform_funcs = {"survival": lambda x: x / x.max("id")}

    @reporting
    def model_inadequacy_metrics(self, idata, indices: Dict[str,xr.DataArray], index:str):
        """Compute different Model inadequacy metrics for comparison

        Parameters
        ----------

        idata : az.InferenceData
            The inference data returned by sim.inferer.idata
        index : xr.DataArray
            a single index of the sim.indices dictionary
        """
        autocorr_lag = 1

        description = (
            "The different metrics are measures for the model inadequacy. The comparison "+
            "to `metric_value_if_normal_dist` is a simulation of normally distributed "+
            "residuals that have the same data structure in terms of dimensionality "+
            "(id x time) and missing values. If the `metric_value` falls within that interval, the model can be assumed as not inadequate.\n\n"+
            "- **autocorrelation**: Measures the correlation of the residuals with\n"+
            "  themselves with a lag of 1. High absolute autocorrelation means, the\n"+
            "  variable is not normally distributed. Ideal would be values close to 0.\n"+
            "- **deviation log-prob**: Uses a t-test to estimate the probability of\n"+
            "  the replicates at a time t being different from zero. The result is\n"+
            "  the summed log-probability. Low (negative) log probs indicate high \n"+
            "  probability for deviation.\n"+
            "- **significant deviations**: Uses a t-test to estimate the probability \n"+
            "  of the replicates at a time t being different from zero. The result is\n"+
            "  the number of significant deviations (for an alpha level of 0.05).\n"+
            "  High number of deviations indicate an inadequate model\n"+
            "- **local/global variance**: This metric calculates the local variance\n"+
            "  as a rolling variance of always 3 direct neighboring residuals. The\n"+
            "  local variances are then averaged and divided by the global averages\n"+
            "  of all residuals. The basis for the calculation is the residuals averaged\n"+
            "  by id. Values close to 1 indicate an adequate model \n"+
            "- **replicate/global variance**: This metric calculates the replicate \n"+
            "  variance at time t, averages it and divides the number by the global\n"+
            "  variance. The local variances are then averaged and divided by the global\n"+
            "  averages of all residuals. The basis for the calculation is the residuals\n"+
            "  averaged by id. Values close to 1 indicate an adequate model\n"
        )

        df = []
        self._write("### Residuals")

        for endpoint in list(idata.posterior_residuals.data_vars.keys()):
            for s_i in range(len(np.unique(indices[index].values))):
                # substance = "diuron"
                selector = indices[index] == s_i
                substance = np.unique(indices[index].sel(id=selector)[index])[0]

                residuals = idata.posterior_residuals[endpoint].mean(("chain", "draw"))
                residuals = residuals.sel(id=selector, drop=True)
                residuals = residuals.sel(time=residuals.time.values[1:])
                # residuals = residuals.where((~residuals.isnull()).sum("time") != 0, drop=True)
                residuals = residuals.dropna("id", how="all")
                residuals = residuals.where(residuals.count("id") > 2, drop=True)
                residuals = residuals.dropna("time", how="all")
                rep_var = _replicate_variance(residuals=residuals.values)
                sig_dev = _significant_deviations(residuals=residuals.values)
                dev_pro = _deviation_probability(residuals=residuals.values)
                
                # statistics on the mean
                residuals_mean = residuals.mean("id")#.interp(time=np.arange(24,121))
                loc_var = _local_variance(residuals=residuals_mean.values)
                autocor = _autocorrelation(residuals=residuals_mean.values, lag=autocorr_lag)


                rep_var_if_normal = []
                loc_var_if_normal = []
                sig_dev_if_normal = []
                dev_pro_if_normal = []
                autocor_if_normal = []
                for i in range(100):
                    r = norm(0, residuals.std()).rvs(residuals.shape)
                    r[residuals.isnull().values] = np.nan
                    rep_var_if_normal.append(_replicate_variance(r))
                    sig_dev_if_normal.append(_significant_deviations(r))
                    dev_pro_if_normal.append(_deviation_probability(r))

                    # statistics on the mean
                    loc_var_if_normal.append(_local_variance(np.ma.masked_invalid(r).mean(axis=0)))
                    autocor_if_normal.append(_autocorrelation(np.ma.masked_invalid(r).mean(axis=0), lag=autocorr_lag))
                    

                def metric_report(data_var, substance, metric_name, metric_value, values_if_normal):
                    value_normal = np.round(np.mean(values_if_normal), 2)
                    quantiles_normal = np.round(np.quantile(values_if_normal, q=[0.05, 0.95]), 2)

                    return pd.Series({
                        "data_variable": data_var, "index": substance, 
                        "metric": metric_name, "metric_value": metric_value,
                        "metric_value_if_normal_dist": f"{value_normal}"+"[{0},{1}]".format(*quantiles_normal)
                    })

                series = metric_report(endpoint, substance, "local/global variance", loc_var, loc_var_if_normal)
                df.append(series)
                series = metric_report(endpoint, substance, "replicate/global variance", rep_var, rep_var_if_normal)
                df.append(series)
                series = metric_report(endpoint, substance, "significant deviations", sig_dev, sig_dev_if_normal)
                df.append(series)
                series = metric_report(endpoint, substance, "deviation log-prob", dev_pro, dev_pro_if_normal)
                df.append(series)
                series = metric_report(endpoint, substance, "autocorrelation", autocor, autocor_if_normal)
                df.append(series)

                fig, ax = plt.subplots(1,1, figsize=(6, 3))
                ax.plot(residuals.time, np.zeros_like(residuals.time), ls="--", color="black", lw=1)
                ax.plot(residuals.time, residuals.T, ls="", marker="o", color="tab:blue")
                ax.plot(residuals_mean.time, residuals_mean, ls="-", color="black", lw=2)
                ax.fill_between(residuals.time, *residuals.quantile(dim="id", q=[.05, .95]), color="grey", alpha=.5)
                ax.set_title(substance.capitalize())
                ax.set_ylabel(f"Standardized residuals ({endpoint})")
                ax.set_xlabel("Time")
                fig.tight_layout()
                fig.savefig(os.path.join(self.config.case_study.output_path, f"residuals_{endpoint}_{substance}.png"))
                self._write(f"![Resiudal {endpoint} dynamics of {substance}](residuals_{endpoint}_{substance}.png)")

                plt.close()

        self._write("### Model inadequacy")
        self._write(description)
        df = pd.DataFrame(df).sort_values(by=["metric", "data_variable", "index"]).set_index(["metric", "data_variable", "index"])
        out = os.path.join(self.config.case_study.output_path, "model_inadequacy_metrics.csv")
        df.to_csv(out)
        self._write(df.reset_index().to_markdown())

        return out

    @reporting
    def visualizations(self, sim):
        _, outs = pretty_posterior_plot_multisubstance(sim, save=True, show=False)
        for o in outs:
            self._write(f"![Posterior model fits]({os.path.basename(o)})")

        return outs