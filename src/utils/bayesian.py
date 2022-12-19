import arviz as az
import pandas as pd
import pymc as pm


class HierarchicalModel:
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        prior_variation: float,
        prior_individual_variation: float,
        prior_error: float,
    ) -> None:
        id_values, id_uniques = pd.factorize(data["doc_id"])
        condition_values, condition_uniques = pd.factorize(data["condition"])
        nationality_values, nationality_uniques = pd.factorize(
            data["nationality"]
        )
        coords = {
            "participants": id_uniques,
            "condition_level": condition_uniques,
            "nationality_level": nationality_uniques,
        }
        with pm.Model(coords=coords) as model:
            participant_id = pm.MutableData("participant_id", id_values)
            condition = pm.MutableData("condition", condition_values)
            nationality = pm.MutableData("nationality", nationality_values)
            mean_intercept = pm.Normal("mean_intercept", mu=0.0, sigma=1.0)
            sd_intercept = pm.HalfNormal(
                "sd_intercept", prior_individual_variation
            )
            population_level_effect = pm.Normal(
                "population_level_effect",
                mu=0.0,
                sigma=prior_variation,
            )
            nation_level_effect = pm.Normal(
                "nation_level_effect",
                mu=population_level_effect,
                sigma=prior_variation,
                dims="nationality_level",
            )
            population_level_variation = pm.HalfNormal(
                "sd_beta_condition", prior_individual_variation
            )
            intercept = pm.Normal(
                "intercept",
                mu=mean_intercept,
                sigma=sd_intercept,
                dims=("participants", "nationality_level"),
            )
            beta_condition = pm.Normal(
                "beta_condition",
                mu=nation_level_effect,
                sigma=population_level_variation,
                dims=("participants", "nationality_level"),
            )
            error = pm.HalfNormal("error", prior_error)
            pm.Normal(
                outcome,
                mu=intercept[participant_id, nationality]
                + beta_condition[participant_id, nationality] * condition,
                sigma=error,
                observed=data[outcome],
            )
            self.model = model
            self.trace: az.InferenceData = pm.sample_prior_predictive(samples=100)  # type: ignore

    def add_trace(self, trace: az.InferenceData) -> None:
        if self.trace is None:
            self.trace = trace
        else:
            self.trace.extend(trace)

    def sample(self) -> az.InferenceData:
        with self.model:
            trace: az.InferenceData = pm.sample()  # type: ignore
            self.add_trace(trace)
        return trace  # type: ignore

    def sample_posterior_predictive(self) -> az.InferenceData:
        with self.model:
            trace: az.InferenceData = pm.sample_posterior_predictive(self.trace)  # type: ignore
            self.add_trace(trace)
        return trace  # type: ignore
