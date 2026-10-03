"""Bounded, client-only TabPFN workflows for targetless analyses.

These are explicit conditional column models, not the extensions' joint-density
runner. Every fit uses reference rows only; evaluation rows remain separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from .benchmark import BenchmarkError, _build_preprocessor
from .schemas import AnalysisResult, DatasetProfile


def analysis_columns(df: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    return [str(c) for c in df if c not in profile.identifier_columns
            and c not in profile.constant_columns and c not in profile.datetime_columns]


def _clients(token):
    try:
        import tabpfn_client
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    except ImportError:
        raise BenchmarkError("Install tabpfn-client in the shared environment to run this analysis.") from None
    if token:
        tabpfn_client.set_access_token(token)
    return TabPFNClassifier, TabPFNRegressor


def _split(df, columns):
    if not 2 <= len(columns) <= 8:
        raise BenchmarkError("Select between 2 and 8 informative fields for this initial evaluation.")
    if any(c not in df for c in columns):
        raise BenchmarkError("One of the selected fields is absent from this dataset.")
    frame = df[columns].replace([np.inf, -np.inf], np.nan).drop_duplicates()
    if len(frame) < 50:
        raise BenchmarkError("At least 50 distinct records are needed for this evaluation.")
    if len(frame) > 1000:
        frame = frame.sample(1000, random_state=42)
    train, test = train_test_split(frame, test_size=.2, random_state=42)
    return train.copy(), test.copy()


def _column_model(train, query, column, predictors, classifier, regressor):
    known = train[column].notna()
    if known.sum() < 30 or train.loc[known, column].nunique() < 2:
        raise BenchmarkError(f"`{column}` needs at least 30 known values and two distinct values.")
    numeric = pd.api.types.is_numeric_dtype(train[column]) and not pd.api.types.is_bool_dtype(train[column])
    y = train.loc[known, column]
    if not numeric:
        y = y.astype(str)
        if y.nunique() > 20:
            raise BenchmarkError(f"`{column}` has more than 20 categories. Choose a simpler field for this initial evaluation.")
    transform = _build_preprocessor(train.loc[known, predictors])
    X = np.asarray(transform.fit_transform(train.loc[known, predictors]), dtype=float)
    X_query = np.asarray(transform.transform(query[predictors]), dtype=float)
    if X.shape[1] == 0:
        raise BenchmarkError(f"No usable inputs remain for `{column}`.")
    model = regressor() if numeric else classifier()
    model.fit(X, y)
    return model, X_query, numeric


def _imputation(train, test, columns, classifier, regressor):
    rng = np.random.default_rng(42)
    metrics, recovered = [], []
    # Mask each field separately on unseen rows, retaining the other observed inputs.
    for column in columns:
        candidates = test.index[test[column].notna()]
        if len(candidates) < 5:
            raise BenchmarkError(f"Too few known holdout values to assess `{column}`.")
        mask = rng.choice(candidates, size=max(5, int(len(candidates) * .2)), replace=False)
        predictors = [c for c in columns if c != column]
        query = test.loc[mask].copy()
        query[column] = np.nan
        model, X, numeric = _column_model(train, query, column, predictors, classifier, regressor)
        predicted = np.asarray(model.predict(X))
        actual = test.loc[mask, column]
        baseline = train[column].median() if numeric else str(train[column].dropna().mode().iloc[0])
        truth = actual.to_numpy() if numeric else actual.astype(str).to_numpy()
        metric = "Average absolute error" if numeric else "Accuracy"
        score = mean_absolute_error(truth, predicted) if numeric else accuracy_score(truth, predicted)
        base_score = mean_absolute_error(truth, np.repeat(baseline, len(truth))) if numeric else accuracy_score(truth, np.repeat(baseline, len(truth)))
        metrics.append({"Field": column, "Measure": metric, "TabPFN": float(score), "Simple baseline": float(base_score), "Cells checked": len(mask)})
        for index, known, estimated in zip(mask, truth, predicted):
            recovered.append({"Record": str(index), "Field": column, "Known value": known, "Recovered value": estimated, "Numeric": numeric})
    # Actual missing cells are filled in held-out rows with models fitted on reference rows.
    completed = test.copy()
    flags = test.isna()
    for column in columns:
        missing = test[column].isna()
        if missing.any():
            model, X, _ = _column_model(train, test.loc[missing], column, [c for c in columns if c != column], classifier, regressor)
            completed[column] = completed[column].astype(object)
            completed.loc[missing, column] = model.predict(X)
    return AnalysisResult("imputation", pd.DataFrame(metrics),
                          {"recovery": pd.DataFrame(recovered), "original": test, "completed": completed, "missing_flags": flags},
                          [f"Reference: {len(train)} records; unseen evaluation: {len(test)} records.",
                           "Numeric errors use each field's original units; category accuracy is higher-is-better.",
                           "Validation hides known cells one field at a time. Real missingness can be harder to recover.",
                           "The download contains completed evaluation records only, with original missing-value flags. Filled values are estimates.",
                           "Method: column-by-column TabPFN prediction using the other selected fields."])


def _anomalies(train, test, columns, classifier, regressor):
    scores = pd.DataFrame(index=test.index)
    for column in columns:
        predictors = [c for c in columns if c != column]
        model, X, numeric = _column_model(train, test, column, predictors, classifier, regressor)
        if numeric:
            estimated = np.asarray(model.predict(X))
            scale = float(train[column].quantile(.75) - train[column].quantile(.25))
            if scale <= 0:
                scale = max(float(train[column].std()), 1e-8)
            surprise = np.abs(pd.to_numeric(test[column]).to_numpy() - estimated) / scale
        else:
            probabilities = np.asarray(model.predict_proba(X))
            lookup = {str(c): i for i, c in enumerate(model.classes_)}
            surprise = np.array([-np.log(max(probabilities[i, lookup[str(v)]], 1e-8)) if str(v) in lookup else -np.log(1e-8)
                                 for i, v in enumerate(test[column])])
        surprise[test[column].isna().to_numpy()] = np.nan
        scores[column] = surprise
    # Percentile ranks put field-specific scales on comparable footing.
    contributions = scores.rank(pct=True).fillna(0)
    combined = contributions.mean(axis=1)
    preprocessor = _build_preprocessor(train)
    baseline = IsolationForest(random_state=42).fit(preprocessor.fit_transform(train))
    isolation = -baseline.score_samples(preprocessor.transform(test))
    queue = test.copy()
    queue.columns = [f"Input: {column}" for column in queue.columns]
    queue.insert(0, "Record", test.index.astype(str))
    queue["Unusualness"] = combined.to_numpy()
    queue["Isolation Forest"] = isolation
    queue["Fields to investigate"] = [", ".join(contributions.loc[i].nlargest(3).index) for i in test.index]
    queue = queue.sort_values("Unusualness", ascending=False)
    agreement = pd.Series(combined.to_numpy()).rank().corr(pd.Series(isolation).rank())
    metrics = pd.DataFrame([{"Measure": "Unseen records ranked", "Value": len(test)},
                            {"Measure": "Rank agreement with Isolation Forest", "Value": float(agreement)}])
    contributions.insert(0, "Record", test.index.astype(str))
    return AnalysisResult("anomaly_detection", metrics,
                          {"queue": queue, "contributions": contributions},
                          [f"Reference: {len(train)} records; unseen evaluation: {len(test)} records.",
                           "Unusualness is the average field-surprise percentile, not the probability of fraud or failure.",
                           "Numeric surprise is scaled prediction error; category surprise comes from the probability of the recorded category.",
                           "Field bars describe what made a record unusual. Missing fields contribute zero; records with many gaps can rank lower.",
                           "Without verified review labels, precision and recall cannot be measured. Baseline rank agreement is not accuracy.",
                           "Method: conditional TabPFN models, not the extensions' joint-density detector."])


def _synthetic(train, test, columns, classifier, regressor, count):
    if not 20 <= count <= 500:
        raise BenchmarkError("Generate between 20 and 500 records for this initial evaluation.")
    rng = np.random.default_rng(42)
    # The first marginal is sampled empirically; later fields are sampled conditionally.
    generated = pd.DataFrame(index=range(count))
    first = columns[0]
    generated[first] = rng.choice(train[first].dropna().to_numpy(), size=count)
    for j, column in enumerate(columns[1:], 1):
        model, X, numeric = _column_model(train, generated, column, columns[:j], classifier, regressor)
        if numeric:
            quantiles = np.linspace(.05, .95, 19)
            values = np.asarray(model.predict(X, output_type="quantiles", quantiles=quantiles.tolist()))
            if values.shape != (len(quantiles), count):
                raise BenchmarkError("The API returned an unexpected quantile shape. Update tabpfn-client and retry.")
            generated[column] = [np.interp(rng.uniform(.05, .95), quantiles, np.sort(values[:, i])) for i in range(count)]
            if pd.api.types.is_integer_dtype(train[column]):
                generated[column] = np.rint(generated[column]).astype(int)
        else:
            probabilities = np.asarray(model.predict_proba(X))
            original_categories = {str(value): value for value in train[column].dropna().unique()}
            generated[column] = [original_categories[str(rng.choice(model.classes_, p=p / p.sum()))] for p in probabilities]
    baseline = train.sample(count, replace=True, random_state=42).reset_index(drop=True)
    fidelity = []
    for column in columns:
        numeric = pd.api.types.is_numeric_dtype(train[column]) and not pd.api.types.is_bool_dtype(train[column])
        for name, table in [("TabPFN conditional sampler", generated), ("Resampled baseline", baseline)]:
            if numeric:
                scale = max(float(train[column].std()), 1e-8)
                distance = wasserstein_distance(test[column].dropna().astype(float), table[column].dropna().astype(float)) / scale
                measure = "Distribution distance (scaled)"
            else:
                a, b = test[column].dropna().astype(str).value_counts(normalize=True), table[column].dropna().astype(str).value_counts(normalize=True)
                values = a.index.union(b.index)
                distance = .5 * (a.reindex(values, fill_value=0) - b.reindex(values, fill_value=0)).abs().sum()
                measure = "Category distribution distance"
            fidelity.append({"Field": column, "Model": name, "Measure": measure, "Distance": float(distance)})
    # A common numeric representation lets us assess relationships across mixed fields.
    transform = _build_preprocessor(train)
    transform.fit(train)
    observed_values = transform.transform(test)
    synthetic_values = transform.transform(generated)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_corr = np.nan_to_num(np.corrcoef(observed_values, rowvar=False))
        generated_corr = np.nan_to_num(np.corrcoef(synthetic_values, rowvar=False))
    train_keys = set(map(tuple, train.fillna("<missing>").astype(str).to_numpy()))
    duplication = float(np.mean([tuple(row) in train_keys for row in generated.fillna("<missing>").astype(str).to_numpy()]))
    metrics = pd.DataFrame([{"Measure": "Generated records", "Value": count},
                            {"Measure": "Exact match fraction against reference records", "Value": duplication},
                            {"Measure": "Average relationship difference", "Value": float(np.abs(observed_corr - generated_corr).mean())}])
    numeric_order = list(train.select_dtypes(include=[np.number, "bool"]).columns)
    ordered = numeric_order + [c for c in columns if c not in numeric_order]
    return AnalysisResult("synthetic_generation", metrics,
                          {"generated": generated, "observed": test, "fidelity": pd.DataFrame(fidelity),
                           "relationship_difference": pd.DataFrame(np.abs(observed_corr - generated_corr), index=ordered, columns=ordered)},
                          [f"Reference: {len(train)} records; untouched fidelity check: {len(test)} records.",
                           "Lower distribution and relationship differences indicate closer matches, not guaranteed business usefulness.",
                           "Method: empirical sampling of the first field, then conditional TabPFN category or quantile sampling in the selected field order.",
                           "This client-only sampler uses the middle 90% of numeric quantiles and can underrepresent extremes. It is not the extensions' joint generator.",
                           "Missingness is not synthesized. Relationship comparisons use encoded categories and are approximate.",
                           "Synthetic records do not guarantee privacy. Constraints and downstream utility still need domain review."])


def run_tabpfn_analysis(df: pd.DataFrame, profile: DatasetProfile, capability: str,
                       columns: list[str], access_token: str | None = None, *, generated_rows: int = 100) -> AnalysisResult:
    allowed = set(analysis_columns(df, profile))
    if any(c not in allowed for c in columns) or len(set(columns)) != len(columns):
        raise BenchmarkError("Choose distinct, informative fields; identifiers, constants, and dates are excluded from this initial analysis.")
    train, test = _split(df, columns)
    classifier, regressor = _clients(access_token)
    try:
        if capability == "imputation":
            result = _imputation(train, test, columns, classifier, regressor)
        elif capability == "anomaly_detection":
            result = _anomalies(train, test, columns, classifier, regressor)
        elif capability == "synthetic_generation":
            return _synthetic(train, test, columns, classifier, regressor, generated_rows)
        else:
            raise BenchmarkError("This capability requires a supervised evaluation.")
        identifiers = df.loc[test.index, profile.identifier_columns].copy()
        identifiers.columns = [f"Identifier: {c}" for c in identifiers]
        result.artifacts["identifiers"] = identifiers
        if capability == "anomaly_detection":
            result.artifacts["queue"] = result.artifacts["queue"].join(identifiers)
        return result
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("TabPFN analysis failed. Check the token, quota, connection, and field types, then retry.") from None
