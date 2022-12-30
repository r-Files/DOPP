import pandas as pd
import numpy as np
from numpy import sort


def get_yearly_deaths(df: pd.DataFrame, custom_index: list = None, include_zero: bool = True) -> pd.Series:
    """ Calculate yearly disaster deaths, assuming a continuous uniform distribution of deaths
    between Start_Year and End_Year of each disaster.

    Parameters
    ----------
    df: pd.DataFrame
        The original DataFrame
    custom_index: list or None
        The labels to group the deaths by. Must be a column in the data, but not "Start_Year" or "End_Year".
    include_zero: bool
        Include rows with 0 deaths.

    Returns
    -------
    pd.Series indexed by "Year" and custom_index if present
    """
    df: pd.DataFrame = df.copy()
    min_start_year = df["Start_Year"].min()
    max_start_year = df["Start_Year"].max()

    alternate_cust_idx = ["__CUST_IDX"]
    if custom_index is None:
        custom_index = alternate_cust_idx
        df[alternate_cust_idx] = 0

    df = df.loc[:, ["Start_Year", "End_Year", "Total_Deaths"] + custom_index]
    df["Duration_CalendarYears"] = df["End_Year"].subtract(df["Start_Year"], fill_value=0)
    df.drop("End_Year", axis=1, inplace=True)
    df["Yearly_Disaster_Deaths"] = df["Total_Deaths"].div(df["Duration_CalendarYears"] + 1, fill_value=np.NaN)
    df.rename(columns={"Start_Year": "Year"}, inplace=True)

    index_labels = ["Year"] + custom_index
    intra_year_disasters = df.loc[df["Duration_CalendarYears"] == 0, ["Total_Deaths"] + index_labels]
    perennial_disasters = df.loc[df["Duration_CalendarYears"] != 0, ["Yearly_Disaster_Deaths", "Duration_CalendarYears"] + index_labels]

    intra_year_disaster_deaths = intra_year_disasters.groupby(index_labels).sum()
    intra_year_disaster_deaths = intra_year_disaster_deaths.loc[:, "Total_Deaths"]

    deaths_per_year: pd.Series
    if include_zero:
        custom_index_values = [list(df[label].unique()) for label in custom_index]
        complete_year_range = range(min_start_year, max_start_year + 1)
        result_index = pd.MultiIndex.from_product(iterables=[complete_year_range]+custom_index_values, names=index_labels).dropna()
        empty_series = pd.Series(data=0, name="Total_Deaths", index=result_index)
        deaths_per_year = empty_series.add(intra_year_disaster_deaths, fill_value=0).astype("float32")  # will be filled with perennial disaster_deaths later

    else:
        deaths_per_year = intra_year_disaster_deaths

    def _flatten_death_distr(year_df: pd.DataFrame) -> None:
        nonlocal deaths_per_year
        curr_year = year_df['Year'].iloc[0]
        year_df.dropna(axis=0, inplace=True)
        if len(year_df) == 0:
            return

        yearly_deaths_by_duration: pd.DataFrame = year_df[["Duration_CalendarYears", "Yearly_Disaster_Deaths"] + custom_index]\
            .groupby(["Duration_CalendarYears"] + custom_index).sum()

        duration_df: pd.DataFrame
        for Duration_CalendarYears, duration_df in yearly_deaths_by_duration.groupby(level="Duration_CalendarYears"):
            duration_df.reset_index(inplace=True)
            duration_df.drop("Duration_CalendarYears", axis=1, inplace=True)

            year_range = np.arange(start=curr_year, stop=curr_year + Duration_CalendarYears + 1)
            year_idx_array: np.ndarray = np.asarray(year_range).repeat(len(duration_df))

            add_df: pd.DataFrame = pd.concat([duration_df] * len(year_range), ignore_index=True)
            add_df["Year"] = year_idx_array
            add_df.set_index(index_labels, inplace=True)
            add_series: pd.Series = add_df["Yearly_Disaster_Deaths"]
            add_series.rename("Total_Deaths", inplace=True)

            deaths_per_year = deaths_per_year.add(add_series, fill_value=0)

    perennial_disasters.groupby("Year", as_index=False).apply(_flatten_death_distr)

    if custom_index == alternate_cust_idx:
        deaths_per_year.reset_index(level=custom_index, inplace=True, drop=True)
    if not include_zero:
        deaths_per_year = deaths_per_year[deaths_per_year!=0]

    return deaths_per_year
