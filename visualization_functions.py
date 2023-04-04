import altair as alt
import pandas as pd


def plot_corr(df: pd.DataFrame) -> alt.Chart:
    """
    This function plots a correlation matrix for a dataframe
    """
    corr_matrix = df.corr()

    # Convert the correlation matrix into a tidy format
    corr_matrix = corr_matrix.stack().reset_index()
    corr_matrix.columns = ["feature_1", "feature_2", "correlation"]

    # Create the feature corrplot
    base = alt.Chart(corr_matrix).encode(
        x="feature_1:N",
        y="feature_2:N"
    )

    # Add the rectangles
    rects = base.mark_rect().encode(
        color=alt.Color("correlation:Q")
    )

    # Add the text labels
    labels = base.mark_text(baseline="middle").encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            alt.datum.correlation > 0.5,
            alt.value("white"),
            alt.value("black")
        )
    )

    # Show only half of the matrix
    corr_plot = (rects + labels).properties(
        width=400,
        height=400
    ).transform_filter(
        alt.datum.feature_1 < alt.datum.feature_2
    )
    return corr_plot