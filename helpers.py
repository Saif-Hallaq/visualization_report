import pandas as pd
import plotly.express as px
import streamlit as st
import io

@st.cache_data
def load_data(file):
    return pd.read_excel(file) if file else None

def create_pie_chart(data, names_col, values_col, title, color_map=None, legend_title_text=None):
    fig = px.pie(
        data,
        names=names_col,
        values=values_col,
        title=title,
        color_discrete_map=color_map
    )
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=100),
        showlegend=True,
        legend_title=dict(
            text=legend_title_text or names_col,  # 👈 custom title if provided
            font=dict(size=16)
        ),
        legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)")
    )
    return fig

def filter_by_timeframe(df, date_column):
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        min_date, max_date = df[date_column].min(), df[date_column].max()

        if pd.notna(min_date) and pd.notna(max_date) and min_date != max_date:
            selected_timeframe = st.slider(
                "Zeitrahmen auswählen",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                format="YYYY-MM-DD"
            )
            df = df[(df[date_column] >= selected_timeframe[0]) & (df[date_column] <= selected_timeframe[1])]
        else:
            st.warning("⚠ Nicht genügend Datenvariationen für die Auswahl eines Zeitrahmens")
    return df

def render_tab_overview(df):
    st.header("📊 Übersicht")
    st.metric(label="Total Records", value=df.shape[0])

    for prefix, label in [("Suchagent", "🔍 Search Agents"), ("Tag", "🏷️ Tags")]:
        columns = [col for col in df.columns if col.startswith(prefix)]
        values = sorted(set(str(row[col]).strip() for col in columns for _, row in df.iterrows()
                            if pd.notna(row[col]) and str(row[col]).strip().lower() != 'nan'))
        if columns:
            st.write(label)
            st.write(", ".join(values) if values else f"No {label} found.")
        else:
            st.warning(f"⚠ No {prefix} columns found in the uploaded file.")

def render_tab_analysis(df, mode):
    st.subheader(f"📊 Auswertungen nach {mode}s")
    columns = [col for col in df.columns if col.startswith(mode)]
    df = filter_by_timeframe(df, "Veröffentlichungsdatum")

    all_values = sorted(set(str(v) for col in columns for v in df[col].dropna().unique()))
    selected = st.multiselect(f"{mode}s auswählen", all_values, placeholder=mode)
    selected = selected or all_values

    mask = df[columns].apply(lambda row: any(str(val) in selected for val in row if pd.notna(val)), axis=1)
    df_filtered = df[mask]

    freq = st.selectbox("Zeitintervall auswählen", ["Täglich", "Monatlich", "Quartalsweise", "Jährlich"])
    freq_map = {"Täglich": "D", "Monatlich": "ME", "Quartalsweise": "Q", "Jährlich": "Y"}
    date_freq = freq_map[freq]

    df_melted = df_filtered.melt(id_vars=['Veröffentlichungsdatum'], value_vars=columns,
                                  var_name="Column", value_name="Value").dropna(subset=['Value'])
    df_melted = df_melted[df_melted['Value'].isin(selected)]
    df_melted['Veröffentlichungsdatum'] = pd.to_datetime(df_melted['Veröffentlichungsdatum'])

    counts = df_melted.groupby([pd.Grouper(key='Veröffentlichungsdatum', freq=date_freq), 'Value']).size().reset_index(name='Count')

    color_palette = px.colors.qualitative.Plotly
    color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(selected)}

    bar_chart = px.bar(counts, x="Veröffentlichungsdatum", y="Count", color="Value",
                       labels={"Count": "Treffer", "Value": mode},
                       title=f"{mode}-Trend im Zeitverlauf ({freq})", barmode="group",
                       color_discrete_map=color_map)
    bar_chart.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend_title_text=mode,  # 👈 This changes the legend title
                        legend=dict(font=dict(size=13))
                        
                  )
    st.plotly_chart(bar_chart)

    line_chart = px.line(counts, x="Veröffentlichungsdatum", y="Count", color="Value", labels={"Count": "Treffer", "Value": mode},
                         color_discrete_map=color_map)
    line_chart.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend_title_text=mode,  # 👈 This changes the legend title
                        legend=dict(font=dict(size=13))
                  
                  )
    st.plotly_chart(line_chart)

    st.dataframe(counts.pivot(index="Veröffentlichungsdatum", columns="Value", values="Count").fillna(0))

    pie_data = counts.groupby('Value')['Count'].sum().reset_index()
    pie_chart = create_pie_chart(
        pie_data,
        'Value',
        'Count',
        f"{mode}-Verteilung",
        color_map,
        legend_title_text=mode  # this is the correct keyword argument
        )
    st.plotly_chart(pie_chart, use_container_width=True)
    st.dataframe(pie_data.rename(columns={"Value": mode, "Count": "Treffer"}).set_index(mode))
    render_mediengattung_analysis(df_filtered, freq=date_freq)
    render_bewertung_analysis(df_filtered, freq=date_freq)

def render_tab_sources(df):
    st.subheader("📊 Auswertungen nach Quellen")
    df.columns = df.columns.str.strip()
    df = filter_by_timeframe(df, "Veröffentlichungsdatum")
    if "Quelle" not in df.columns:
        st.warning("⚠ Kein 'Quelle'-Feld gefunden.")
        return

    # Combined filtering
    def collect_values(prefix):
        cols = [col for col in df.columns if col.startswith(prefix)]
        return sorted(set(val for col in cols for val in df[col].dropna().unique() if val != "")), cols

    selected_sa, sa_cols = collect_values("Suchagent")
    selected_st, st_cols = collect_values("Smart-Tag")
    selected_tag, tag_cols = collect_values("Tag")

    agent_filter = st.multiselect("Suchagenten auswählen", selected_sa, placeholder="SuchAgenten")
    smart_filter = st.multiselect("Smart-Tag auswählen", selected_st, placeholder="SmartTags")
    tag_filter = st.multiselect("Tags auswählen", selected_tag, placeholder="Tags")

    cond = pd.Series([True] * len(df))
    if agent_filter:
        cond &= df[sa_cols].apply(lambda row: any(val in agent_filter for val in row), axis=1)
    if smart_filter:
        cond &= df[st_cols].apply(lambda row: any(val in smart_filter for val in row), axis=1)
    if tag_filter:
        cond &= df[tag_cols].apply(lambda row: any(val in tag_filter for val in row), axis=1)

    df = df[cond & df["Quelle"].notna() & (df["Quelle"] != "")]
    if df.empty:
        st.warning("⚠ No data available for the selected filters.")
        return

    quelle_counts = df.groupby("Quelle").size().reset_index(name="Total Treffer")
    top_10 = quelle_counts.nlargest(10, "Total Treffer")

    fig = px.bar(top_10, x="Total Treffer", y="Quelle", title="Top 10 Quellen", color="Quelle", text_auto=True)
    fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
    st.plotly_chart(fig)
    st.dataframe(top_10.set_index("Quelle"))

def render_tab_datenblatt(df):
    st.subheader(" Datenblatt")
    tag_columns = [col for col in df.columns if col.startswith("Tag")]
    smart_tag_columns = [col for col in df.columns if col.startswith("Smart-Tag")]
    search_agent_columns = [col for col in df.columns if col.startswith("Suchagent")]

    df_transformed = df.copy()
    for col in tag_columns + smart_tag_columns + search_agent_columns:
        valid_vals = df_transformed[col].dropna()
        if not valid_vals.empty:
            df_transformed.rename(columns={col: valid_vals.iloc[0]}, inplace=True)

    st.dataframe(df_transformed)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_transformed.to_excel(writer, index=False, sheet_name='Datenblatt')
    output.seek(0)

    st.download_button(
        label="Download Processed Excel",
        data=output,
        file_name="transformed_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def render_mediengattung_analysis(df, freq):
    if "Mediengattung" not in df.columns:
        return

    media_counts = df.groupby([
        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
        'Mediengattung'
    ]).size().reset_index(name='Count')

    if not media_counts.empty:
        media_colors = {m: px.colors.qualitative.Plotly[i % 20] for i, m in enumerate(media_counts['Mediengattung'].unique())}

        bar_fig = px.bar(
            media_counts,
            x="Veröffentlichungsdatum",
            y="Count",
            title="Mediengattungen",
            color="Mediengattung",
            color_discrete_map=media_colors,
            barmode="group",
            labels={"Count": "Treffer"}
        )
        bar_fig.update_layout(
            legend_title=dict(font=dict(size=16)),
            legend=dict(font=dict(size=13))   )     
        st.plotly_chart(bar_fig)

        line_fig = px.line(
            media_counts,
            x="Veröffentlichungsdatum",
            y="Count",
            color="Mediengattung",
            color_discrete_map=media_colors,
            labels={"Count": "Treffer"}
        )

        line_fig.update_layout(
            legend_title=dict(font=dict(size=16)),
            legend=dict(font=dict(size=13))   ) 
        st.plotly_chart(line_fig)

        st.dataframe(media_counts.pivot(index="Veröffentlichungsdatum", columns="Mediengattung", values="Count").fillna(0))

        media_dist = df['Mediengattung'].value_counts().reset_index()
        media_dist.columns = ['Mediengattung', 'count']
        media_fig = create_pie_chart(media_dist, 'Mediengattung', 'count', 'Mediengattungen-Verteilung', media_colors)
        st.plotly_chart(media_fig, use_container_width=True)
        st.dataframe(media_dist.rename(columns={"Mediengattung": "Media Type", "count": "Treffer"}).set_index("Media Type"))

def render_bewertung_analysis(df, freq):
    if "Bewertung" not in df.columns:
        return

    df_ratings = df[df["Bewertung"].notna() & (df["Bewertung"] != "")]
    if df_ratings.empty:
        return

    rating_data = df_ratings.groupby([
        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
        'Bewertung'
    ]).size().reset_index(name='Count')

    rating_colors = {"Positiv": "green", "Negativ": "red", "Neutral": "blue"}

    bar_fig = px.bar(
        rating_data,
        x="Veröffentlichungsdatum",
        y="Count",
        color="Bewertung",
        title="Bewertung",
        color_discrete_map=rating_colors,
        barmode="group",
        labels={"Count": "Treffer"}
    )

    bar_fig.update_layout(
            legend_title=dict(font=dict(size=16)),
            legend=dict(font=dict(size=13))   ) 
    st.plotly_chart(bar_fig)

    line_fig = px.line(
        rating_data,
        x="Veröffentlichungsdatum",
        y="Count",
        color="Bewertung",
        color_discrete_map=rating_colors,
        labels={"Count": "Treffer"}
    )

    line_fig.update_layout(
            legend_title=dict(font=dict(size=16)),
            legend=dict(font=dict(size=13))   ) 
    st.plotly_chart(line_fig)

    st.dataframe(rating_data.pivot(index="Veröffentlichungsdatum", columns="Bewertung", values="Count").fillna(0))

    rating_counts = df_ratings['Bewertung'].value_counts().reset_index()
    rating_counts.columns = ['Bewertung', 'Treffer']
    rating_pie_fig = create_pie_chart(rating_counts, 'Bewertung', 'Treffer', 'Bewertungsverteilung', rating_colors)
    st.plotly_chart(rating_pie_fig, use_container_width=True)
    st.dataframe(rating_counts.set_index("Bewertung"))