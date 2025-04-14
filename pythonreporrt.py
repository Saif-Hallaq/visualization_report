import time  
import streamlit as st
import pandas as pd
import plotly.express as px
import io



# Page title
st.set_page_config(page_title="📊 Excel Data Dashboard", layout="wide")


# Sidebar navigation
tab = st.sidebar.radio(
    "",
    [
        "Übersicht",
        "Auswertungen nach Suchagenten",
        "Auswertungen nach Tags",
        "Auswertungen nach Smart-Tags",
        "Auswertungen nach Quellen",
        "Datenblatt"
    ]
)

with st.sidebar:
    uploaded_file = st.file_uploader("Datei hierhin ziehen", type=["xls", "xlsx"])



@st.cache_data
def load_data(file):
    """Load the Excel file once and cache it."""
    return pd.read_excel(file) if file else None

# Store DataFrame in session state to persist across tabs
if "df" not in st.session_state or uploaded_file:
    st.session_state.df = load_data(uploaded_file) if uploaded_file else None

df = st.session_state.df

def create_pie_chart(data, names_col, values_col, title, color_map=None):
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
        text=names_col,
        font=dict(size=16)
    ),
        legend=dict(
            font=dict(
                size=12,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(0,0,0,0)",
        )
    )

    return fig

# Function to filter data by timeframe
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

            # Apply timeframe filtering
            df = df[(df[date_column] >= selected_timeframe[0]) & 
                    (df[date_column] <= selected_timeframe[1])]
        else:
            st.warning("⚠ Not enough data variation to select a timeframe.")
    
    return df
# Ensure the file is loaded before proceeding
if uploaded_file:
    df = load_data(uploaded_file)

    # Data processing starts here
    search_agent_columns = [col for col in df.columns if col.startswith("Suchagent")]

    if not search_agent_columns:
        st.warning("No 'Suchagent' columns found.")
    else:
        # Melt DataFrame to restructure
        df_melted = df.melt(id_vars=['Veröffentlichungsdatum'], value_vars=search_agent_columns, 
                            var_name="Suchagent Column", value_name="Suchagent")
        agent_counts_over_time = df_melted.groupby(['Veröffentlichungsdatum', 'Suchagent']).size().reset_index(name='Count')

        # Convert date and aggregate by month
        df['Veröffentlichungsdatum'] = pd.to_datetime(df['Veröffentlichungsdatum'], errors='coerce')
        agent_counts_over_time = agent_counts_over_time.groupby(
            [pd.Grouper(key="Veröffentlichungsdatum", freq="M"), 'Suchagent']
        )['Count'].sum().reset_index()

        # Display success message
        st.sidebar.success("Datei erfolgreich hochgeladen! ✅")
     

    # ---------------------- #
    # Tab: Data Overview
    # ---------------------- # 
    if tab == "Übersicht":
        
        st.header("📊 Übersicht")

        # Total number of records
        total_records = df.shape[0]
        st.metric(label="Total Records", value=total_records)

        # Find the columns: Suchagent1, Suchagent2, Suchagent3, Suchagent4, Suchagent5
        search_agent_columns = [col for col in df.columns if col.startswith("Suchagent")]
        if search_agent_columns:
            # For each row, collect all search agents from the relevant columns
            # Remove duplicates across columns and exclude `nan` values
            search_agents_set = set()
            for index, row in df.iterrows():
                for col in search_agent_columns:
                    agent = str(row[col]).strip()
                    if agent and agent.lower() != 'nan':  # Ignore `nan` values
                        search_agents_set.add(agent)

            # Format the search agents list
            formatted_agents = ", ".join(sorted(search_agents_set))

            st.write("🔍 Search Agents")
            st.write(formatted_agents if formatted_agents else "No search agents found.")
        
        # ---------------------- #
        # Tags Overview
        # ---------------------- #
        # Collect all columns that start with 'Smart-Tag'
        tag_columns_existing = [col for col in df.columns if col.startswith("Tag")]

        if tag_columns_existing:
            # Collect all tags across the relevant columns
            tags_set = set()
            for index, row in df.iterrows():
                for col in tag_columns_existing:
                    tag = str(row[col]).strip()
                    if tag and tag.lower() != 'nan':  # Ignore `nan` values
                        tags_set.add(tag)

            # Format the tags list
            formatted_tags = ", ".join(sorted(tags_set))

            st.write("🏷️ Tags")
            st.write(formatted_tags if formatted_tags else "No tags found.")
        else:
            st.warning("⚠ No Tag or Smart-Tag columns found in the uploaded file.")

    # ---------------------- #
    # SuchagentenZeitreihe Tab
    # ---------------------- #

    if tab == "Auswertungen nach Suchagenten":
       st.subheader("📊 Auswertungen nach Suchagenten")
       if df is None:
        st.warning("⚠️ No data loaded. Please upload an Excel file.")
        st.stop()

        # 🔹 Ensure df is fully available before rendering tables
        if "df_ready" not in st.session_state:
            st.session_state.df_ready = False

        if not st.session_state.df_ready:
            time.sleep(1)  # Allow Streamlit to catch up
            st.session_state.df_ready = True
            st.experimental_rerun()  # 🔄 Force page refresh to ensure stable rendering

       if search_agent_columns:
            # Apply filters
            df = filter_by_timeframe(df, "Veröffentlichungsdatum")
            
            # Get unique agents
            all_agents = sorted(set(str(agent) for col in search_agent_columns 
                                for agent in df[col].dropna().unique()))
            
            # Agent selection
            selected_agents = st.multiselect(
                "Suchagenten auswählen",
                all_agents
            )
            
            if not selected_agents:
                selected_agents = all_agents 
            
            # Filter data for selected agents
            mask = df[search_agent_columns].apply(
                lambda row: any(str(agent) in selected_agents for agent in row if pd.notna(agent)),
                axis=1
            )
            df_filtered = df[mask]

            # Time granularity
            time_granularity = st.selectbox(
                "Zeitintervall auswählen",
                ["Täglich", "Monatlich", "Vierteljährlich", "Jährlich"]
            )
            freq = {"Täglich": "D", "Monatlich": "ME", 
                "Vierteljährlich": "Q", "Jährlich": "Y"}[time_granularity]

            # 1. Search Agent Analysis
            st.subheader("Suchagenten-Analyse")
            df_melted = df_filtered.melt(
                id_vars=['Veröffentlichungsdatum'], 
                value_vars=search_agent_columns,
                var_name="Column", 
                value_name="Suchagent"
            ).dropna()
            
            df_melted = df_melted[df_melted['Suchagent'].isin(selected_agents)]
            df_melted['Veröffentlichungsdatum'] = pd.to_datetime(df_melted['Veröffentlichungsdatum'])
                
            agent_data = df_melted.groupby([
                pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                'Suchagent'
            ]).size().reset_index(name='Count')

            if not agent_data.empty:
                # Visualizations
                colors = {a: px.colors.qualitative.Prism[i%len(px.colors.qualitative.Prism)] 
                        for i,a in enumerate(selected_agents)}
                
                # Bar Chart
                fig = px.bar(
                    agent_data,
                    x="Veröffentlichungsdatum",
                    y="Count",
                    color="Suchagent",
                    color_discrete_map=colors,
                    barmode="group"
                )

                # Rename the Y-axis label
                fig.update_layout(
                    yaxis_title="Treffer",
                    legend_title=dict(
                        text="Suchagent",
                        font=dict(size=16)
                    )
                )

                # Show the chart
                st.plotly_chart(fig)

                
                # Line Chart
                fig = px.line(
                    agent_data,
                    x="Veröffentlichungsdatum",
                    y="Count",
                    color="Suchagent",
                    color_discrete_map=colors
                )

                fig.update_layout(
                    yaxis_title="Treffer",
                    legend_title=dict(
                    text="Suchagent",
                    font=dict(size=16)
                )
                )

                st.plotly_chart(fig)
               

                # Time-based Table
                st.dataframe(agent_data.pivot(index="Veröffentlichungsdatum", 
                                            columns="Suchagent", 
                                            values="Count").fillna(0))
                
                # Pie Chart
                # Filter and prepare agent count data
                agent_counts_total = (
                    df_melted[df_melted['Suchagent'].isin(selected_agents)]
                    .groupby('Suchagent')
                    .size()
                    .reset_index(name='Treffer')
                )

                # Clean any potential NaN or invalid data
                agent_counts_total = agent_counts_total.dropna()

                # Create the pie chart using the reusable function
                suchagent_fig = create_pie_chart(
                    agent_counts_total,
                    names_col='Suchagent',
                    values_col='Treffer',
                    title='Suchagent-Verteilung',
                    color_map=colors  # your predefined color dict
                )

                # Show pie chart in Streamlit
                st.plotly_chart(suchagent_fig, use_container_width=True)

                # Show the summary table
                st.dataframe(agent_counts_total.set_index("Suchagent"))

            # 2. Media Type Analysis
            if "Mediengattung" in df_filtered.columns:
                st.subheader("Mediengattungen")
                media_data = df_filtered.groupby([
                    pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                    'Mediengattung'
                ]).size().reset_index(name='Count')

                if not media_data.empty:
                    # Visualizations
                    media_types = media_data['Mediengattung'].unique()

                    # Replace 'Radio Tv' with 'Radio and TV'
                    media_types = [media_type.replace('Radio Tv', 'Radio and TV') for media_type in media_types]

                   
                    media_colors = {m: px.colors.qualitative.Plotly[i%len(px.colors.qualitative.Plotly)]
                                for i,m in enumerate(media_types)}
                    
                    # Bar Chart
                    fig = px.bar(
                        media_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Mediengattung",
                        color_discrete_map=media_colors,
                        barmode="group"
                    )

                    fig.update_layout(
                        yaxis_title="Treffer",
                        legend_title=dict(
                            text="Mediengattung",  # Optional, will auto-use column name
                            font=dict(size=16)     # Adjust font size as needed
                        )
                    )

                    st.plotly_chart(fig)
                    # Line Chart
                    fig = px.line(
                        media_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Mediengattung",
                        color_discrete_map=media_colors
                    )

                    fig.update_layout(
                        yaxis_title="Treffer",
                        legend_title=dict(
                            text="Mediengattung",  # Optional, will auto-use column name
                            font=dict(size=16)     # Adjust font size as needed
                        )
                    )

                    st.plotly_chart(fig)


                    # Time-based Table
                    st.dataframe(media_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Mediengattung",
                                                values="Count").fillna(0))
                    
                    # Pie Chart
                    
                    media_dist = df_filtered['Mediengattung'].value_counts().reset_index()
                    media_dist.columns = ['Mediengattung', 'count']  # Rename columns for clarity

                    media_gattung_fig = create_pie_chart(
                        media_dist,
                        names_col='Mediengattung',
                        values_col='count',
                        title="Mediengattungen-Verteilung",
                        color_map=media_colors  # your predefined color dict
                    )

                    # Show the pie chart in Streamlit
                    st.plotly_chart(media_gattung_fig, use_container_width=True)
                    
                    # Summary Table
                    st.dataframe(
                        media_dist.rename(columns={"Mediengattung": "Media Type", "count": "Treffer"}).set_index("Media Type")
                    )

            # 3. Rating Analysis
            if "Bewertung" in df_filtered.columns:
                st.subheader("Bewertungen")
                df_ratings = df_filtered[
                    df_filtered["Bewertung"].notna() & 
                    (df_filtered["Bewertung"] != "")
                ]
                
                if not df_ratings.empty:
                    rating_data = df_ratings.groupby([
                        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                        'Bewertung'
                    ]).size().reset_index(name='Count')

                    # Visualizations
                    rating_colors = {"Positiv": "green", "Negativ": "red", "Neutral": "blue"}
                    
                    # Bar Chart
                    fig = px.bar(
                        rating_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Bewertung",
                        color_discrete_map=rating_colors,
                        barmode="group"
                    )

                    
                    fig.update_layout(
                        yaxis_title="Treffer",
                        legend_title=dict(
                            text="Bewertung",
                            font=dict(size=16)  # Change size as needed
                        ))

                    st.plotly_chart(fig)

                    # Line Chart
                    fig = px.line(
                        rating_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Bewertung",
                        color_discrete_map=rating_colors
                    )

                    fig.update_layout(
                        yaxis_title="Treffer",
                        legend_title=dict(
                            text="Bewertung",
                            font=dict(size=16)  # Change size as needed
                        )
                    )

                    st.plotly_chart(fig)


                    # Time-based Table
                    st.dataframe(rating_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Bewertung",
                                                values="Count").fillna(0))
                    
                    # Pie Chart (EXACTLY AS IN ORIGINAL)
                    rating_counts = df_ratings['Bewertung'].value_counts().reset_index()
                    rating_counts.columns = ['Bewertung', 'count']  # Rename columns for clarity

                    # Create the pie chart using the reusable function
                    bewertung_fig = create_pie_chart(
                        rating_counts,
                        names_col='Bewertung',
                        values_col='count',
                        title="Bewertungen-verteilung",
                        color_map=rating_colors  # your predefined color dict
                    )

                    # Show the pie chart in Streamlit
                    st.plotly_chart(bewertung_fig, use_container_width=True)
                    
                    # Summary Table
                    rating_totals = df_ratings['Bewertung'].value_counts().reset_index()
                    rating_totals.columns = ["Bewertung", "Gesamtanzahl"]
                    st.dataframe(rating_totals.set_index("Bewertung"))
                    
      
      
        # 📌 TagsZeitreihe
    # ---------------------- #
    if tab == "Auswertungen nach Tags":
       st.subheader("📊 Auswertungen nach Tags")  # Changed title
       if df is None:
        st.warning("⚠️ No data loaded. Please upload an Excel file.")
        st.stop()

        # 🔹 Ensure df is fully available before rendering tables
        if "df_ready" not in st.session_state:
            st.session_state.df_ready = False

        if not st.session_state.df_ready:
            time.sleep(1)  # Allow Streamlit to catch up
            st.session_state.df_ready = True
            st.experimental_rerun()  # 🔄 Force page refresh to ensure stable rendering



       # Identify tag columns (both standard and smart tags)
       tag_columns = [col for col in df.columns if col.startswith(("Tag"))]
       
       if tag_columns:
            # Apply timeframe filter
            df = filter_by_timeframe(df, "Veröffentlichungsdatum")
            
            # Get all unique tags from the data
            all_tags = sorted(set(str(tag) for col in tag_columns 
                            for tag in df[col].dropna().unique()))
            
            # Tag selection (empty by default)
            selected_tags = st.multiselect(
                "Tags auswählen",  # Changed label
                all_tags
            )
            
            # If no tags selected, show nothing
            if not selected_tags:
                selected_tags = all_tags
            
            # Filter the DataFrame to only include rows with selected tags
            mask = df[tag_columns].apply(
                lambda row: any(str(tag) in selected_tags for tag in row if pd.notna(tag)),
                axis=1
            )
            df_filtered = df[mask]

            # Time granularity selection
            time_granularity = st.selectbox(
                "Zeitintervall auswählen",
                ["Täglich", "Monatlich", "Vierteljährlich", "Jährlich"]
            )
            
            # Apply time grouping
            freq = {
                "Täglich": "D",
                "Monatlich": "ME",
                "Vierteljährlich": "Q",
                "Jährlich": "Y"
            }[time_granularity]

            # --------------------------
            # TAG VISUALIZATIONS 
            # --------------------------
            
            # Melt tag data
            df_melted = df_filtered.melt(
                id_vars=['Veröffentlichungsdatum'], 
                value_vars=tag_columns,
                var_name="Tag Column", 
                value_name="Tag"
            ).dropna(subset=['Tag'])
            
            # Filter to selected tags
            df_melted = df_melted[df_melted['Tag'].isin(selected_tags)]
            df_melted['Veröffentlichungsdatum'] = pd.to_datetime(df_melted['Veröffentlichungsdatum'])
                
            tag_counts = df_melted.groupby([
                pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                'Tag'
            ]).size().reset_index(name='Count')

            if not tag_counts.empty:
                # Color mapping for tags
                color_palette = px.colors.qualitative.Prism
                color_map = {tag: color_palette[i % len(color_palette)] 
                            for i, tag in enumerate(selected_tags)}

                # 1. Bar Chart
                bar_chart = px.bar(
                    tag_counts, 
                    x="Veröffentlichungsdatum", 
                    y="Count", 
                    color="Tag",
                    title=f"Tag Trend im Zeitverlauf ({time_granularity})",  # Changed title
                    labels={     "Count": "Treffer", "Tag": "Tag"},  # Changed labels
                    barmode="group",
                    text_auto=True,
                    color_discrete_map=color_map
                )
                bar_chart.update_layout(
                    legend_title=dict(
                        font=dict(size=16)
                    )
                )
                st.plotly_chart(bar_chart)

                # 2. Line Chart
                line_chart = px.line(
                    tag_counts, 
                    x="Veröffentlichungsdatum", 
                    y="Count", 
                    color="Tag",
                    labels={ "Count": "Treffer", "Tag": "Tag"},  # Changed labels

                    color_discrete_map=color_map
                )

                line_chart.update_layout(
                    legend_title=dict(
                        font=dict(size=16)
                    )
                )
                st.plotly_chart(line_chart)
            
                # 3. Pivot Table
                pivot_table = tag_counts.pivot(
                    index="Veröffentlichungsdatum",
                    columns="Tag",
                    values="Count"
                ).fillna(0)
                st.dataframe(pivot_table)
                
                # Create the Pie chart using Plotly Express
                pie_data = tag_counts.groupby('Tag')['Count'].sum().reset_index()
                fig1 = create_pie_chart(pie_data, 'Tag', 'Count', "Tag-Verteilung", color_map)
                st.plotly_chart(fig1, use_container_width=True)

            

                st.dataframe(
                    pie_data.rename(columns={"Tag": "Tag", "Count": "Treffer"}).set_index("Tag")
                )
               


            # --------------------------
            # MEDIA TYPE ANALYSIS 
            # --------------------------
            if "Mediengattung" in df_filtered.columns:
                st.subheader("Mediengattungen")
                media_counts = df_filtered.groupby([
                    pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                    'Mediengattung'
                ]).size().reset_index(name='Count')

                if not media_counts.empty:
                    media_colors = {m: px.colors.qualitative.Plotly[i%20] for i,m in enumerate(media_counts['Mediengattung'].unique())}
                    
                    # Bar chart
                bar_fig = px.bar(
                    media_counts,
                    x="Veröffentlichungsdatum",
                    y="Count",
                    color="Mediengattung",
                    color_discrete_map=media_colors,
                    labels={ "Count": "Treffer"},  # Changed labels

                    barmode="group"
                )

                bar_fig.update_layout(
                    legend_title=dict(font=dict(size=16)),
                    legend=dict(font=dict(size=13))  # optional: change legend item size too
                )

                st.plotly_chart(bar_fig)

                # Line chart
                line_fig = px.line(
                    media_counts,
                    x="Veröffentlichungsdatum",
                    y="Count",
                    color="Mediengattung",
                    labels={ "Count": "Treffer"},  # Changed labels

                    color_discrete_map=media_colors
                )

                line_fig.update_layout(
                    legend_title=dict(font=dict(size=16)),
                    legend=dict(font=dict(size=13))  # optional
                )

                st.plotly_chart(line_fig)

                st.dataframe(media_counts.pivot(index="Veröffentlichungsdatum",
                                                                columns="Mediengattung",
                                                                values="Count").fillna(0))
                    
                media_dist = df_filtered['Mediengattung'].value_counts().reset_index()
                media_dist.columns = ['Mediengattung', 'count']
                media_fig = create_pie_chart(
                    media_dist,
                    names_col='Mediengattung',
                    values_col='count',
                    title='Mediengattungen-Verteilung'
                )

                st.plotly_chart(media_fig, use_container_width=True)

                    
                #Table
                st.dataframe(
                    media_dist.rename(columns={"Mediengattung": "Media Type", "count": "Treffer"}).set_index("Media Type")
                )
            # --------------------------
            # RATING ANALYSIS 
            # --------------------------
            if "Bewertung" in df_filtered.columns:
                st.subheader("Bewertungen")
                df_ratings = df_filtered[
                    df_filtered["Bewertung"].notna() & 
                    (df_filtered["Bewertung"] != "") 
                ]
                
                if not df_ratings.empty:
                    rating_data = df_ratings.groupby([
                        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                        'Bewertung'
                    ]).size().reset_index(name='Count')

                    rating_colors = {"Positiv": "green", "Negativ": "red", "Neutral": "blue"}

                    # Bar chart with updated legend styling
                    bar_fig = px.bar(
                        rating_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Bewertung",
                        color_discrete_map=rating_colors,
                        labels={ "Count": "Treffer"},
                        barmode="group"
                    )

                    bar_fig.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend=dict(font=dict(size=13))  # Optional: adjust legend item font size
                    )

                    st.plotly_chart(bar_fig)

                    # Line chart with updated legend styling
                    line_fig = px.line(
                        rating_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        labels={ "Count": "Treffer"},  # Changed labels

                        color="Bewertung",
                        color_discrete_map=rating_colors
                    )

                    line_fig.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend=dict(font=dict(size=13))
                    )

                    st.plotly_chart(line_fig)

                    
                    st.dataframe(rating_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Bewertung",
                                                values="Count").fillna(0))
                    
                    # Prepare pie chart data
                    rating_counts = df_ratings['Bewertung'].value_counts().reset_index()
                    rating_counts.columns = ['Bewertung', 'Treffer']

                    # Create and show the pie chart
                    rating_pie_fig = create_pie_chart(
                        rating_counts,
                        names_col='Bewertung',
                        values_col='Treffer',
                        title='Bewertungen-Verteilung',
                        color_map=rating_colors
                    )

                    st.plotly_chart(rating_pie_fig, use_container_width=True)

                    # Show data table
                    st.dataframe(rating_counts.set_index("Bewertung"))


        # 📌 SmartTagsZeitreihe
    # ---------------------- #
    if tab == "Auswertungen nach Smart-Tags":
       st.subheader("📊 Auswertungen nach Smart-Tags")  

       # Identify tag columns (both standard and smart tags)
       tag_columns = [col for col in df.columns if col.startswith("Smart-Tag")]      
      
       if tag_columns:
            # Apply timeframe filter
            df = filter_by_timeframe(df, "Veröffentlichungsdatum")
            
            # Get all unique tags from the data
            all_tags = sorted(set(str(tag) for col in tag_columns 
                            for tag in df[col].dropna().unique()))
            
            # Tag selection (empty by default)
            selected_tags = st.multiselect(
                "Smart-Tags auswählen",  # Changed label
                all_tags
            )
            
            # If no tags selected, show nothing
            if not selected_tags:
                selected_tags = all_tags
            
            # Filter the DataFrame to only include rows with selected tags
            mask = df[tag_columns].apply(
                lambda row: any(str(tag) in selected_tags for tag in row if pd.notna(tag)),
                axis=1
            )
            df_filtered = df[mask]

            # Time granularity selection
            time_granularity = st.selectbox(
                "Zeitintervall auswählen",
                ["Täglich", "Monatlich", "Vierteljährlich", "Jährlich"]
            )
            
            # Apply time grouping
            freq = {
                "Täglich": "D",
                "Monatlich": "ME",
                "Vierteljährlich": "Q",
                "Jährlich": "Y"
            }[time_granularity]

            # --------------------------
            # TAG VISUALIZATIONS 
            # --------------------------
            
            # Melt tag data
            df_melted = df_filtered.melt(
                id_vars=['Veröffentlichungsdatum'], 
                value_vars=tag_columns,
                var_name="Tag Column", 
                value_name="Tag"
            ).dropna(subset=['Tag'])
            
            # Filter to selected tags
            df_melted = df_melted[df_melted['Tag'].isin(selected_tags)]
            df_melted['Veröffentlichungsdatum'] = pd.to_datetime(df_melted['Veröffentlichungsdatum'])
                
            tag_counts = df_melted.groupby([
                pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                'Tag'
            ]).size().reset_index(name='Count')

            if not tag_counts.empty:
                # Color mapping for tags
                color_palette = px.colors.qualitative.Prism
                color_map = {tag: color_palette[i % len(color_palette)] 
                            for i, tag in enumerate(selected_tags)}

                # 1. Bar Chart
                bar_chart = px.bar(
                    tag_counts, 
                    x="Veröffentlichungsdatum", 
                    y="Count", 
                    color="Tag",
                    title=f"Smart Tag Trend im Zeitverlauf ({time_granularity})",  # Changed title
                    labels={ "Count": "Treffer", "Tag": "SmartTag"},  # Changed labels
                    barmode="group",
                    text_auto=True,
                    color_discrete_map=color_map
                )
                bar_chart.update_layout(
                                        legend_title=dict(font=dict(size=16)),
                                        legend=dict(font=dict(size=13))  # Optional: change legend item font size
                                    )
                st.plotly_chart(bar_chart)

                # 2. Line Chart
                line_chart = px.line(
                    tag_counts, 
                    x="Veröffentlichungsdatum", 
                    y="Count", 
                    labels={"Count": "Treffer", "Tag": "SmartTag"},
                    color="Tag",
                    color_discrete_map=color_map
                )

                line_chart.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend=dict(font=dict(size=13))  # Optional: change legend item font size
                    )
                st.plotly_chart(line_chart)
            
                # 3. Pivot Table
                pivot_table = tag_counts.pivot(
                    index="Veröffentlichungsdatum",
                    columns="Tag",
                    values="Count"
                ).fillna(0)
                st.dataframe(pivot_table)

                # 4. Pie Chart
                pie_data = tag_counts.groupby('Tag')['Count'].sum().reset_index()

                smarttag_fig = create_pie_chart(
                    pie_data,
                    names_col='Tag',
                    values_col='Count',
                    title='SmartTag-Verteilung',
                    color_map=color_map
                )

                st.plotly_chart(smarttag_fig, use_container_width=True)


                # 5. Summary Table
                st.dataframe(
                    pie_data.rename(columns={"Tag": "SmartTag", "Count": "Treffer"}).set_index("SmartTag")
                )

            # --------------------------
            # MEDIA TYPE ANALYSIS 
            # --------------------------
            if "Mediengattung" in df_filtered.columns:
                st.subheader("Mediengattungen")
                media_counts = df_filtered.groupby([
                    pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                    'Mediengattung'
                ]).size().reset_index(name='Count')

                if not media_counts.empty:
                    media_colors = {m: px.colors.qualitative.Plotly[i%20] for i,m in enumerate(media_counts['Mediengattung'].unique())}
                    
              # Bar chart with custom legend title font size
                    bar_fig = px.bar(
                        media_counts,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Mediengattung",
                        color_discrete_map=media_colors,
                        barmode="group"
                    )

                    bar_fig.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend=dict(font=dict(size=13))  # Optional: change legend item font size
                    )

                    st.plotly_chart(bar_fig)

                    # Line chart with custom legend title font size
                    line_fig = px.line(
                        media_counts,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Mediengattung",
                        color_discrete_map=media_colors
                    )

                    line_fig.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend=dict(font=dict(size=13))
                    )

                    st.plotly_chart(line_fig)

                    
                    st.dataframe(media_counts.pivot(index="Veröffentlichungsdatum",
                                                columns="Mediengattung",
                                                values="Count").fillna(0))
                    
                    media_dist = df_filtered['Mediengattung'].value_counts().reset_index()
                    media_dist.columns = ['Mediengattung', 'count']

                    media_fig = create_pie_chart(
                        media_dist,
                        names_col='Mediengattung',
                        values_col='count',
                        title='Mediengattungen-Verteilung'
                    )

                    st.plotly_chart(media_fig, use_container_width=True)

                     # Summary Table
                    st.dataframe(
                        media_dist.rename(columns={"Mediengattung": "Media Type", "count": "Treffer"}).set_index("Media Type")
                    )
                    

            # --------------------------
            # RATING ANALYSIS 
            # --------------------------
            if "Bewertung" in df_filtered.columns:
                st.subheader("Bewertungen")
                df_ratings = df_filtered[
                    df_filtered["Bewertung"].notna() & 
                    (df_filtered["Bewertung"] != "") 
                ]
                
                if not df_ratings.empty:
                    rating_data = df_ratings.groupby([
                        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                        'Bewertung'
                    ]).size().reset_index(name='Count')

                    rating_colors = {"Positiv": "green", "Negativ": "red", "Neutral": "blue"}

                    # Bar chart
                    bar_fig = px.bar(
                        rating_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Bewertung",
                        color_discrete_map=rating_colors,
                        barmode="group"
                    )

                    bar_fig.update_layout(
                        legend_title=dict(font=dict(size=16)),  # Legend title font size
                        legend=dict(font=dict(size=13))         # Legend item font size (optional)
                    )

                    st.plotly_chart(bar_fig)

                    # Line chart
                    line_fig = px.line(
                        rating_data,
                        x="Veröffentlichungsdatum",
                        y="Count",
                        color="Bewertung",
                        color_discrete_map=rating_colors
                    )

                    line_fig.update_layout(
                        legend_title=dict(font=dict(size=16)),
                        legend=dict(font=dict(size=13))
                    )

                    st.plotly_chart(line_fig)


                    st.dataframe(rating_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Bewertung",
                                                values="Count").fillna(0))
                    
                    df_ratings_grouped = df_ratings['Bewertung'].value_counts().reset_index()
                    df_ratings_grouped.columns = ['Bewertung', 'count']

                    bewertung_fig = create_pie_chart(
                        df_ratings_grouped,
                        names_col='Bewertung',
                        values_col='count',
                        title='Bewertungen-Verteilung',
                        color_map=rating_colors
                    )

                    st.plotly_chart(bewertung_fig, use_container_width=True)
                  
                  # Count and prepare the rating data
                    rating_counts = df_ratings['Bewertung'].value_counts().reset_index()
                    rating_counts.columns = ['Bewertung', 'Treffer']

                    # Display the dataframe with 'Bewertung' as index, like the others
                    st.dataframe(rating_counts.set_index("Bewertung"))
                                        
                  
      

        # ---------------------- #
        # Datenblatt
        # ---------------------- #
        
    if tab == "Datenblatt":
                
        st.subheader(" Datenblatt")
        
   
        # Identify columns related to Tags, Smart-Tags, and Search Agents
        tag_columns = [col for col in df.columns if col.startswith("Tag")]
        smart_tag_columns = [col for col in df.columns if col.startswith("Smart-Tag")]
        search_agent_columns = [col for col in df.columns if col.startswith("Suchagent")]

        # Create a copy of the DataFrame
        df_transformed = df.copy()

        # Rename columns dynamically based on the first non-null value in each column
        for col in tag_columns + smart_tag_columns + search_agent_columns:
            first_valid_value = df_transformed[col].dropna().iloc[0]  # Get the first non-null value
            if pd.notna(first_valid_value):
                df_transformed.rename(columns={col: first_valid_value}, inplace=True)

        # Display transformed table
        st.dataframe(df_transformed)

        # 📥 Allow downloading of the processed file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_transformed.to_excel(writer, index=False, sheet_name='Datenblatt')
        output.seek(0)

        st.download_button(
            label="📥 Download Processed Excel",
            data=output,
            file_name="transformed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ---------------------- #
        # Top-Quellen
        # ---------------------- #
    elif tab == "Auswertungen nach Quellen":
            
       
        st.subheader("📊 Auswertungen nach Quellen")
        
        # Ensure column names have no extra spaces
        df.columns = df.columns.str.strip()

        # Ensure necessary columns exist
        if "Quelle" in df.columns and "Veröffentlichungsdatum" in df.columns:
            # Convert date column to datetime format
            df["Veröffentlichungsdatum"] = pd.to_datetime(df["Veröffentlichungsdatum"], errors="coerce")

            # Apply Timeframe Filter (assuming filter_by_timeframe is a function defined elsewhere)
            df = filter_by_timeframe(df, "Veröffentlichungsdatum")

            # Initialize conditions as True (this allows for no filtering if no options are selected)
            final_condition = pd.Series([True] * len(df))

            # Initialize 'search_agents' by extracting the unique values from the 'Suchagent' columns
            search_agent_columns = [col for col in df.columns if "Suchagent" in col]
            search_agents = set()
            for col in search_agent_columns:
                search_agents.update(df[col].dropna().unique())

            # ------------------- Search Agent Filter -------------------
            selected_agents = st.multiselect("Suchagenten auswählen", sorted(search_agents))

            if selected_agents:
                agent_condition = df[search_agent_columns].apply(
                    lambda row: any(agent in selected_agents for agent in row.values), axis=1
                )
                final_condition &= agent_condition  # Combine with previous conditions

            # Initialize 'smarttags' by extracting the unique values from the 'Smart-Tag' columns
            smarttag_columns = [col for col in df.columns if col.startswith("Smart-Tag")]
            smarttags = set()
            for col in smarttag_columns:
                smarttags.update(df[col].dropna().unique())

            # ------------------- Smart Tag Filter -------------------
            selected_smarttags = st.multiselect("Smart-Tag auswählen", sorted(smarttags))

            if selected_smarttags:
                smarttag_condition = df[smarttag_columns].apply(
                    lambda row: any(tag in selected_smarttags for tag in row.values), axis=1
                )
                final_condition &= smarttag_condition  # Combine with previous conditions

            # ------------------- Tag Filter -------------------
            tag_columns = [col for col in df.columns if col.startswith("Tag")]
            tags = set()
            for col in tag_columns:
                tags.update(df[col].dropna().unique())

            # ------------------- Tag Filter -------------------
            selected_tags = st.multiselect("Tags auswählen", sorted(tags))

            if selected_tags:
                tag_condition = df[tag_columns].apply(
                    lambda row: any(tag in selected_tags for tag in row.values), axis=1
                )
                final_condition &= tag_condition  # Combine with previous conditions

            # Ensure the boolean series aligns with the DataFrame index
            final_condition = final_condition.reindex(df.index, fill_value=False)
            
            # Apply the filter
            df = df[final_condition]
            
            # Handle case if the DataFrame becomes empty after filtering
            if df.empty:
                print("Filtered DataFrame is empty!")

            # Handle empty DataFrame after filtering
            if df.empty:
                st.warning("⚠ No data available for the selected filters.")

            # ------------------- Filter Out 'Ohne Bewertung' and Empty Values -------------------
            df = df[df["Quelle"].notna() & (df["Quelle"] != "")]

            # 🏆 **If data exists, process it**
            if not df.empty:
                # Aggregate Data by Quelle (Source)
                quelle_counts = df.groupby("Quelle").size().reset_index(name="Total Treffer")

                # Get the top 10 Quelle sources
                top_10_quelle = quelle_counts.nlargest(10, "Total Treffer")

                # 📊 **Bar Chart (Top 10 Quelle Sources)**
                fig = px.bar(
                    top_10_quelle,
                    x="Total Treffer",
                    y="Quelle",
                    title="Top 10 Quellen nach Anzahl der Treffer",
                    color="Quelle",
                    text_auto=True
                )

                # Improve readability
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="",
                    xaxis=dict(tickangle=-45),  # Rotate X-axis labels for better readability
                    showlegend=False
                )

                st.plotly_chart(fig)

                # Display Data Table
                st.dataframe(top_10_quelle.reset_index(drop=True).set_index("Quelle"))

            else:
                st.warning("⚠ No data available for the selected filters.")
