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
        "Datenübersicht",
        "Suchagentenanalyse",
        "Tagsanalyse",
        "Top-Quellenanalyse",
        "Datenblatt"
    ]
)

with st.sidebar:
    uploaded_file = st.file_uploader("", type=["xls", "xlsx"])
st.markdown(
    """
    <style>
        body {
            font-size: 14px;  # Adjust the font size
            
        }
      
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data(file):
    """Load the Excel file once and cache it."""
    return pd.read_excel(file) if file else None


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


# 📌 Ensure the file is loaded before proceeding
if uploaded_file:
    df = load_data(uploaded_file)

    # ✅ Data processing starts here
    search_agent_columns = [col for col in df.columns if col.startswith("Suchagent")]

    if not search_agent_columns:
        st.warning("No 'Suchagent' columns found.")
    else:
        # Melt DataFrame to restructure
        df_melted = df.melt(id_vars=['Veröffentlichungsdatum'], value_vars=search_agent_columns, 
                            var_name="Suchagent Column", value_name="Suchagent")
        agent_counts_over_time = df_melted.groupby(['Veröffentlichungsdatum', 'Suchagent']).size().reset_index(name='Count')

        # 📌 Convert date and aggregate by month
        df['Veröffentlichungsdatum'] = pd.to_datetime(df['Veröffentlichungsdatum'], errors='coerce')
        agent_counts_over_time = agent_counts_over_time.groupby(
            [pd.Grouper(key="Veröffentlichungsdatum", freq="M"), 'Suchagent']
        )['Count'].sum().reset_index()

        # 📌 Display success message
        st.sidebar.success("File uploaded successfully! ✅")

     

    # ---------------------- #
    # 📌 Tab: Data Overview
    # ---------------------- #
    
    if tab == "Datenübersicht":
        
        st.header("📊 Datenübersicht")

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
        # 📌 Tags Overview: Search for Tags and Remove Duplicates
        # ---------------------- #
        # Collect all columns that start with 'Tag' or 'Smart-Tag'
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
    # 📌 SuchagentenZeitreihe Tab
    # ---------------------- #

    if tab == "Suchagentenanalyse":
       st.subheader("📊 Suchagentenanalyse")

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
                st.plotly_chart(px.bar(agent_data, x="Veröffentlichungsdatum", y="Count",
                                    color="Suchagent", color_discrete_map=colors,barmode="group"))
                
                # Line Chart
                st.plotly_chart(px.line(agent_data, x="Veröffentlichungsdatum", y="Count",
                                    color="Suchagent", color_discrete_map=colors))

                # Time-based Table
                st.dataframe(agent_data.pivot(index="Veröffentlichungsdatum", 
                                            columns="Suchagent", 
                                            values="Count").fillna(0))
                
                # Pie Chart (EXACTLY AS IN ORIGINAL)
                agent_counts_total = df_melted[df_melted['Suchagent'].isin(selected_agents)]['Suchagent'].value_counts()
                st.plotly_chart(px.pie(
                    names=agent_counts_total.index, 
                    values=agent_counts_total.values,
                    color=agent_counts_total.index,
                    title="Suchagent-Verteilung",
                    color_discrete_map=colors
                ))
                
                # Summary Table (EXACTLY AS IN ORIGINAL)
                df_agents = agent_counts_total.reset_index()
                df_agents.columns = ["Suchagent", "Treffer"]
                st.dataframe(df_agents)

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
                    media_colors = {m: px.colors.qualitative.Plotly[i%len(px.colors.qualitative.Plotly)]
                                for i,m in enumerate(media_types)}
                    
                    # Bar Chart
                    st.plotly_chart(px.bar(media_data, x="Veröffentlichungsdatum", y="Count",
                                        color="Mediengattung", color_discrete_map=media_colors,barmode="group"))
                    
                    # Line Chart
                    st.plotly_chart(px.line(media_data, x="Veröffentlichungsdatum", y="Count",
                                        color="Mediengattung", color_discrete_map=media_colors))

                    # Time-based Table
                    st.dataframe(media_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Mediengattung",
                                                values="Count").fillna(0))
                    
                    # Pie Chart (EXACTLY AS IN ORIGINAL)
                    
                    media_dist = df_filtered['Mediengattung'].value_counts().reset_index()
                    st.plotly_chart(px.pie(
                        names=media_dist["Mediengattung"], 
                        values=media_dist["count"],
                        title="Mediengattungen-verteilung",
                        color_discrete_map=media_colors
                    ))
                    
                    # Summary Table
                    st.dataframe(media_dist.rename(columns={"Mediengattung": "Media Type", "count": "Treffer"}))

            # 3. Rating Analysis
            if "Bewertung" in df_filtered.columns:
                st.subheader("Bewertungen")
                df_ratings = df_filtered[
                    df_filtered["Bewertung"].notna() & 
                    (df_filtered["Bewertung"] != "") & 
                    (df_filtered["Bewertung"] != "Ohne Bewertung")
                ]
                
                if not df_ratings.empty:
                    rating_data = df_ratings.groupby([
                        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                        'Bewertung'
                    ]).size().reset_index(name='Count')

                    # Visualizations
                    rating_colors = {"Positiv": "green", "Negativ": "red", "Neutral": "blue"}
                    
                    # Bar Chart
                    st.plotly_chart(px.bar(rating_data, x="Veröffentlichungsdatum", y="Count",
                                        color="Bewertung", color_discrete_map=rating_colors,barmode="group"))
                    
                    # Line Chart
                    st.plotly_chart(px.line(rating_data, x="Veröffentlichungsdatum", y="Count",
                                        color="Bewertung", color_discrete_map=rating_colors))

                    # Time-based Table
                    st.dataframe(rating_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Bewertung",
                                                values="Count").fillna(0))
                    
                    # Pie Chart (EXACTLY AS IN ORIGINAL)
                    st.plotly_chart(px.pie(
                        df_ratings,
                        names="Bewertung",
                        title="Bewertungen-verteilung",
                        color="Bewertung",
                        color_discrete_map=rating_colors
                    ))
                    
                    # Summary Table
                    rating_totals = df_ratings['Bewertung'].value_counts().reset_index()
                    rating_totals.columns = ["Bewertung", "Gesamtanzahl"]
                    st.dataframe(rating_totals)
        # 📌 TagsZeitreihe
    # ---------------------- #
    if tab == "Tagsanalyse":
       st.subheader("📊 Tagnalyse")  # Changed title

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
            st.subheader("Tag-Trends")  # Changed header
            
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
                    labels={"Veröffentlichungsdatum": "Date", "Count": "Treffer", "Tag": "Tag"},  # Changed labels
                    barmode="group",
                    text_auto=True,
                    color_discrete_map=color_map
                )
                st.plotly_chart(bar_chart)

                # 2. Line Chart
                line_chart = px.line(
                    tag_counts, 
                    x="Veröffentlichungsdatum", 
                    y="Count", 
                    color="Tag",
                    color_discrete_map=color_map
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
                st.plotly_chart(px.pie(
                    pie_data,
                    names='Tag',
                    values='Count',
                    title="Tag-Verteilung",  
                    color_discrete_map=color_map
                ))

                # 5. Summary Table
                st.dataframe(pie_data.rename(columns={"Tag": "Tag", "Count": "Treffer"}))

            # --------------------------
            # MEDIA TYPE ANALYSIS (Same as before)
            # --------------------------
            if "Mediengattung" in df_filtered.columns:
                st.subheader("Mediengattungen")
                media_counts = df_filtered.groupby([
                    pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                    'Mediengattung'
                ]).size().reset_index(name='Count')

                if not media_counts.empty:
                    media_colors = {m: px.colors.qualitative.Plotly[i%20] for i,m in enumerate(media_counts['Mediengattung'].unique())}
                    
                    st.plotly_chart(px.bar(media_counts, x="Veröffentlichungsdatum", y="Count",
                                        color="Mediengattung", color_discrete_map=media_colors, barmode="group"))
                    st.plotly_chart(px.line(media_counts, x="Veröffentlichungsdatum", y="Count",
                                        color="Mediengattung", color_discrete_map=media_colors))
                    
                    st.dataframe(media_counts.pivot(index="Veröffentlichungsdatum",
                                                columns="Mediengattung",
                                                values="Count").fillna(0))
                    
                    media_dist = df_filtered['Mediengattung'].value_counts().reset_index()
                    st.plotly_chart(px.pie(media_dist, names='Mediengattung', values='count',
                                        title="Mediengattungen-Verteilung"))
                    st.dataframe(media_dist.rename(columns={"Mediengattung": "Media Type", "count": "Treffer"}))

            # --------------------------
            # RATING ANALYSIS (Same as before)
            # --------------------------
            if "Bewertung" in df_filtered.columns:
                st.subheader("Bewertungen")
                df_ratings = df_filtered[
                    df_filtered["Bewertung"].notna() & 
                    (df_filtered["Bewertung"] != "") & 
                    (df_filtered["Bewertung"] != "Ohne Bewertung")
                ]
                
                if not df_ratings.empty:
                    rating_data = df_ratings.groupby([
                        pd.Grouper(key='Veröffentlichungsdatum', freq=freq),
                        'Bewertung'
                    ]).size().reset_index(name='Count')

                    rating_colors = {"Positiv": "green", "Negativ": "red", "Neutral": "blue"}
                    st.plotly_chart(px.bar(rating_data, x="Veröffentlichungsdatum", y="Count",
                                        color="Bewertung", color_discrete_map=rating_colors,barmode="group"))
                    st.plotly_chart(px.line(rating_data, x="Veröffentlichungsdatum", y="Count",
                                        color="Bewertung", color_discrete_map=rating_colors))
                    
                    st.dataframe(rating_data.pivot(index="Veröffentlichungsdatum",
                                                columns="Bewertung",
                                                values="Count").fillna(0))
                    
                    st.plotly_chart(px.pie(df_ratings, names="Bewertung", title="Bewertungen-Verteilung",
                                        color="Bewertung", color_discrete_map=rating_colors))
                    st.dataframe(df_ratings['Bewertung'].value_counts().reset_index().rename(
                        columns={"index": "Bewertung", "Bewertung": "Treffer"}))
        # ---------------------- #
        # 📌 Top-Quellen
        # ---------------------- #
    elif tab == "Top-Quellenanalyse":
            
        st.subheader("📊 Top Quelle")

        # Ensure column names have no extra spaces
        df.columns = df.columns.str.strip()

        # Ensure necessary columns exist
        if "Quelle" in df.columns and "Veröffentlichungsdatum" in df.columns:
            # Convert date column to datetime format
            df["Veröffentlichungsdatum"] = pd.to_datetime(df["Veröffentlichungsdatum"], errors="coerce")

            # Apply Timeframe Filter**
            df = filter_by_timeframe(df, "Veröffentlichungsdatum")

            # Search Agent Filtering**
            search_agent_columns = [col for col in df.columns if "Suchagent" in col]
            search_agents = set()
            for col in search_agent_columns:
                search_agents.update(df[col].dropna().unique())

            if search_agents:
                selected_agents = st.multiselect("Select Search Agents", sorted(search_agents))
                if selected_agents:
                    agent_condition = df[search_agent_columns].apply(
                        lambda row: any(agent in selected_agents for agent in row.values), axis=1
                    )
                    df = df[agent_condition]  # Filter by selected agents
            else:
                st.warning("⚠ No Suchagent values found in the uploaded file.")

            # 🚀 **Filter Out 'Ohne Bewertung' and Empty Values**
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
                    title="Top 10 Quellen by Number of Treffer",
                    
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
                # Reset index and remove it

               
                # 📋 **Display Data Table**
                st.dataframe(top_10_quelle)
                

            else:
                st.warning("⚠ No data available for the selected filters.")


        # ---------------------- #
        # 📌 Datenblatt
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


  