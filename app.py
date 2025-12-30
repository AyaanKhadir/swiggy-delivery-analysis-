import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import plotly.express as px
from scipy.stats import linregress

sns.set_style("whitegrid")

# Load your swiggy.csv
df = pd.read_csv("data/swiggy.csv")

# Clean up if there's an extra index column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Feature Engineering
df['Total_Delivery_Time_Min'] = df['Preparation_Time_Min'] + (df['Rider_Distance_KM'] * 4) + np.random.randint(1, 5,
                                                                                                               len(df))
df['Is_Late'] = (df['Total_Delivery_Time_Min'] > 45).astype(int)


def analyze_swiggy():
    outputs = []

    # Overview
    outputs.append(gr.Markdown(f"### 📊 Swiggy Simulated Orders Analysis ({len(df)} Orders)"))
    outputs.append(gr.Dataframe(df.head(10), label="Sample Orders"))

    # Statistics
    outputs.append(gr.Markdown("### Key Statistics"))
    stats_summary = df[['Avg_Meal_Price_INR', 'Customer_Rating', 'Total_Delivery_Time_Min', 'Rider_Distance_KM']].agg(
        ['mean', 'median', 'std']).round(2)
    outputs.append(gr.Dataframe(stats_summary))

    # Late Delivery
    late_prob = df['Is_Late'].mean()
    city_late = df.groupby('City')['Is_Late'].mean().sort_values(ascending=False)
    outputs.append(gr.Markdown(f"**Overall Late Delivery Probability: {late_prob:.1%}**"))
    outputs.append(gr.Dataframe(city_late.reset_index().rename(columns={'Is_Late': 'Late Probability'})))

    # Correlation
    corr_result = linregress(df['Total_Delivery_Time_Min'], df['Customer_Rating'])
    corr = corr_result.rvalue
    outputs.append(gr.Markdown(f"**Correlation (Delivery Time vs Rating): {corr:.2f}**"))

    # Plot 1: Delivery Time Histogram
    fig1, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df['Total_Delivery_Time_Min'], kde=True, bins=10, color='skyblue', ax=ax)
    ax.set_title('Distribution of Total Delivery Time (Minutes)')
    outputs.append(gr.Plot(fig1))

    # Plot 2: Late Probability by City
    fig2 = px.bar(city_late.reset_index(), x='City', y='Is_Late',
                  title='Late Delivery Probability by City',
                  labels={'Is_Late': 'Probability'}, color='Is_Late',
                  color_continuous_scale='Reds')
    outputs.append(gr.Plot(fig2))

    # Plot 3: Delivery Time vs Rating Scatter (no trendline to avoid statsmodels)
    fig3 = px.scatter(df, x='Total_Delivery_Time_Min', y='Customer_Rating',
                      color='City', hover_data=['Cuisine'],
                      title=f'Delivery Time vs Customer Rating (R = {corr:.2f})')
    outputs.append(gr.Plot(fig3))

    # Extra Plots
    outputs.append(gr.Markdown("### Additional Insights"))

    # Meal Price by City
    fig4 = px.box(df, x='City', y='Avg_Meal_Price_INR', color='City',
                  title='Meal Price Distribution by City')
    outputs.append(gr.Plot(fig4))

    # Cuisine Popularity
    cuisine_count = df['Cuisine'].value_counts()
    fig5 = px.pie(values=cuisine_count.values, names=cuisine_count.index,
                  title='Cuisine Popularity Share')
    outputs.append(gr.Plot(fig5))

    # Distance vs Late Delivery
    fig6 = px.scatter(df, x='Rider_Distance_KM', y='Total_Delivery_Time_Min',
                      color='Is_Late', size='Avg_Meal_Price_INR',
                      title='Distance vs Delivery Time (Size = Price, Color = Late)')
    outputs.append(gr.Plot(fig6))

    return outputs


# Gradio Dashboard
with gr.Blocks(title="Swiggy Delivery Analysis") as demo:
    gr.Markdown("# 🍛 Swiggy Delivery Analysis Dashboard")
    gr.Markdown("Interactive EDA on simulated food delivery data with late delivery modeling")

    with gr.Column():
        btn = gr.Button("🚀 Run Full Analysis", variant="primary", size="lg")
        outputs_list = [
            gr.Markdown(), gr.Dataframe(),
            gr.Markdown(), gr.Dataframe(),
            gr.Markdown(), gr.Dataframe(),
            gr.Markdown(),
            gr.Plot(), gr.Plot(), gr.Plot(),
            gr.Markdown(),
            gr.Plot(), gr.Plot(), gr.Plot()
        ]
        btn.click(fn=analyze_swiggy, outputs=outputs_list)

demo.launch()