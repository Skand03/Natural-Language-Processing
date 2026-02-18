import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import plotly.graph_objects as go
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import spaCy (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Indian Crime Analysis", page_icon="📊")

# --- LOAD DATA ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    for col in ['Date Reported', 'Date of Occurrence', 'Date Case Closed']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', format='%d-%m-%Y %H:%M')
            except ValueError:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Try default parsing
    return df

# --- NLP ANALYSIS FUNCTIONS ---

def show_crime_keywords(df, plotly_template='plotly_white'):
    """Extract and display top 20 keywords from Crime Description using CountVectorizer"""
    st.subheader("🔤 Keyword Frequency Analysis")
    
    # Handle missing values
    crime_descriptions = df['Crime Description'].dropna().astype(str)
    
    if len(crime_descriptions) == 0:
        st.warning("No crime descriptions available for analysis.")
        return
    
    # Use CountVectorizer to extract keywords
    vectorizer = CountVectorizer(max_features=20, stop_words='english', lowercase=True)
    try:
        word_count_matrix = vectorizer.fit_transform(crime_descriptions)
        word_freq = word_count_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        
        # Create DataFrame for visualization
        keyword_df = pd.DataFrame({'Keyword': words, 'Frequency': word_freq})
        keyword_df = keyword_df.sort_values('Frequency', ascending=False)
        
        # Plot using Plotly
        fig = px.bar(keyword_df, x='Keyword', y='Frequency', 
                     title='Top 20 Keywords in Crime Descriptions',
                     color='Frequency',
                     template=plotly_template)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during keyword extraction: {e}")


def crime_search_engine(df):
    """Search engine to find similar crime descriptions using cosine similarity"""
    st.subheader("🔍 Crime Description Search Engine")
    
    # Handle missing values
    df_clean = df.dropna(subset=['Crime Description']).copy()
    df_clean['Crime Description'] = df_clean['Crime Description'].astype(str)
    
    if len(df_clean) == 0:
        st.warning("No crime descriptions available for search.")
        return
    
    # User input
    user_query = st.text_input("Enter a crime description to find similar cases:", 
                                placeholder="e.g., theft of vehicle")
    
    if user_query:
        # Vectorize crime descriptions
        vectorizer = CountVectorizer(stop_words='english', lowercase=True)
        try:
            crime_vectors = vectorizer.fit_transform(df_clean['Crime Description'])
            query_vector = vectorizer.transform([user_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, crime_vectors).flatten()
            
            # Get top 5 most similar
            top_indices = similarities.argsort()[-5:][::-1]
            top_scores = similarities[top_indices]
            
            # Display results
            st.write("### Top 5 Most Similar Crime Records:")
            results_df = df_clean.iloc[top_indices].copy()
            results_df['Similarity Score'] = [f"{score:.2%}" for score in top_scores]
            
            # Select relevant columns to display
            display_cols = ['Crime Description', 'City', 'Crime Domain', 'Victim Age', 
                           'Victim Gender', 'Weapon Used', 'Similarity Score']
            available_cols = [col for col in display_cols if col in results_df.columns]
            
            st.dataframe(results_df[available_cols], use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during search: {e}")


def show_wordcloud(df):
    """Generate and display word cloud from Crime Description"""
    st.subheader("☁️ Word Cloud Visualization")
    
    # Handle missing values
    crime_descriptions = df['Crime Description'].dropna().astype(str)
    
    if len(crime_descriptions) == 0:
        st.warning("No crime descriptions available for word cloud.")
        return
    
    # Combine all descriptions
    text = ' '.join(crime_descriptions)
    
    try:
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             stopwords='english',
                             max_words=100).generate(text)
        
        # Display using matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Crime Descriptions', fontsize=16, fontweight='bold')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")


def crime_description_length_analysis(df, plotly_template='plotly_white'):
    """Analyze and visualize crime description length distribution"""
    st.subheader("📏 Crime Description Length Analysis")
    
    # Handle missing values
    df_clean = df.dropna(subset=['Crime Description']).copy()
    df_clean['Crime Description'] = df_clean['Crime Description'].astype(str)
    
    if len(df_clean) == 0:
        st.warning("No crime descriptions available for length analysis.")
        return
    
    # Calculate description length
    df_clean['Description Length'] = df_clean['Crime Description'].apply(len)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Length", f"{df_clean['Description Length'].mean():.0f} chars")
    col2.metric("Median Length", f"{df_clean['Description Length'].median():.0f} chars")
    col3.metric("Max Length", f"{df_clean['Description Length'].max():.0f} chars")
    
    # Plot histogram
    fig = px.histogram(df_clean, x='Description Length', 
                       nbins=30,
                       title='Distribution of Crime Description Lengths',
                       labels={'Description Length': 'Number of Characters'},
                       template=plotly_template)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add box plot
    fig_box = px.box(df_clean, y='Description Length',
                     title='Box Plot of Description Lengths',
                     template=plotly_template)
    st.plotly_chart(fig_box, use_container_width=True)


# --- ADVANCED NLP INTELLIGENCE FUNCTIONS ---

@st.cache_data(ttl=3600)
def tfidf_keyword_analysis(df, plotly_template='plotly_white', top_n=20):
    """Extract and display top keywords using TF-IDF with importance scores"""
    st.subheader("🎯 TF-IDF Keyword Intelligence")
    
    crime_descriptions = df['Crime Description'].dropna().astype(str)
    
    if len(crime_descriptions) < 2:
        st.warning("Need at least 2 crime descriptions for TF-IDF analysis.")
        return
    
    try:
        # Use TF-IDF Vectorizer
        tfidf = TfidfVectorizer(max_features=top_n, stop_words='english', 
                                lowercase=True, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(crime_descriptions)
        
        # Get feature names and scores
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        # Create DataFrame
        keyword_df = pd.DataFrame({
            'Keyword': feature_names,
            'TF-IDF Score': tfidf_scores,
            'Importance': (tfidf_scores / tfidf_scores.max() * 100)
        }).sort_values('TF-IDF Score', ascending=False)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Keywords", len(keyword_df))
        col2.metric("Top Keyword", keyword_df.iloc[0]['Keyword'])
        col3.metric("Avg TF-IDF Score", f"{keyword_df['TF-IDF Score'].mean():.2f}")
        
        # Plot
        fig = px.bar(keyword_df, x='Keyword', y='TF-IDF Score',
                     title=f'Top {top_n} Keywords by TF-IDF Importance',
                     color='Importance',
                     color_continuous_scale='Viridis',
                     template=plotly_template)
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📊 View Detailed Keyword Data"):
            st.dataframe(keyword_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in TF-IDF analysis: {e}")


def advanced_crime_search(df, plotly_template='plotly_white'):
    """Professional search engine with TF-IDF and advanced UI controls"""
    st.subheader("🔍 Advanced Crime Intelligence Search Engine")
    
    df_clean = df.dropna(subset=['Crime Description']).copy()
    df_clean['Crime Description'] = df_clean['Crime Description'].astype(str)
    
    if len(df_clean) == 0:
        st.warning("No crime descriptions available for search.")
        return
    
    # UI Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("🔎 Enter crime description to search:", 
                                    placeholder="e.g., mobile theft near ATM",
                                    key="advanced_search")
    
    col_a, col_b = st.columns(2)
    with col_a:
        min_similarity = st.slider("Minimum Similarity Threshold (%)", 
                                    0, 100, 20, 5, key="similarity_threshold")
    with col_b:
        num_results = st.selectbox("Number of results to show", 
                                    [5, 10, 15, 20], index=0, key="num_results")
    
    if user_query:
        try:
            # Use TF-IDF for better search
            tfidf = TfidfVectorizer(stop_words='english', lowercase=True, 
                                    ngram_range=(1, 2), max_features=500)
            crime_vectors = tfidf.fit_transform(df_clean['Crime Description'])
            query_vector = tfidf.transform([user_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, crime_vectors).flatten()
            
            # Filter by threshold
            threshold = min_similarity / 100.0
            valid_indices = np.where(similarities >= threshold)[0]
            
            if len(valid_indices) == 0:
                st.warning(f"No results found with similarity >= {min_similarity}%")
                return
            
            # Get top N results
            valid_similarities = similarities[valid_indices]
            sorted_idx = valid_similarities.argsort()[::-1][:num_results]
            top_indices = valid_indices[sorted_idx]
            top_scores = similarities[top_indices]
            
            # Prepare results
            results_df = df_clean.iloc[top_indices].copy()
            results_df['Similarity %'] = (top_scores * 100).round(2)
            results_df['Confidence'] = pd.cut(top_scores * 100, 
                                              bins=[0, 30, 60, 100],
                                              labels=['Low', 'Medium', 'High'])
            
            # Display results count
            st.success(f"✅ Found {len(top_indices)} matching crime records")
            
            # Display results
            display_cols = ['Similarity %', 'Confidence', 'Crime Description', 
                           'Crime Domain', 'City', 'Victim Age', 'Victim Gender']
            available_cols = [col for col in display_cols if col in results_df.columns]
            
            # Add date if available
            if 'Date of Occurrence' in results_df.columns:
                available_cols.insert(4, 'Date of Occurrence')
            
            st.dataframe(results_df[available_cols].reset_index(drop=True), 
                        use_container_width=True, height=400)
            
            # Visualization
            fig = px.bar(x=range(1, len(top_scores)+1), y=top_scores * 100,
                        title='Similarity Scores Distribution',
                        labels={'x': 'Result Rank', 'y': 'Similarity %'},
                        template=plotly_template)
            fig.add_hline(y=min_similarity, line_dash="dash", 
                         line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Search error: {e}")


@st.cache_data(ttl=3600)
def topic_modeling_analysis(df, plotly_template='plotly_white', n_topics=5):
    """LDA-based topic modeling to discover hidden crime patterns"""
    st.subheader("🧠 Crime Topic Modeling (LDA)")
    
    crime_descriptions = df['Crime Description'].dropna().astype(str)
    
    if len(crime_descriptions) < 10:
        st.warning("Need at least 10 crime descriptions for topic modeling.")
        return
    
    try:
        with st.spinner("🔄 Discovering crime topics..."):
            # Vectorize
            tfidf = TfidfVectorizer(max_features=100, stop_words='english',
                                    lowercase=True, max_df=0.8, min_df=2)
            tfidf_matrix = tfidf.fit_transform(crime_descriptions)
            
            # LDA Model
            lda = LatentDirichletAllocation(n_components=n_topics, 
                                           random_state=42, 
                                           max_iter=20)
            lda_output = lda.fit_transform(tfidf_matrix)
            
            # Get feature names
            feature_names = tfidf.get_feature_names_out()
            
            # Display topics
            st.write("### 📋 Discovered Crime Topics:")
            
            topics_data = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-8:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics_data.append({
                    'Topic': f"Topic {topic_idx + 1}",
                    'Keywords': ', '.join(top_words),
                    'Weight': topic.sum()
                })
            
            topics_df = pd.DataFrame(topics_data)
            st.dataframe(topics_df, use_container_width=True)
            
            # Topic distribution
            topic_distribution = lda_output.sum(axis=0)
            topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
            
            fig = px.bar(x=topic_labels, y=topic_distribution,
                        title='Topic Distribution Across Crime Descriptions',
                        labels={'x': 'Topic', 'y': 'Total Weight'},
                        color=topic_distribution,
                        color_continuous_scale='Blues',
                        template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of document-topic distribution
            sample_size = min(50, len(crime_descriptions))
            fig_heat = go.Figure(data=go.Heatmap(
                z=lda_output[:sample_size].T,
                x=[f"Doc {i+1}" for i in range(sample_size)],
                y=topic_labels,
                colorscale='Viridis'
            ))
            fig_heat.update_layout(
                title=f'Topic Distribution Heatmap (First {sample_size} Documents)',
                xaxis_title='Documents',
                yaxis_title='Topics',
                template=plotly_template,
                height=400
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
    except Exception as e:
        st.error(f"Topic modeling error: {e}")


def named_entity_analysis(df, plotly_template='plotly_white'):
    """Extract and visualize named entities using spaCy"""
    st.subheader("🏷️ Named Entity Recognition (NER)")
    
    if not SPACY_AVAILABLE:
        st.warning("⚠️ spaCy not available. Install with: `pip install spacy` and `python -m spacy download en_core_web_sm`")
        return
    
    crime_descriptions = df['Crime Description'].dropna().astype(str).tolist()
    
    if len(crime_descriptions) == 0:
        st.warning("No crime descriptions available.")
        return
    
    try:
        with st.spinner("🔄 Extracting named entities..."):
            # Limit to first 200 for performance
            sample_descriptions = crime_descriptions[:200]
            
            entities = {'PERSON': [], 'GPE': [], 'ORG': [], 'LOC': []}
            
            for desc in sample_descriptions:
                doc = nlp(desc[:500])  # Limit text length
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
            
            # Count frequencies
            entity_counts = {}
            for ent_type, ent_list in entities.items():
                if ent_list:
                    counter = Counter(ent_list)
                    entity_counts[ent_type] = counter.most_common(10)
            
            if not entity_counts:
                st.info("No named entities found in the crime descriptions.")
                return
            
            # Display metrics
            cols = st.columns(4)
            labels = {'PERSON': 'Persons', 'GPE': 'Locations', 'ORG': 'Organizations', 'LOC': 'Places'}
            for idx, (ent_type, label) in enumerate(labels.items()):
                with cols[idx]:
                    count = len(entities[ent_type])
                    st.metric(label, count)
            
            # Visualize each entity type
            for ent_type, top_entities in entity_counts.items():
                if top_entities:
                    ent_df = pd.DataFrame(top_entities, columns=['Entity', 'Frequency'])
                    
                    fig = px.bar(ent_df, x='Entity', y='Frequency',
                                title=f'Top {labels.get(ent_type, ent_type)} Mentioned',
                                color='Frequency',
                                color_continuous_scale='Reds',
                                template=plotly_template)
                    fig.update_layout(xaxis_tickangle=-45, height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
    except Exception as e:
        st.error(f"NER error: {e}")


@st.cache_data(ttl=3600)
def crime_similarity_heatmap(df, plotly_template='plotly_white', sample_size=50):
    """Generate crime similarity matrix heatmap"""
    st.subheader("🔥 Crime Similarity Matrix")
    
    crime_descriptions = df['Crime Description'].dropna().astype(str)
    
    if len(crime_descriptions) < 5:
        st.warning("Need at least 5 crime descriptions for similarity analysis.")
        return
    
    try:
        # Sample for performance
        sample_size = min(sample_size, len(crime_descriptions))
        sampled_descriptions = crime_descriptions.sample(n=sample_size, random_state=42)
        
        with st.spinner(f"🔄 Computing similarity matrix for {sample_size} crimes..."):
            # Vectorize
            tfidf = TfidfVectorizer(stop_words='english', max_features=200)
            tfidf_matrix = tfidf.fit_transform(sampled_descriptions)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=[f"Crime {i+1}" for i in range(sample_size)],
                y=[f"Crime {i+1}" for i in range(sample_size)],
                colorscale='RdYlGn',
                zmid=0.5
            ))
            
            fig.update_layout(
                title=f'Crime Description Similarity Heatmap ({sample_size} samples)',
                xaxis_title='Crime Records',
                yaxis_title='Crime Records',
                template=plotly_template,
                height=600,
                width=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            avg_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean()
            st.info(f"📊 Average similarity between crimes: {avg_similarity:.2%}")
            
    except Exception as e:
        st.error(f"Similarity heatmap error: {e}")


def plot_3d_crime_scatter(df, plotly_template='plotly_white'):
    """3D scatter plot: Victim Age × Police Deployed × Description Length"""
    st.subheader("🌐 3D Crime Analysis Scatter Plot")
    
    df_clean = df.dropna(subset=['Crime Description', 'Victim Age', 'Police Deployed']).copy()
    df_clean['Description Length'] = df_clean['Crime Description'].astype(str).apply(len)
    
    if len(df_clean) < 10:
        st.warning("Need at least 10 complete records for 3D visualization.")
        return
    
    try:
        # Sample for performance
        sample_size = min(500, len(df_clean))
        df_sample = df_clean.sample(n=sample_size, random_state=42)
        
        fig = px.scatter_3d(
            df_sample,
            x='Victim Age',
            y='Police Deployed',
            z='Description Length',
            color='Crime Domain' if 'Crime Domain' in df_sample.columns else None,
            hover_data=['City', 'Crime Description'] if 'City' in df_sample.columns else None,
            title=f'3D Crime Analysis (Age × Police × Description Length)',
            template=plotly_template,
            height=700
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"3D scatter plot error: {e}")


@st.cache_data(ttl=3600)
def plot_3d_crime_clusters(df, plotly_template='plotly_white'):
    """3D crime cluster visualization using TSNE"""
    st.subheader("🎨 3D Crime Clusters (TSNE)")
    
    crime_descriptions = df['Crime Description'].dropna().astype(str)
    
    if len(crime_descriptions) < 20:
        st.warning("Need at least 20 crime descriptions for clustering.")
        return
    
    try:
        with st.spinner("🔄 Computing 3D clusters with TSNE..."):
            # Vectorize
            sample_size = min(300, len(crime_descriptions))
            sampled_desc = crime_descriptions.sample(n=sample_size, random_state=42)
            
            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(sampled_desc)
            
            # TSNE reduction to 3D
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, sample_size-1))
            tsne_result = tsne.fit_transform(tfidf_matrix.toarray())
            
            # Create DataFrame
            cluster_df = pd.DataFrame({
                'X': tsne_result[:, 0],
                'Y': tsne_result[:, 1],
                'Z': tsne_result[:, 2],
                'Description': sampled_desc.values
            })
            
            # Add crime domain if available
            if 'Crime Domain' in df.columns:
                cluster_df['Crime Domain'] = df.loc[sampled_desc.index, 'Crime Domain'].values
                color_col = 'Crime Domain'
            else:
                color_col = None
            
            fig = px.scatter_3d(
                cluster_df,
                x='X', y='Y', z='Z',
                color=color_col,
                hover_data=['Description'],
                title=f'3D Crime Clusters (TSNE Projection of {sample_size} crimes)',
                template=plotly_template,
                height=700
            )
            
            fig.update_traces(marker=dict(size=6, opacity=0.8))
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"3D clustering error: {e}")


def plot_3d_bubble_chart(df, plotly_template='plotly_white'):
    """3D bubble chart with multiple dimensions"""
    st.subheader("💫 3D Multi-Dimensional Bubble Chart")
    
    df_clean = df.dropna(subset=['Crime Description', 'Victim Age', 'Police Deployed']).copy()
    df_clean['Description Length'] = df_clean['Crime Description'].astype(str).apply(len)
    
    if len(df_clean) < 10:
        st.warning("Need at least 10 complete records.")
        return
    
    try:
        # Sample
        sample_size = min(200, len(df_clean))
        df_sample = df_clean.sample(n=sample_size, random_state=42)
        
        # Calculate bubble size (normalized)
        df_sample['Bubble Size'] = (df_sample['Description Length'] / df_sample['Description Length'].max() * 30) + 5
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df_sample['Victim Age'],
            y=df_sample['Police Deployed'],
            z=df_sample['Description Length'],
            mode='markers',
            marker=dict(
                size=df_sample['Bubble Size'],
                color=df_sample.index,
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Index")
            ),
            text=df_sample['Crime Domain'] if 'Crime Domain' in df_sample.columns else None,
            hovertemplate='<b>Age:</b> %{x}<br><b>Police:</b> %{y}<br><b>Length:</b> %{z}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Bubble Chart: Crime Characteristics',
            scene=dict(
                xaxis_title='Victim Age',
                yaxis_title='Police Deployed',
                zaxis_title='Description Length'
            ),
            template=plotly_template,
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"3D bubble chart error: {e}")


@st.cache_data(ttl=3600)
def crime_complexity_analysis(df, plotly_template='plotly_white'):
    """Analyze crime complexity score"""
    st.subheader("🎯 Crime Complexity Intelligence")
    
    df_clean = df.dropna(subset=['Crime Description']).copy()
    df_clean['Crime Description'] = df_clean['Crime Description'].astype(str)
    
    if len(df_clean) == 0:
        st.warning("No crime descriptions available.")
        return
    
    try:
        # Calculate complexity metrics
        df_clean['Description Length'] = df_clean['Crime Description'].apply(len)
        df_clean['Unique Words'] = df_clean['Crime Description'].apply(
            lambda x: len(set(re.findall(r'\w+', x.lower())))
        )
        
        # Complexity formula: length × unique_words
        df_clean['Complexity Score'] = (
            df_clean['Description Length'] * df_clean['Unique Words']
        )
        
        # Normalize to 0-100
        max_complexity = df_clean['Complexity Score'].max()
        df_clean['Normalized Complexity'] = (df_clean['Complexity Score'] / max_complexity * 100)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Complexity", f"{df_clean['Normalized Complexity'].mean():.1f}")
        col2.metric("Max Complexity", f"{df_clean['Normalized Complexity'].max():.1f}")
        col3.metric("Median Complexity", f"{df_clean['Normalized Complexity'].median():.1f}")
        col4.metric("Std Dev", f"{df_clean['Normalized Complexity'].std():.1f}")
        
        # Histogram
        fig = px.histogram(df_clean, x='Normalized Complexity',
                          nbins=40,
                          title='Crime Complexity Score Distribution',
                          labels={'Normalized Complexity': 'Complexity Score (0-100)'},
                          template=plotly_template)
        fig.add_vline(x=df_clean['Normalized Complexity'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text="Mean")
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter: Length vs Unique Words
        fig2 = px.scatter(df_clean.sample(min(500, len(df_clean))), 
                         x='Description Length', y='Unique Words',
                         color='Normalized Complexity',
                         color_continuous_scale='Reds',
                         title='Description Length vs Unique Words',
                         template=plotly_template)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Most complex crimes
        st.write("### 🔝 Top 10 Most Complex Crime Descriptions:")
        top_complex = df_clean.nlargest(10, 'Complexity Score')[
            ['Crime Description', 'Description Length', 'Unique Words', 'Normalized Complexity']
        ]
        st.dataframe(top_complex.reset_index(drop=True), use_container_width=True)
        
    except Exception as e:
        st.error(f"Complexity analysis error: {e}")


def keyword_trend_over_time(df, plotly_template='plotly_white'):
    """Analyze keyword trends over time"""
    st.subheader("📈 Keyword Trends Over Time")
    
    if 'Date of Occurrence' not in df.columns:
        st.warning("Date of Occurrence column not found.")
        return
    
    df_clean = df.dropna(subset=['Crime Description', 'Date of Occurrence']).copy()
    df_clean['Crime Description'] = df_clean['Crime Description'].astype(str)
    
    if len(df_clean) < 10:
        st.warning("Need at least 10 records with dates.")
        return
    
    try:
        # Extract top keywords
        tfidf = TfidfVectorizer(max_features=10, stop_words='english')
        tfidf.fit(df_clean['Crime Description'])
        top_keywords = tfidf.get_feature_names_out()
        
        # Let user select keyword
        selected_keyword = st.selectbox("Select keyword to track:", top_keywords)
        
        if selected_keyword:
            # Count keyword occurrences by month
            df_clean['Month'] = pd.to_datetime(df_clean['Date of Occurrence']).dt.to_period('M')
            df_clean['Has Keyword'] = df_clean['Crime Description'].str.contains(
                selected_keyword, case=False, na=False
            )
            
            trend_data = df_clean.groupby('Month')['Has Keyword'].sum().reset_index()
            trend_data['Month'] = trend_data['Month'].astype(str)
            
            fig = px.line(trend_data, x='Month', y='Has Keyword',
                         title=f'Trend of "{selected_keyword}" Over Time',
                         labels={'Has Keyword': 'Frequency'},
                         markers=True,
                         template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Keyword trend error: {e}")


# --- THEME TOGGLE ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

def toggle_theme():
    st.session_state['theme'] = 'dark' if st.session_state['theme'] == 'light' else 'light'

st.sidebar.button("Toggle Theme", on_click=toggle_theme)

# Apply theme
if st.session_state['theme'] == 'dark':
    st.markdown(
        """
        <style>
        body {
            background-color: #262730;
            color: white;
        }
        .streamlit-expanderHeader {
            color: white !important;
        }
        .css-1adrfps {
            color: white !important;
        }
        .css-qrbk6 {
            color: white !important;
        }
        .css-qrbk6 a {
            color: white !important;
        }
        div.stRadio > label {
             color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    plotly_template = "plotly_dark"
else:
    plotly_template = "plotly_white"  # Or default

# --- SIDEBAR ---
st.sidebar.header("Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV content
    csv_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = load_data(csv_data)

    # --- ADVANCED FILTERS ---
    st.sidebar.header("Advanced Filters")

    # Multi-select for Cities
    available_cities = sorted(df['City'].unique().tolist())
    selected_cities = st.sidebar.multiselect("Select Cities", available_cities)

    # Multi-select for Crime Domains
    available_crime_domains = sorted(df['Crime Domain'].unique().tolist())
    selected_crime_domains = st.sidebar.multiselect("Select Crime Domains", available_crime_domains)

    # Slider for Victim Age
    min_age = int(df['Victim Age'].min())
    max_age = int(df['Victim Age'].max())
    selected_age_range = st.sidebar.slider("Select Victim Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Date Range for Date of Occurrence
    min_date = df['Date of Occurrence'].min()
    max_date = df['Date of Occurrence'].max()
    selected_date_range = st.sidebar.date_input("Select Date of Occurrence Range",
                                                  value=[min_date, max_date])

    # --- FILTERING LOGIC ---
    df_filtered = df.copy()

    # Filter by Cities
    if selected_cities:
        df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]

    # Filter by Crime Domains
    if selected_crime_domains:
        df_filtered = df_filtered[df_filtered['Crime Domain'].isin(selected_crime_domains)]

    # Filter by Age Range
    df_filtered = df_filtered[(df_filtered['Victim Age'] >= selected_age_range[0]) & (df_filtered['Victim Age'] <= selected_age_range[1])]

   # Convert selected_date_range to datetime64[ns]
    start_date, end_date = pd.to_datetime(selected_date_range)

    # Apply date filter
    df_filtered = df_filtered[(df_filtered['Date of Occurrence'] >= start_date) & (df_filtered['Date of Occurrence'] <= end_date)]

    # --- DISPLAY FILTERED DATA ---
    if st.sidebar.checkbox("Show Filtered Data"):
        st.sidebar.write(df_filtered)

    # --- MAIN DASHBOARD ---
    st.title("Crime Analysis Dashboard")

    # --- KPIs ---
    total_cases = df_filtered.shape[0]
    cities_involved = len(df_filtered['City'].unique())
    avg_age = df_filtered['Victim Age'].mean()

    kpi1, kpi2, kpi3 = st.columns(3)

    kpi1.metric(label="Total Cases", value=total_cases)
    kpi2.metric(label="Cities Involved", value=cities_involved)
    kpi3.metric(label="Average Victim Age", value=f"{avg_age:.2f}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # --- VISUALIZATIONS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["City & Crime", "Victim Analysis", "Case Details", "Comparative Analysis", "🤖 NLP Intelligence"])

    with tab1:
        st.subheader("City and Crime Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Configurable Chart Selection
            chart_options_city = st.multiselect("Select charts to display for City Analysis:",
                                                ["Crimes by City (Bar)", "Crimes by City (Pie)", "Top Crime Types by City", "Crime Rate by City (Bar)"])

            if "Crimes by City (Bar)" in chart_options_city:
                st.subheader("Crimes by City")
                city_crime_counts = df_filtered['City'].value_counts().reset_index()
                city_crime_counts.columns = ['City', 'Number of Crimes']
                fig_city = px.bar(city_crime_counts, x='City', y='Number of Crimes', color='Number of Crimes', title="Crimes by City", template=plotly_template)
                st.plotly_chart(fig_city, use_container_width=True)

            if "Crimes by City (Pie)" in chart_options_city:
                st.subheader("Crimes by City (Pie Chart)")
                city_crime_counts = df_filtered['City'].value_counts().reset_index()
                city_crime_counts.columns = ['City', 'Number of Crimes']
                fig_city_pie = px.pie(city_crime_counts, names='City', values='Number of Crimes', title='Crimes Distribution Across Cities', template=plotly_template)
                st.plotly_chart(fig_city_pie, use_container_width=True)

            if "Top Crime Types by City" in chart_options_city:
                st.subheader("Top 5 Crime Types by City")
                top_crime_types = df_filtered.groupby(['City', 'Crime Description']).size().reset_index(name='count')
                top_crime_types = top_crime_types.groupby('City').apply(lambda x: x.nlargest(5, 'count')).reset_index(drop=True)
                fig_top_crimes = px.bar(top_crime_types, x='City', y='count', color='Crime Description', title='Top 5 Crime Types by City', template=plotly_template)
                st.plotly_chart(fig_top_crimes, use_container_width=True)
            
            if "Crime Rate by City (Bar)" in chart_options_city:
                st.subheader("Crime Rate by City")
                city_crime_counts = df_filtered['City'].value_counts().reset_index()
                city_crime_counts.columns = ['City', 'Number of Crimes']
                # Assuming a constant population for each city for simplicity
                crime_rate_city = city_crime_counts.copy()
                crime_rate_city['Population'] = 100000  # Placeholder population
                crime_rate_city['Crime Rate'] = (crime_rate_city['Number of Crimes'] / crime_rate_city['Population']) * 100000
                fig_crime_rate_city = px.bar(crime_rate_city, x='City', y='Crime Rate', color='Crime Rate', title='Crime Rate by City (per 100,000 population)', template=plotly_template)
                st.plotly_chart(fig_crime_rate_city, use_container_width=True)

        with col2:
            # Configurable Chart Selection
            chart_options_domain = st.multiselect("Select charts to display for Crime Domain Analysis:",
                                                    ["Crime Domains Distribution (Pie)", "Crime Domains (Bar)", "Crime Domain vs. City (Sunburst)", "Crime Domain Over Time (Line)"])

            if "Crime Domains Distribution (Pie)" in chart_options_domain:
                st.subheader("Crime Domains Distribution")
                crime_domain_counts = df_filtered['Crime Domain'].value_counts().reset_index()
                crime_domain_counts.columns = ['Crime Domain', 'Number of Crimes']
                fig_domain = px.pie(crime_domain_counts, names='Crime Domain', values='Number of Crimes', title="Crime Domain Distribution", template=plotly_template)
                st.plotly_chart(fig_domain, use_container_width=True)

            if "Crime Domains (Bar)" in chart_options_domain:
                st.subheader("Crime Domains (Bar Chart)")
                crime_domain_counts = df_filtered['Crime Domain'].value_counts().reset_index()
                crime_domain_counts.columns = ['Crime Domain', 'Number of Crimes']
                fig_domain_bar = px.bar(crime_domain_counts, x='Crime Domain', y='Number of Crimes', color='Crime Domain', title='Number of Crimes by Domain', template=plotly_template)
                st.plotly_chart(fig_domain_bar, use_container_width=True)

            if "Crime Domain vs. City (Sunburst)" in chart_options_domain:
                st.subheader("Crime Domain vs. City (Sunburst Chart)")
                fig_sunburst = px.sunburst(df_filtered, path=['City', 'Crime Domain'], title='Crime Domain Distribution by City', template=plotly_template)
                st.plotly_chart(fig_sunburst, use_container_width=True)
            
            if "Crime Domain Over Time (Line)" in chart_options_domain:
                st.subheader("Crime Domain Over Time (Line Chart)")
                crime_domain_time = df_filtered.groupby(['Date of Occurrence', 'Crime Domain']).size().reset_index(name='Crimes')
                crime_domain_time['Date of Occurrence'] = pd.to_datetime(crime_domain_time['Date of Occurrence'])
                crime_domain_time = crime_domain_time.sort_values('Date of Occurrence')
                fig_domain_time = px.line(crime_domain_time, x='Date of Occurrence', y='Crimes', color='Crime Domain', title='Crime Domain Trend Over Time', template=plotly_template)
                st.plotly_chart(fig_domain_time, use_container_width=True)

        st.subheader("Crimes Over Time (Line Chart)")
        date_counts = df_filtered.dropna(subset=['Date of Occurrence']).set_index('Date of Occurrence').resample('M').size().reset_index(name='Crimes')
        if not date_counts.empty:
            fig_time = px.line(date_counts, x='Date of Occurrence', y='Crimes', title='Monthly Trend of Crimes', template=plotly_template)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.write("No date-related data to display this chart.")

    with tab2:
        st.subheader("Victim Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Configurable Chart Selection
            chart_options_victim_age = st.multiselect("Select charts to display for Victim Age Analysis:",
                                                    ["Victim Age Distribution (Histogram)", "Victim Age (Box Plot)", "Victim Age vs. Crime Domain (Violin Plot)", "Victim Age Density (KDE)"])

            if "Victim Age Distribution (Histogram)" in chart_options_victim_age:
                st.subheader("Victim Age Distribution (Histogram)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df_filtered['Victim Age'], kde=True, bins=20, color="skyblue", ax=ax)
                ax.set_title("Victim Age Distribution")
                ax.set_xlabel("Age")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            if "Victim Age (Box Plot)" in chart_options_victim_age:
                st.subheader("Victim Age (Box Plot)")
                fig_age_box = px.box(df_filtered, y='Victim Age', title='Distribution of Victim Ages', template=plotly_template)
                st.plotly_chart(fig_age_box, use_container_width=True)

            if "Victim Age vs. Crime Domain (Violin Plot)" in chart_options_victim_age:
                st.subheader("Victim Age vs. Crime Domain (Violin Plot)")
                fig_age_violin = px.violin(df_filtered, x='Crime Domain', y='Victim Age', color='Crime Domain', title='Victim Age Distribution by Crime Domain', template=plotly_template)
                st.plotly_chart(fig_age_violin, use_container_width=True)
            
            if "Victim Age Density (KDE)" in chart_options_victim_age:
                st.subheader("Victim Age Density (KDE)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(df_filtered['Victim Age'], fill=True, color="skyblue", ax=ax)
                ax.set_title("Victim Age Density")
                ax.set_xlabel("Age")
                ax.set_ylabel("Density")
                st.pyplot(fig)

        with col2:
            # Configurable Chart Selection
            chart_options_victim_gender = st.multiselect("Select charts to display for Victim Gender Analysis:",
                                                        ["Victim Gender Distribution (Bar Chart)", "Victim Gender (Pie Chart)", "Gender vs. Crime Domain (Stacked Bar Chart)", "Gender Distribution Over Time (Line)"])

            if "Victim Gender Distribution (Bar Chart)" in chart_options_victim_gender:
                st.subheader("Victim Gender Distribution (Bar Chart)")
                gender_counts = df_filtered['Victim Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Number of Cases']
                fig_gender = px.bar(gender_counts, x='Gender', y='Number of Cases', color='Gender', title="Distribution of Victims by Gender", template=plotly_template)
                st.plotly_chart(fig_gender, use_container_width=True)

            if "Victim Gender (Pie Chart)" in chart_options_victim_gender:
                st.subheader("Victim Gender (Pie Chart)")
                gender_counts = df_filtered['Victim Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Number of Cases']
                fig_gender_pie = px.pie(gender_counts, names='Gender', values='Number of Cases', title='Proportion of Victims by Gender', template=plotly_template)
                st.plotly_chart(fig_gender_pie, use_container_width=True)

            if "Gender vs. Crime Domain (Stacked Bar Chart)" in chart_options_victim_gender:
                st.subheader("Gender vs. Crime Domain (Stacked Bar Chart)")
                gender_crime = pd.crosstab(df_filtered['Victim Gender'], df_filtered['Crime Domain'])
                fig_gender_crime = px.bar(gender_crime, title='Crime Domains by Gender', labels={'value': 'Number of Cases'}, template=plotly_template)
                st.plotly_chart(fig_gender_crime, use_container_width=True)
            
            if "Gender Distribution Over Time (Line)" in chart_options_victim_gender:
                st.subheader("Gender Distribution Over Time (Line)")
                gender_time = df_filtered.groupby(['Date of Occurrence', 'Victim Gender']).size().reset_index(name='Cases')
                gender_time['Date of Occurrence'] = pd.to_datetime(gender_time['Date of Occurrence'])
                gender_time = gender_time.sort_values('Date of Occurrence')
                fig_gender_time = px.line(gender_time, x='Date of Occurrence', y='Cases', color='Victim Gender', title='Gender Distribution Trend Over Time', template=plotly_template)
                st.plotly_chart(fig_gender_time, use_container_width=True)

        st.subheader("Victim Age vs. Gender (Scatter Plot)")
        fig_age_gender = px.scatter(df_filtered, x='Victim Age', y='Victim Gender', color='Crime Domain', title='Victim Age vs Gender', template=plotly_template)
        st.plotly_chart(fig_age_gender, use_container_width=True)

    with tab3:
        st.subheader("Case Details")
        col1, col2 = st.columns(2)
        with col1:
            # Configurable Chart Selection
            chart_options_weapon = st.multiselect("Select charts to display for Weapon Analysis:",
                                                    ["Weapon Used Distribution (Bar Chart)", "Case Closed Status (Pie Chart)", "Police Deployed Distribution", "Weapon Used vs. Crime Domain (Stacked Bar)"])

            if "Weapon Used Distribution (Bar Chart)" in chart_options_weapon:
                st.subheader("Weapon Used Distribution (Bar Chart)")
                weapon_counts = df_filtered['Weapon Used'].value_counts().reset_index()
                weapon_counts.columns = ['Weapon', 'Number of Crimes']
                fig_weapon = px.bar(weapon_counts, x='Weapon', y='Number of Crimes', color='Weapon', title="Distribution of Weapons Used", template=plotly_template)
                st.plotly_chart(fig_weapon, use_container_width=True)

            if "Case Closed Status (Pie Chart)" in chart_options_weapon:
                st.subheader("Case Closed Status (Pie Chart)")
                case_closed_counts = df_filtered['Case Closed'].value_counts().reset_index()
                case_closed_counts.columns = ['Case Closed', 'Number of Cases']
                fig_case_closed = px.pie(case_closed_counts, names='Case Closed', values='Number of Cases', title='Proportion of Cases Closed', template=plotly_template)
                st.plotly_chart(fig_case_closed, use_container_width=True)

            if "Police Deployed Distribution" in chart_options_weapon:
                st.subheader("Police Deployed Distribution")
                fig_police_hist = px.histogram(df_filtered, x='Police Deployed', title='Distribution of Police Deployed', template=plotly_template)
                st.plotly_chart(fig_police_hist, use_container_width=True)
            
            if "Weapon Used vs. Crime Domain (Stacked Bar)" in chart_options_weapon:
                st.subheader("Weapon Used vs. Crime Domain (Stacked Bar)")
                weapon_crime = pd.crosstab(df_filtered['Weapon Used'], df_filtered['Crime Domain'])
                fig_weapon_crime = px.bar(weapon_crime, title='Crime Domains by Weapon Used', labels={'value': 'Number of Cases'}, template=plotly_template)
                st.plotly_chart(fig_weapon_crime, use_container_width=True)

        with col2:
            # Configurable Chart Selection
            chart_options_police = st.multiselect("Select charts to display for Police Analysis:",
                                                    ["Police Deployed vs Case Closed (Stacked Bar Chart)", "Police Deployed (Box Plot)", "Police Deployed Over Time (Line)"])

            if "Police Deployed vs Case Closed (Stacked Bar Chart)" in chart_options_police:
                st.subheader("Police Deployed vs Case Closed (Stacked Bar Chart)")
                cross_tab = pd.crosstab(df_filtered['Police Deployed'], df_filtered['Case Closed'])
                fig, ax = plt.subplots()
                cross_tab.plot(kind='bar', stacked=False, ax=ax, colormap='viridis')
                plt.title('Police Deployed vs Case Closed')
                plt.xlabel('Police Deployed')
                plt.ylabel('Number of Cases')
                plt.xticks(rotation=0)
                st.pyplot(fig)

            if "Police Deployed (Box Plot)" in chart_options_police:
                st.subheader("Police Deployed (Box Plot)")
                fig_police_box = px.box(df_filtered, y='Police Deployed', title='Distribution of Police Deployed', template=plotly_template)
                st.plotly_chart(fig_police_box, use_container_width=True)
            
            if "Police Deployed Over Time (Line)" in chart_options_police:
                st.subheader("Police Deployed Over Time (Line)")
                police_time = df_filtered.groupby(['Date of Occurrence', 'Police Deployed']).size().reset_index(name='Cases')
                police_time['Date of Occurrence'] = pd.to_datetime(police_time['Date of Occurrence'])
                police_time = police_time.sort_values('Date of Occurrence')
                fig_police_time = px.line(police_time, x='Date of Occurrence', y='Cases', color='Police Deployed', title='Police Deployment Trend Over Time', template=plotly_template)
                st.plotly_chart(fig_police_time, use_container_width=True)

        st.subheader("Case Closed Over Time (Line Chart)")
        closed_over_time = df_filtered.dropna(subset=['Date Case Closed']).set_index('Date Case Closed').sort_index()
        if not closed_over_time.empty:
            monthly_cases = closed_over_time.resample('M').size()
            monthly_cases = monthly_cases.reset_index(name='Cases Closed')
            fig_time = px.line(monthly_cases, x='Date Case Closed', y='Cases Closed', title='Monthly Trend of Closed Cases', template=plotly_template)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.write("No date-related data to display this chart.")

    with tab4:
        st.subheader("Comparative Analysis")

        # Configurable Chart Selection
        chart_options_comparative = st.multiselect("Select charts to display for Comparative Analysis:",
                                                    ["Crime Domain vs Victim Age (Box Plot)", "Victim Age vs. Police Deployed (Scatter Chart)",
                                                     "Tree Chart of Crime Domains", "Parallel Categories Diagram", "Crime Domain vs Weapon Used (Heatmap)", "City vs. Case Closed (Bar)"])

        if "Crime Domain vs Victim Age (Box Plot)" in chart_options_comparative:
            st.subheader("Crime Domain vs Victim Age (Box Plot)")
            fig_box = px.box(df_filtered, x='Crime Domain', y='Victim Age', color='Crime Domain', title='Crime Domain vs Victim Age', template=plotly_template)
            st.plotly_chart(fig_box, use_container_width=True)

        if "Victim Age vs. Police Deployed (Scatter Chart)" in chart_options_comparative:
            st.subheader("Scatter Chart: Victim Age vs. Police Deployed")
            fig_scatter = px.scatter(df_filtered, x='Victim Age', y='Police Deployed', color='Crime Domain',
                                     hover_data=['City'], title='Victim Age vs Police Deployed', template=plotly_template)
            st.plotly_chart(fig_scatter, use_container_width=True)

        if "Tree Chart of Crime Domains" in chart_options_comparative:
            st.subheader("Tree Chart of Crime Domains")
            fig_tree = px.treemap(df_filtered, path=['City', 'Crime Domain'], title='Tree Chart of Crime Domains by City', template=plotly_template)
            st.plotly_chart(fig_tree, use_container_width=True)

        if "Parallel Categories Diagram" in chart_options_comparative:
            st.subheader("Parallel Categories Diagram: Crime Analysis")
            fig_parallel = px.parallel_categories(df_filtered,
                                                 dimensions=['City', 'Crime Domain', 'Victim Gender'],
                                                 title='Parallel Categories Diagram of Crime Factors', template=plotly_template)
            st.plotly_chart(fig_parallel, use_container_width=True)
        
        if "Crime Domain vs Weapon Used (Heatmap)" in chart_options_comparative:
            st.subheader("Crime Domain vs Weapon Used (Heatmap)")
            cross_tab = pd.crosstab(df_filtered['Crime Domain'], df_filtered['Weapon Used'])
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt='d', ax=ax)
            plt.title('Heatmap of Crime Domain vs Weapon Used')
            plt.xlabel('Weapon Used')
            plt.ylabel('Crime Domain')
            st.pyplot(fig)
        
        if "City vs. Case Closed (Bar)" in chart_options_comparative:
            st.subheader("City vs. Case Closed (Bar)")
            city_case_closed = pd.crosstab(df_filtered['City'], df_filtered['Case Closed'])
            fig_city_case_closed = px.bar(city_case_closed, title='Case Closed Status by City', labels={'value': 'Number of Cases'}, template=plotly_template)
            st.plotly_chart(fig_city_case_closed, use_container_width=True)

    with tab5:
        st.title("🤖 AI-Powered Crime Intelligence & NLP Analysis")
        
        # Check if Crime Description column exists
        if 'Crime Description' not in df_filtered.columns:
            st.error("❌ Crime Description column not found in the dataset. NLP features are unavailable.")
        else:
            # Display header with metrics
            st.success(f"✅ Analyzing {len(df_filtered)} crime records (after filters applied)")
            
            # Create tabs for different NLP sections
            nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs([
                "🔍 Search & Keywords",
                "🧠 Topic Modeling & NER",
                "📊 2D Analytics",
                "🌐 3D Visualizations"
            ])
            
            with nlp_tab1:
                st.header("Advanced Search & Keyword Intelligence")
                
                # Advanced Search Engine
                with st.expander("🔍 Advanced Crime Search Engine", expanded=True):
                    advanced_crime_search(df_filtered, plotly_template)
                
                st.markdown("---")
                
                # TF-IDF Keywords
                with st.expander("🎯 TF-IDF Keyword Intelligence", expanded=True):
                    tfidf_keyword_analysis(df_filtered, plotly_template, top_n=20)
                
                st.markdown("---")
                
                # Basic keyword frequency
                with st.expander("🔤 Basic Keyword Frequency"):
                    show_crime_keywords(df_filtered, plotly_template)
                
                st.markdown("---")
                
                # Word Cloud
                with st.expander("☁️ Word Cloud Visualization"):
                    show_wordcloud(df_filtered)
            
            with nlp_tab2:
                st.header("Topic Modeling & Named Entity Recognition")
                
                # Topic Modeling
                with st.expander("🧠 Crime Topic Modeling (LDA)", expanded=True):
                    topic_modeling_analysis(df_filtered, plotly_template, n_topics=5)
                
                st.markdown("---")
                
                # Named Entity Recognition
                with st.expander("🏷️ Named Entity Recognition (NER)", expanded=True):
                    named_entity_analysis(df_filtered, plotly_template)
            
            with nlp_tab3:
                st.header("2D Analytics & Intelligence Metrics")
                
                # Crime Similarity Heatmap
                with st.expander("🔥 Crime Similarity Matrix", expanded=True):
                    crime_similarity_heatmap(df_filtered, plotly_template, sample_size=50)
                
                st.markdown("---")
                
                # Complexity Analysis
                with st.expander("🎯 Crime Complexity Intelligence", expanded=True):
                    crime_complexity_analysis(df_filtered, plotly_template)
                
                st.markdown("---")
                
                # Description Length Analysis
                with st.expander("📏 Description Length Analysis"):
                    crime_description_length_analysis(df_filtered, plotly_template)
                
                st.markdown("---")
                
                # Keyword Trends
                with st.expander("📈 Keyword Trends Over Time"):
                    keyword_trend_over_time(df_filtered, plotly_template)
            
            with nlp_tab4:
                st.header("3D Visualizations & Cluster Analysis")
                
                # 3D Scatter Plot
                with st.expander("🌐 3D Crime Scatter Plot", expanded=True):
                    plot_3d_crime_scatter(df_filtered, plotly_template)
                
                st.markdown("---")
                
                # 3D Clusters
                with st.expander("🎨 3D Crime Clusters (TSNE)", expanded=True):
                    plot_3d_crime_clusters(df_filtered, plotly_template)
                
                st.markdown("---")
                
                # 3D Bubble Chart
                with st.expander("💫 3D Multi-Dimensional Bubble Chart"):
                    plot_3d_bubble_chart(df_filtered, plotly_template)
else:
    st.warning("Upload data to continue!")
