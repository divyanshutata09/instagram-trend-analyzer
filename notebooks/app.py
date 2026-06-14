import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime

# --- CONFIGURATION & VIEW STATE SETUP ---
st.set_page_config(
    page_title="Instagram Content Virality Index Engine",
    page_icon="🔮",
    layout="wide"
)

# Deep Dark Glassmorphism Styling Injection
st.markdown("""
<style>
    body { background-color: #0E1117; color: #FAFAFA; }
    .stTextArea textarea { background-color: #1E222B !important; color: #FAFAFA !important; border: 1px solid #3F444E !important; }
    .metric-container { background: #161A22; border: 1px solid #30363D; padding: 24px; border-radius: 12px; margin-bottom: 20px; }
    .recommendation-pill { background-color: #21262D; color: #58A6FF; border: 1px solid #30363D; padding: 6px 12px; border-radius: 20px; display: inline-block; margin: 4px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- IN-MEMORY ARTIFACT LOADING LAYER ---
@st.cache_resource
def load_production_pipeline_dependencies():
    """Reads system serialized models into the global application worker state context."""
    try:
        # Expected production paths mapped within the repository[cite: 1]
        classifier = joblib.load('models/viral_post_predictor.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        historical_corpus_matrix = joblib.load('models/corpus_embeddings.joblib')
        reference_data = pd.read_csv('data/social_media_performance.csv')
        return classifier, vectorizer, historical_corpus_matrix, reference_data
    except FileNotFoundError:
        return None, None, None, None

model, tfidf_vec, corpus_embeddings, df_reference = load_production_pipeline_dependencies()

# --- HELPER WORKER LOGIC ---
def extract_hashtag_count(text: str) -> int:
    """Helper method executing regex pattern tracking over text stream segments."""
    return len(re.findall(r'#\w+', text))

# --- APP LAYOUT VIEW ---
st.title("🔮 Instagram Virality Optimizer Engine")
st.caption("Execute validation pipelines against draft media copies using trained tree ensembles and vector space architectures.")
st.divider() # Corrected from st.hr()

left_panel, right_panel = st.columns([1, 1], gap="large")

with left_panel:
    st.markdown("### 📝 Draft Composition Metadata")
    
    # Added a 'help' tooltip to guide production users
    caption_input = st.text_area(
        "Post Draft Copy",
        placeholder="Draft your promotional visual caption hooks, content blocks, or hashtag groupings here...",
        height=180,
        help="💡 Press Ctrl+Enter or click outside this box to instantly refresh the hashtag counter!"
    )
    
    col1, col2 = st.columns(2)
    with col2:
        computed_hashtags = extract_hashtag_count(caption_input)
        st.metric(label="Detected Dynamic Hashtags", value=computed_hashtags)
        
    with col1:
        target_hour = st.slider("Target Posting Hour (24h format)", min_value=0, max_value=23, value=18)
        target_day = st.selectbox("Scheduled Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
        
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    execute_pipeline = st.button("Run Virality Diagnostic Pipeline", type="primary", use_container_width=True)


with right_panel:
    st.markdown("### 📊 Production Inference Pipeline Outputs")
    
    if execute_pipeline:
        if not caption_input.strip():
            st.error("🚨 Active validation pipeline aborted: Draft composition text canvas cannot be blank.")
        
        elif model is None:
            # --- SIMULATION FALLBACK ENGINE ---
            st.warning("🚧 Native workspace model files missing. Executing heuristic processing module.")
            
            base_probability = 35.0
            if "growth" in caption_input.lower() or "ai" in caption_input.lower() or "tech" in caption_input.lower():
                base_probability += 30.0
            if 16 <= target_hour <= 21:
                base_probability += 15.0
            if computed_hashtags > 5:
                base_probability -= 10.0
                
            final_prob = max(min(base_probability, 98.4), 8.1)
            
            st.markdown(f"""
            <div class='metric-container'>
                <p style='color: #8B949E; margin-bottom: 4px; text-transform: uppercase; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;'>Simulated Performance Likelihood Score</p>
                <h1 style='color: #FFa257; margin: 0; font-size: 48px;'>{final_prob:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            if final_prob >= 65.0:
                st.success("🔥 Strategy Forecast: High Probability Content Index Matching Observed Pattern Profiles.")
            else:
                st.info("📉 Strategy Forecast: Standard Base Audience Coverage Window Expected.")
                
            st.markdown("#### 🏷️ Content-Aware Recommended Tags")
            mock_tags = ["#socialmedia", "#analytics", "#trendanalysis", "#growthhacking", "#marketingtips"]
            for tag in mock_tags:
                st.markdown(f"<span class='recommendation-pill'>{tag}</span>", unsafe_allow_html=True)
        
        else:
            # --- PRODUCTION PIPELINE EXECUTIONS ---
            feature_array = np.array([[
                target_hour,
                day_mapping[target_day],
                computed_hashtags,
                len(caption_input)
            ]])
            
            prediction_probabilities = model.predict_proba(feature_array)[0]
            viral_class_likelihood = float(prediction_probabilities[1] * 100)
            
            st.markdown(f"""
            <div class='metric-container'>
                <p style='color: #8B949E; margin-bottom: 4px; text-transform: uppercase; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;'>Model Confidence Probability Mapping</p>
                <h1 style='color: #2EA043; margin: 0; font-size: 48px;'>{viral_class_likelihood:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            transformed_query = tfidf_vec.transform([caption_input])
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_scores = cosine_similarity(transformed_query, corpus_embeddings).flatten()
            top_matching_indices = similarity_scores.argsort()[-3:][::-1]
            
            st.markdown("#### 🚀 Associated Keyword Tag Recommendations")
            extracted_recommendations = set()
            for lookup_index in top_matching_indices:
                matched_row_tags = str(df_reference.iloc[lookup_index]['Hashtags']).split()
                for individual_tag in matched_row_tags:
                    if len(extracted_recommendations) < 6:
                        extracted_recommendations.add(individual_tag)
                        
            for tag in extracted_recommendations:
                st.markdown(f"<span class='recommendation-pill'>{tag}</span>", unsafe_allow_html=True)
    else:
        st.info("💡 Complete input configurations inside the left panel configuration zone to run model validation sequences.")