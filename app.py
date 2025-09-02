import os
import time
import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

import streamlit as st
from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import SecretStr
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

load_dotenv()

# Utility help functions
def now_ts():
    return time.time()


def elapsed_ms(start, end):
    return int((end - start) * 1000)


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# Core agent

class AgentTimings:
    def __init__(self):
        self.nodes = []

    def add(self, name: str, start: float, end: float):
        self.nodes.append({'node': name, 'ms': elapsed_ms(start, end)})

    def total(self) -> int:
        return sum(n['ms'] for n in self.nodes)


def validate_topic(topic: str) -> Tuple[bool, str]:
    if not topic or not topic.strip():
        return False, "Topic is empty"
    t = topic.strip()
    if len(t) < 3:
        return False, "Topic too short"
    if len(t) > 500:
        return False, "Topic too long, keep under 500 chars"
    # Basic profanity filter
    banned = ['bomb', 'hate', 'kill', 'terror', 'violence', 'abuse']
    for b in banned:
        if b in t.lower():
            return False, "Topic includes disallowed terms."
    return True, "OK"


def plan_structure(topic: str, details: str, persona: str, audience: str, n_points: int = 5) -> List[str]:
    #Return key points for post planning
    base = []
    base.append(f"Hook: Start with a surprising stat or bold statement about {topic}.")
    base.append(f"Context: Explain why {topic} matters to {audience or 'professionals'}. Use 1-2 sentences.")
    base.append(f"Strategy angles: Provide 2–3 actionable strategies or steps (practical and concise).")
    base.append(f"Proof loop: Add a short case study or personal example (metrics, outcome).")
    base.append(f"CTA: Close with a single clear CTA — invite comments, link to resource, or DM for help.")
    if details:
        base.append(f"Personal detail: weave in user's supplied detail: {details}" )
    return base[:n_points]

#Draft generator 
#Draft generator 
def simple_draft_with_llm(topic: str, plan_points: List[str], persona: str, audience: str, length_pref: str,
                          tone: str, model_settings: dict, research_snippets: List[Dict[str, Any]] = None, 
                          draft_number: int = 1, total_drafts: int = 1, writing_style: str = None) -> str:
    """Generate a unique draft using LangChain or fallback prompt template."""
    
    style_context = ""
    if writing_style:
        style_context = f"WRITING STYLE TO EMULATE: {writing_style}\n\n"
    
    research_text = "\n".join([f"- {r['title']}: {r['snippet']} ({r['link']})" for r in (research_snippets or [])])
    plan_text = '\n'.join(plan_points)
    
    prompt = f"""
You are a professional LinkedIn writer. Create a unique LinkedIn post draft following the plan below.
{style_context}
Topic: {topic}
Persona: {persona or 'N/A'}
Audience: {audience or 'N/A'}
Tone: {tone or 'neutral'}
Length: {length_pref or 'short'}
Draft: {draft_number} of {total_drafts}
Plan:
{plan_text}
Research snippets:
{research_text}
Write a single LinkedIn ready post (no hashtags, no CTA). Keep it in plain text.
Make this draft unique and different from other drafts while maintaining the same core message.
""".strip()

    if LANGCHAIN_AVAILABLE:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_settings.get("model", "models/gemini-1.5-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=model_settings.get("temperature", 0.7)
            )
            template = PromptTemplate(input_variables=["input_text"], template="{input_text}")
            chain = LLMChain(llm=llm, prompt=template)
            resp = chain.run(prompt)  # Use the prompt variable directly
            return clean_text(resp)
        except Exception as e:
            st.sidebar.warning(f"GOOGLE_API_KEY call failed: {str(e)}")
            # Fall through to deterministic generation
    
    # Fallback deterministic generation with variation based on draft number
    body = []
    if draft_number == 1:
        body.append(f"Let's talk about {topic}. Did you know that focusing on this can yield amazing results?")
    elif draft_number == 2:
        body.append(f"Many professionals overlook the power of {topic}. Here's why it matters:")
    else:
        body.append(f"Ready to master {topic}? Here's what you need to know:")
    
    body.append(plan_points[1] if len(plan_points) > 1 else f"Discussion about {topic}")
    
    if len(plan_points) > 2:
        strategies = plan_points[2:4]
        if draft_number % 2 == 0:
            body.append(' • '.join(strategies))
        else:
            body.append(' → '.join(strategies))
    
    if draft_number == 1:
        body.append('In my experience, focusing on first principles yields measurable results.')
    elif draft_number == 2:
        body.append('Ive seen teams achieve 2x results by implementing these strategies.')
    else:
        body.append('The data shows clear benefits when applying these approaches consistently.')
    
    return ' '.join(body)
#TOol for fetching hashtags
def fetch_trending_hashtags(topic: str) -> List[str]:
    if not TAVILY_AVAILABLE or not os.getenv("TAVILY_API_KEY"):
        return []
    
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query=f"LinkedIn hashtags for {topic}", max_results=3)
        
        all_text = ""
        if response and 'results' in response:
            for result in response['results']:
                all_text += result.get('content', '') + " "
        
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text.lower())
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(10) if count > 1]
        return [f"#{word}" for word in common_words[:5]]
    except Exception as e:
        st.sidebar.warning(f"Tavily API call failed: {str(e)}")
        return []

#Custom LLM-based guardrails validation to check for inappropriate content
def custom_guardrails_check(text: str, model_settings: dict) -> Tuple[bool, str]:
    
    if not LANGCHAIN_AVAILABLE:
        return False, "LLM not available for guardrails check"
    
    try:
        prompt = """
Analyze the following LinkedIn post content for inappropriate content. Check for:
1. Profanity or offensive language
2. Hate speech or discriminatory content
3. Sensitive topics (politics, religion, etc) that might be inappropriate for professional platforms
4. False claims or misinformation
5. Any content that violates LinkedIn's professional community guidelines

Content to analyze:
{content}

Respond with JSON format only:
{{
  "is_appropriate": true/false,
  "reason": "brief explanation if inappropriate, else 'appropriate'",
  "flagged_sections": ["list of specific problematic phrases if any"]
}}
""".strip()
        
        llm = ChatGoogleGenerativeAI(
            model=model_settings.get("model", "models/gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1  # Low temperature for deterministic checking
        )
        template = PromptTemplate(input_variables=["content"], template=prompt)
        chain = LLMChain(llm=llm, prompt=template)
        response = chain.run(text)
        
        # Try to parse JSON response
        try:
            result = json.loads(response)
            is_appropriate = result.get("is_appropriate", True)
            reason = result.get("reason", "appropriate")
            return not is_appropriate, reason
        except json.JSONDecodeError:
            # If JSON parsing fails, check response text
            if "inappropriate" in response.lower() or "false" in response.lower():
                return True, "Content flagged as inappropriate by guardrails"
            return False, "Guardrails check completed"
            
    except Exception as e:
        return False, f"Guardrails validation error: {str(e)}"

#Tool to add hashtag and cta
def enhance_post(
    post_text: str, 
    hashtags_on: bool, 
    cta_on: bool, 
    user_hashtags: List[str], 
    topic: str,
    writing_style: str = None
) -> Dict[str, Any]:
    hashtags = user_hashtags or []

    # Fetching Tavily suggested hashtags 
    trending_hashtags = fetch_trending_hashtags(topic)
    
    if hashtags_on:
        # Using LLM to generate context aware hashtags and combine with trending ones
        llm_hashtags = generate_llm_hashtags(post_text, topic, trending_hashtags)
        hashtags += llm_hashtags

    hashtags = list(dict.fromkeys(hashtags))  

    # Call to action
    cta = ''
    if cta_on:
        cta_options = [
            'If you found this helpful, drop a comment or DM me to discuss further.',
            'What are your thoughts on this? I\'d love to hear your experiences in the comments.',
            'Found this useful? Share it with your network who might benefit from it.',
            'Have you tried these strategies? Let me know what worked for you!'
        ]
        cta = cta_options[len(hashtags) % len(cta_options)]

    enhanced = post_text.strip()
    if cta:
        enhanced = enhanced + '\n\n' + cta
    if hashtags:
        enhanced = enhanced + '\n\n' + ' '.join(hashtags)

    return {
        'text': enhanced,
        'hashtags': hashtags,
        'cta': cta,
        'trending_insights': trending_hashtags,
        'llm_generated_hashtags': llm_hashtags if hashtags_on else []
    }

def generate_llm_hashtags(post_text: str, topic: str, trending_hashtags: List[str]) -> List[str]:
    prompt = f"""
You are a social media expert specializing in LinkedIn content. Generate 8-12 highly relevant hashtags for this post.

POST TOPIC: {topic}
POST CONTENT: {post_text}
TRENDING HASHTAGS: {', '.join(trending_hashtags)}

Generate hashtags that:
1. Are directly relevant to the post content
2. Include some of the trending hashtags when appropriate
3. Mix broad and niche tags for optimal reach
4. Are professional and appropriate for B2B/LinkedIn
5. Include 2-3 industry-specific tags

Return ONLY a comma-separated list of hashtags (no other text).
Example: #DigitalMarketing, #ContentStrategy, #B2B, #MarketingTips
"""

    try:
        if LANGCHAIN_AVAILABLE:
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3  
            )
            response = llm.invoke(prompt)
            hashtags_text = response.content.strip()
            
            # Parse the response into a list of hashtags
            generated_hashtags = [tag.strip() for tag in hashtags_text.split(',') if tag.strip()]
            return generated_hashtags[:12]  
            
    except Exception as e:
        print(f"LLM hashtag generation failed: {e}")
        # Fallback to simple extraction (if LLM fails)
        return simple_hashtag_fallback(post_text, trending_hashtags)
    
    return simple_hashtag_fallback(post_text, trending_hashtags)

def simple_hashtag_fallback(post_text: str, trending_hashtags: List[str]) -> List[str]:
    """Fallback hashtag generation if LLM fails."""
    candidates = re.findall(r"\b[A-Z]?[a-z]{3,20}\b", post_text)
    candidates = [f"#{c}" for c in candidates if len(c) > 3][:6]
    return candidates + trending_hashtags[:4] 


# Quality and engagement metrics

def quality_score_metrics(post_text: str) -> Dict[str, float]:
    """This metric computes heuristic quality metrics- Return score for readability, originality, clarity, and overall."""
    #Readability- approximate using sentence length
    sentences = re.split(r'[\.\n!\?]+', post_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_len = sum(len(s.split()) for s in sentences) / (len(sentences) or 1)
    readability = max(0, 1 - (avg_len - 20) / 50)  # favor avg sentence length ~20

    #Originality- naive uniqueness score by checking repeated phrases
    phrases = re.findall(r"\b\w{4,}\b", post_text.lower())
    unique_ratio = len(set(phrases)) / (len(phrases) or 1)
    originality = min(1.0, unique_ratio)

    #Clarity- penalize passive voice heuristically ("was", "were", "is being")
    passive_markers = sum(1 for _ in re.finditer(r'\b(was|were|is being|are being|been)\b', post_text.lower()))
    clarity = max(0.0, 1 - passive_markers * 0.1)

    # Length suitability 
    # favor medium length 60-300 words
    words = len(post_text.split())
    if words < 30:
        length_score = 0.4
    elif words <= 300:
        length_score = 1.0
    else:
        length_score = max(0.2, 1 - (words - 200)/400)

    #final weighted metric
    overall = 0.35*readability + 0.25*originality + 0.2*clarity + 0.2*length_score
    return {
        'readability': round(readability, 3),
        'originality': round(originality, 3),
        'clarity': round(clarity, 3),
        'length': round(length_score, 3),
        'overall': round(overall, 3)
    }


def engagement_simulator(post_text: str, audience: str) -> Dict[str, Any]:
    """Simulate expected engagement- likes, comments, share probability, and best audience match."""
    # Heuristics: posts with questions and CTAs get more comments; shorter posts may get more likes.
    likes = 50 + len(post_text) % 300
    comments = 5
    if '?' in post_text or 'comment' in post_text.lower() or 'dm' in post_text.lower():
        comments += 10
    # Hashtag factor
    hashtags = len(re.findall(r'#\w+', post_text))
    likes += hashtags * 5

    share_prob = min(0.4, 0.05 + (len(post_text.split()) / 1000))

    # Audience scoring
    audience_scores = {
        'founders': 0.3,
        'marketers': 0.8,
        'engineers': 0.5,
        'investors': 0.6,
        'product': 0.7
    }
    best_aud = audience or 'marketers'
    audience_fit = audience_scores.get(best_aud.lower(), 0.5)

    score = int((likes * (1 + audience_fit)) // 1)
    return {'expected_likes': int(likes), 'expected_comments': int(comments), 'share_prob': round(share_prob, 3),
            'audience_fit': round(audience_fit, 3), 'score': score}

# Analyze writing style from sample text
def analyze_writing_style(sample_text: str) -> str:
    """Analyze writing style from sample text and return a description."""
    if not sample_text or len(sample_text.strip()) < 50:
        return "No sufficient writing style sample provided."
    
    if LANGCHAIN_AVAILABLE:
        try:
            prompt = f"""
Analyze the writing style of the following text and provide a concise description of its key characteristics.
Focus on aspects like:
- Sentence structure (short/long, simple/complex)
- Tone (formal, casual, conversational, authoritative)
- Vocabulary level (simple, technical, sophisticated)
- Use of rhetorical devices (questions, metaphors, lists)
- Overall impression and style

Text to analyze:
{sample_text}

Provide a clear, concise description of the writing style in 2-3 sentences.
""".strip()
            
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1  # Low temperature for analysis
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Style analysis failed: {str(e)}"
    
    # Fallback analysis
    words = sample_text.split()
    avg_word_len = sum(len(word) for word in words) / len(words) if words else 0
    sentences = re.split(r'[.!?]+', sample_text)
    avg_sent_len = sum(len(sent.split()) for sent in sentences if sent.strip()) / len(sentences) if sentences else 0
    
    style_desc = f"Writing sample analyzed: {len(words)} words, {len(sentences)} sentences. "
    if avg_word_len > 6:
        style_desc += "Uses longer, more sophisticated vocabulary. "
    else:
        style_desc += "Uses shorter, more direct vocabulary. "
    
    if avg_sent_len > 20:
        style_desc += "Prefers longer, more complex sentence structures."
    else:
        style_desc += "Prefers shorter, more concise sentence structures."
    
    return style_desc

#streamlit code

# Set page config first
st.set_page_config(
    page_title="LinkCraft- AI based LinkedIn Post Generator", 
    layout='wide',
    initial_sidebar_state="expanded"
)

# Custom CSS for yellow theme - Lighter background
st.markdown("""
<style>
    /* Main background gradient - Lighter */
    .stApp {
        background: linear-gradient(135deg, #FFFEF0 0%, #FFF9C4 50%, #FFECB3 100%) !important;
    }
    
    /* Homepage specific styles */
    .homepage-container {
        background: linear-gradient(135deg, rgba(255, 254, 240, 0.9) 0%, rgba(255, 249, 196, 0.9) 50%, rgba(255, 236, 179, 0.9) 100%);
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem;
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
        text-align: center;
    }
    
    .main-header {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: #D35400 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.7);
    }
    
    .sub-header {
        font-size: 1.8rem !important;
        color: #B8860B !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    .description {
        font-size: 1.3rem !important;
        color: #8B4513 !important;
        text-align: center;
        margin-bottom: 3rem;
        line-height: 1.8;
        font-weight: 500;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #000000 !important;
        font-weight: bold;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 30px;
        font-size: 1.2rem;
        margin: 0 auto;
        display: block;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 165, 0, 0.6);
    }
    
    .features-grid {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 3rem;
        flex-wrap: wrap;
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #FFD700;
        text-align: center;
        min-width: 200px;
        box-shadow: 0 4px 10px rgba(255, 215, 0, 0.3);
    }
    
    /* Main app styles */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        border: 2px solid #FFD700;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #FFFEF0 0%, #FFF9C4 100%) !important;
        border-right: 3px solid #FFD700;
    }
    
    /* Content containers with slight transparency */
    .stForm {
        background-color: rgba(255, 248, 220, 0.9) !important;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #FFD700;
    }
    
    .stContainer {
        background-color: rgba(255, 248, 220, 0.9) !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #FFD700;
        margin-bottom: 1rem;
    }
    
    /* Input fields styling */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background-color: rgba(255, 248, 220, 0.8) !important;
        border: 2px solid #FFD700 !important;
        border-radius: 8px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #D35400 !important;
        border-bottom: 2px solid #FFD700;
        padding-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(255, 248, 220, 0.9) !important;
        border: 2px solid #FFD700 !important;
        border-radius: 10px;
    }
    
    /* Writing style section */
    .writing-style-box {
        background: linear-gradient(135deg, #FFFEF0 0%, #FFF9C4 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #FFD700;
        margin-bottom: 1rem;
    }
    
    /* Copy button styling */
    .copy-button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
        color: white !important;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Compact buttons for draft counter */
    .compact-button {
        padding: 0.25rem 0.5rem !important;
        font-size: 0.9rem !important;
        height: auto !important;
    }
    .number-display {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
        padding: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation and writing style
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'writing_style' not in st.session_state:
    st.session_state.writing_style = None

if 'writing_style_analyzed' not in st.session_state:
    st.session_state.writing_style_analyzed = ""

if 'num_posts' not in st.session_state:
    st.session_state.num_posts = 4

if 'final_drafts' not in st.session_state:
    st.session_state.final_drafts = None

if 'processing' not in st.session_state:
    st.session_state.processing = False

# Homepage
if st.session_state.page == 'home':
    st.markdown("""
    <div class="homepage-container">
        <div class="main-header">LinkCraft</div>
        <div class="sub-header">AI-Powered LinkedIn Post Generator</div>
        <div class="description">
            Transform your ideas into engaging LinkedIn content with our intelligent AI assistant.<br>
            Generate multiple professional posts tailored to your brand voice and audience.
        </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button('🚀 Start Creating Posts', use_container_width=True):
            st.session_state.page = 'main'
            st.rerun()
    
    # Features grid
    st.markdown("""
    <div class="features-grid">
        <div class="feature-item">
            <h3>✨ Multiple Drafts</h3>
            <p>Generate several variations to choose from</p>
        </div>
        <div class="feature-item">
            <h3>🎯 Custom Tone</h3>
            <p>Tailor content to your preferred style</p>
        </div>
        <div class="feature-item">
            <h3>📊 Quality Scoring</h3>
            <p>Get insights on post effectiveness</p>
        </div>
        <div class="feature-item">
            <h3>🔍 Research Integration</h3>
            <p>Enhance with relevant information</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Main application page
if st.session_state.page == 'main':
    # Add a button to go back to homepage
    if st.sidebar.button('🏠 Back to Home'):
        st.session_state.page = 'home'
        st.session_state.final_drafts = None
        st.session_state.processing = False
        st.rerun()

    st.title("LinkCraft")
    st.markdown("Generate multiple LinkedIn ready post drafts from a topic using an agentic pipeline")

    # Sidebar settings with yellow theme
    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox("Model", options=['gemini-1.5-flash'], index=0)
    creativity = st.sidebar.slider("🎨 Creativity (temperature)", 0.0, 1.0, 0.7)
    use_guardrails = st.sidebar.checkbox("🛡️ Enable custom guardrails (LLM-based)", value=True)
    use_tavily = st.sidebar.checkbox("🔍 Enable Tavily (if installed)", value=False)
    add_hashtags = st.sidebar.checkbox("🏷️ Auto-add hashtags in enhance step", value=True)
    add_cta = st.sidebar.checkbox("📢 Append CTA", value=True)
    show_debug = st.sidebar.checkbox("🐛 Show debug/diagnostics info", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔑 Model & Keys**: Set GOOGLE_API_KEY and (optionally) TAVILY_API_KEY in your environment.")
    if not LANGCHAIN_AVAILABLE:
        st.sidebar.warning("LangChain not available- app will use fallback prompt generator. Install langchain for best results.")
    if use_tavily and not TAVILY_AVAILABLE:
        st.sidebar.warning("Tavily not installed- toggle ignored.")

    # A Check for required API keys
    if not os.getenv("GOOGLE_API_KEY"):
        st.sidebar.error("GOOGLE_API_KEY environment variable is required!")
    if use_tavily and not os.getenv("TAVILY_API_KEY"):
        st.sidebar.warning("TAVILY_API_KEY not set - Tavily features disabled.")
        use_tavily = False

    # Writing Style Section
    st.sidebar.header("✍️ Writing Style")
    with st.sidebar.expander("Add Your Writing Style"):
        writing_sample = st.text_area(
            "Paste your writing sample (LinkedIn posts, articles, etc.)",
            height=150,
            help="This helps the AI emulate your unique writing style"
        )
        
        if st.button("Analyze Writing Style"):
            if writing_sample and len(writing_sample.strip()) > 50:
                with st.spinner("Analyzing your writing style..."):
                    st.session_state.writing_style = writing_sample
                    st.session_state.writing_style_analyzed = analyze_writing_style(writing_sample)
                    st.success("Writing style analyzed!")
            else:
                st.warning("Please provide a longer writing sample (at least 50 characters)")
        
        if st.session_state.writing_style_analyzed:
            st.markdown("**Style Analysis:**")
            st.info(st.session_state.writing_style_analyzed)
            
            if st.button("Clear Writing Style"):
                st.session_state.writing_style = None
                st.session_state.writing_style_analyzed = ""
                st.rerun()

    # Main input area
    st.subheader('📝 Input')

    # Number of drafts with increase/decrease buttons
    st.markdown("**Number of drafts:**")

    # Use more balanced column ratios
    col1, col2, col3 = st.columns([1, 0.8, 1])

    with col1:
        if st.button("➖ Decrease", key="decrease_drafts", use_container_width=True, 
                    help="Decrease number of drafts (min: 3)"):
            st.session_state.num_posts = max(3, st.session_state.num_posts - 1)
            st.rerun()

    with col2:
        st.markdown(f"<div class='number-display'>{st.session_state.num_posts}</div>", 
                unsafe_allow_html=True)

    with col3:
        if st.button("➕ Increase", key="increase_drafts", use_container_width=True,
                    help="Increase number of drafts (max: 8)"):
            st.session_state.num_posts = min(8, st.session_state.num_posts + 1)
            st.rerun()

    # Add a small spacer
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Dynamic tone inputs based on number of drafts
    st.markdown("**🎭 Tone/style for each draft:**")
    tone_options = ['informative', 'conversational', 'bold', 'inspirational', 'witty', 'professional', 'casual', 'authoritative']
    selected_tones = []

    for i in range(st.session_state.num_posts):
        tone = st.selectbox(
            f'Tone for draft {i+1}', 
            options=tone_options,
            index=i % len(tone_options),
            key=f'tone_{i}'
        )
        selected_tones.append(tone)

    # Now the rest of the form
    with st.form('main_form'):
        topic = st.text_input('📌 Topic', placeholder='e.g., IIM Entrance exam tips', max_chars=200)
        details = st.text_area('📋 Details (optional) — examples, numbers, personal notes', height=80)
        persona = st.text_input('👤 Persona (optional) — e.g., "Founder, Head of Growth"')
        audience = st.text_input('🎯 Target audience (optional) — e.g., marketers, founders, engineers')
        language = st.selectbox('🌐 Language', ['English', 'Hindi', 'Spanish'], index=0)
        length_pref = st.selectbox('📏 Post length', ['short (<=60 words)', 'medium (60-300 words)', 'long (>300 words)'], index=1)
        
        user_hashtags = st.text_input('🏷️ Your hashtags (optional, comma separated) — e.g, #product #growth')

        # Loading button for form submission
        submitted = st.form_submit_button('✨ Generate Posts', use_container_width=True)

    # Handle form submission with loading state
    if submitted and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.final_drafts = None
        st.rerun()

    # Show loading spinner and process in background
    if st.session_state.processing:
        with st.spinner('🚀 Generating your LinkedIn posts... This may take a few moments.'):
            try:
                # Your existing processing code here...
                start_total = now_ts()
                agent_timings = AgentTimings()

                #validation
                s = now_ts()
                valid, vmsg = validate_topic(topic)
                e = now_ts(); agent_timings.add('validate', s, e)

                if not valid:
                    st.error(f'Validation failed: {vmsg}')
                    st.session_state.processing = False
                    st.stop()

                # Plan
                s = now_ts()
                plan_points = plan_structure(topic, details, persona, audience, n_points=6)
                e = now_ts(); agent_timings.add('plan', s, e)

                # Research 
                research_results = []

                # Drafts
                s = now_ts()
                model_settings = {'temperature': creativity, 'model': model_choice}
                raw_drafts = []
                user_hashtags_list = [h.strip() for h in (user_hashtags.split(',') if user_hashtags else []) if h.strip()]

                for i in range(st.session_state.num_posts):
                    tone_for_this = selected_tones[i] if i < len(selected_tones) else 'informative'
                    try:
                        draft_text = simple_draft_with_llm(
                            topic, 
                            plan_points, 
                            persona, 
                            audience, 
                            length_pref, 
                            tone_for_this, 
                            model_settings, 
                            research_results,
                            i + 1,  # draft number
                            st.session_state.num_posts,  # total drafts
                            st.session_state.writing_style  # writing style context
                        )
                        raw_drafts.append({'text': draft_text, 'tone': tone_for_this})
                    except Exception as e:
                        st.error(f"Failed to generate draft {i+1}: {str(e)}")
                        # Continue with other drafts even if one fails
                        continue

                e = now_ts(); agent_timings.add('draft', s, e)

                # Enhance (hashtags + CTA)
                s = now_ts()
                enhanced_drafts = []
                for d in raw_drafts:
                    try:
                        enhanced = enhance_post(
                            d['text'], 
                            add_hashtags, 
                            add_cta, 
                            user_hashtags_list, 
                            topic,
                            st.session_state.writing_style  # writing style context
                        )
                        enhanced_drafts.append({'base': d['text'], 'tone': d['tone'], 'enhanced': enhanced['text'], 'hashtags': enhanced['hashtags'], 'cta': enhanced['cta']})
                    except Exception as e:
                        st.error(f"Failed to enhance draft: {str(e)}")
                        # Fallback: use original text without enhancement
                        enhanced_drafts.append({'base': d['text'], 'tone': d['tone'], 'enhanced': d['text'], 'hashtags': [], 'cta': ''})
                        
                e = now_ts(); agent_timings.add('enhance', s, e)

                # Quality checks and scoring
                s = now_ts()
                final_drafts = []
                seen_texts = set()
                for d in enhanced_drafts:
                    q = quality_score_metrics(d['enhanced'])
                    eng = engagement_simulator(d['enhanced'], audience)
                    
                    # Improved duplicate detection using fuzzy matching
                    text_lower = d['enhanced'].lower()
                    dedup = any(text_lower in seen_text for seen_text in seen_texts)
                    if not dedup:
                        seen_texts.add(text_lower)
                    
                    guardrail_flag = False
                    guardrail_msg = ''
                    if use_guardrails:
                        guardrail_flag, guardrail_msg = custom_guardrails_check(d['enhanced'], model_settings)

                    # Tavily integration
                    tavily_info = None
                    if use_tavily and TAVILY_AVAILABLE:
                        try:
                            tavily_info = {'topic': topic, 'hashtags': fetch_trending_hashtags(topic)}
                        except Exception as e:
                            tavily_info = {'error': str(e)}

                    final_drafts.append({
                        'enhanced': d['enhanced'],
                        'hashtags': d['hashtags'],
                        'cta': d['cta'],
                        'tone': d['tone'],
                        'quality': q,
                        'engagement': eng,
                        'dedup': dedup,
                        'guardrail': {'flag': guardrail_flag, 'msg': guardrail_msg},
                        'tavily': tavily_info
                    })
                e = now_ts(); agent_timings.add('quality', s, e)

                end_total = now_ts(); agent_timings.add('total', start_total, end_total)

                # Store the results
                st.session_state.final_drafts = final_drafts
                st.session_state.agent_timings = agent_timings
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
            finally:
                st.session_state.processing = False
                st.rerun()

    # Display results after loading completes
    if st.session_state.final_drafts and not st.session_state.processing:
        # Presenting outputs
        st.header('📄 Generated Posts')

        for idx, doc in enumerate(st.session_state.final_drafts, start=1):
            with st.container():
                st.markdown(f"### Post {idx} — Tone: {doc['tone']} {'(DUPLICATE)' if doc['dedup'] else ''}")
                st.write('---')
                # two-column layout: post and meta
                c1, c2 = st.columns([3,1])
                with c1:
                    # Create a text area for better copy functionality
                    post_text = st.text_area(
                        f"Post {idx} Content",
                        value=doc['enhanced'],
                        height=200,
                        key=f"post_{idx}_content",
                        label_visibility="collapsed"
                    )
                    
                    # Fixed copy button using JavaScript
                    if st.button('📋 Copy to Clipboard', key=f'copy_{idx}', use_container_width=True):
                        js_code = f"""
                        <script>
                        function copyToClipboard() {{
                            const textArea = document.querySelector('textarea[data-testid="stTextArea"]');
                            textArea.select();
                            document.execCommand('copy');
                            alert('Copied to clipboard!');
                        }}
                        copyToClipboard();
                        </script>
                        """
                        st.components.v1.html(js_code, height=0)
                    
                with c2:
                    st.markdown('**📊 Quality metrics**')
                    st.json(doc['quality'])
                    st.markdown('**📈 Engagement sim**')
                    st.json(doc['engagement'])
                    st.markdown('**🏷️ Hashtags**')
                    st.write(', '.join(doc['hashtags']))
                    if doc['guardrail']['flag']:
                        st.error(f"🛡️ Guardrail flagged: {doc['guardrail']['msg']}")

                st.markdown('---')

        # Diagnostics
        st.sidebar.header('🔧 Diagnostics')
        st.sidebar.write('Model: ' + model_choice)
        st.sidebar.write('Guardrails: ' + ('on' if use_guardrails else 'off'))
        
        if st.session_state.writing_style:
            st.sidebar.write('Writing Style: ✅ Enabled')

        if show_debug:
            st.subheader('⏱️ Agent timings (ms per node)')
            for node in st.session_state.agent_timings.nodes:
                st.write(f"{node['node']}: {node['ms']} ms")
            st.write('Total runtime (ms):', st.session_state.agent_timings.total())

        # Health endpoint area
        st.markdown('---')
        st.write('Status: ✅ 200 OK')
        st.write('💡 Tips: For best results add short relevant details and sample tone variants.')
        
        # Add a button to generate new posts
        if st.button('🔄 Generate New Posts', use_container_width=True):
            st.session_state.final_drafts = None
            st.rerun()
            
    elif not st.session_state.processing:
        st.info('Fill the form and click Generate to create LinkedIn post drafts.')