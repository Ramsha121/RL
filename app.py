import streamlit as st
import numpy as np
import random
import re
import matplotlib.pyplot as plt

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Employment Selection System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Intelligent Employment Selection System")

st.caption(
    "AI Resume Analysis using NLP, K-Armed Bandit, "
    "TF-IDF and Gradient Ascent"
)


# ============================================================
# SKILL DICTIONARY
# ============================================================

skill_dictionary = [
    "python",
    "r",
    "sql",
    "excel",
    "tableau",
    "power bi",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai",
    "data analysis",
    "statistics",
    "nlp",
    "computer vision",
    "tensorflow",
    "keras",
    "pytorch",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "data visualization",
    "big data",
    "hadoop",
    "spark",
    "mysql",
    "postgresql",
    "mongodb",
    "scikit-learn",
    "scikit",
    "git",
    "github",
    "api",
    "dashboard",
    "analytics",
    "regression",
    "classification",
    "streamlit",
    "powerpoint",
    "ms office"
]


# ============================================================
# EDUCATION KEYWORDS
# ============================================================

education_keywords = [
    "bsc",
    "bachelor",
    "b.tech",
    "btech",
    "msc",
    "master",
    "m.tech",
    "phd",
    "computer science",
    "data science",
    "statistics",
    "mathematics",
    "engineering",
    "artificial intelligence",
    "machine learning",
    "data analytics",
    "analytics",
    "computer applications"
]


# ============================================================
# PROJECT KEYWORDS
# ============================================================

project_keywords = [
    "project",
    "developed",
    "develop",
    "built",
    "designed",
    "implemented",
    "created",
    "analyzed",
    "analysis",
    "dashboard",
    "model",
    "trained",
    "prediction",
    "classification",
    "visualization",
    "simulation",
    "deployment",
    "streamlit",
    "machine learning",
    "data analysis"
]


# ============================================================
# EXPERIENCE KEYWORDS
# ============================================================

experience_keywords = [
    "experience",
    "intern",
    "internship",
    "worked",
    "company",
    "client",
    "professional",
    "employment",
    "responsibilities",
    "job",
    "role",
    "team"
]


# ============================================================
# CERTIFICATION KEYWORDS
# ============================================================

certification_keywords = [
    "certification",
    "certified",
    "certificate",
    "nism",
    "coursera",
    "udemy",
    "google",
    "microsoft",
    "ibm",
    "deloitte",
    "accenture",
    "aws",
    "azure",
    "power bi",
    "sql",
    "python"
]


# ============================================================
# SOFT SKILLS
# ============================================================

soft_skill_keywords = [
    "communication",
    "leadership",
    "teamwork",
    "team",
    "collaboration",
    "problem solving",
    "critical thinking",
    "adaptability",
    "presentation",
    "management",
    "planning",
    "decision making",
    "creativity",
    "innovation",
    "initiative",
    "time management"
]


# ============================================================
# JOB ROLES
# ============================================================

roles = [
    {
        "name": "Business Analyst",

        "weights": np.array([
            0.35,   # Technical Skills
            0.15,   # Education
            0.20,   # Projects
            0.10,   # Experience
            0.10,   # Certifications
            0.10    # Soft Skills
        ]),

        "link":
            "https://jobs.company.com/business-analyst"
    },

    {
        "name": "Data Analyst",

        "weights": np.array([
            0.40,
            0.15,
            0.20,
            0.10,
            0.10,
            0.05
        ]),

        "link":
            "https://jobs.company.com/data-analyst"
    },

    {
        "name": "Machine Learning Intern",

        "weights": np.array([
            0.45,
            0.15,
            0.20,
            0.05,
            0.10,
            0.05
        ]),

        "link":
            "https://jobs.company.com/ml-intern"
    },

    {
        "name": "AI Intern",

        "weights": np.array([
            0.50,
            0.15,
            0.20,
            0.05,
            0.05,
            0.05
        ]),

        "link":
            "https://jobs.company.com/ai-intern"
    },

    {
        "name": "Research Analyst",

        "weights": np.array([
            0.30,
            0.20,
            0.20,
            0.10,
            0.10,
            0.10
        ]),

        "link":
            "https://jobs.company.com/research-analyst"
    }
]


# ============================================================
# JOB DESCRIPTIONS
# ============================================================

job_descriptions = {

    "Business Analyst":
        """
        business analyst business analysis
        sql excel power bi tableau
        reporting dashboard data visualization
        requirements analysis stakeholder communication
        data analytics
        """,

    "Data Analyst":
        """
        data analyst data analysis
        sql python excel
        statistics tableau power bi
        data visualization dashboard
        reporting analytics
        """,

    "Machine Learning Intern":
        """
        machine learning python
        pandas numpy scikit-learn
        regression classification
        model training artificial intelligence
        data science
        """,

    "AI Intern":
        """
        artificial intelligence
        machine learning deep learning
        python tensorflow keras
        nlp neural networks
        data science
        """,

    "Research Analyst":
        """
        research analyst statistics
        data analysis
        research methodology
        python sql
        quantitative analysis
        modeling
        """
}


# ============================================================
# REQUIRED SKILLS BY ROLE
# ============================================================

required_skills = {

    "Business Analyst": [
        "sql",
        "excel",
        "power bi",
        "tableau",
        "data analysis"
    ],

    "Data Analyst": [
        "sql",
        "python",
        "excel",
        "statistics",
        "data visualization"
    ],

    "Machine Learning Intern": [
        "python",
        "machine learning",
        "pandas",
        "numpy",
        "scikit-learn"
    ],

    "AI Intern": [
        "python",
        "machine learning",
        "deep learning",
        "nlp",
        "tensorflow"
    ],

    "Research Analyst": [
        "statistics",
        "python",
        "data analysis",
        "research",
        "sql"
    ]
}


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# KEYWORD MATCHING
# ============================================================

def get_matches(text, keywords):

    matches = []

    for keyword in keywords:

        if keyword.lower() in text:

            matches.append(keyword)

    return sorted(set(matches))


# ============================================================
# TECHNICAL SKILL SCORE
# ============================================================

def calculate_skill_score(text):

    detected = get_matches(
        text,
        skill_dictionary
    )

    skill_count = len(detected)

    # Deliberately not 100 just because many keywords
    # are detected.

    if skill_count >= 15:
        score = 95

    elif skill_count >= 12:
        score = 90

    elif skill_count >= 10:
        score = 85

    elif skill_count >= 8:
        score = 78

    elif skill_count >= 6:
        score = 70

    elif skill_count >= 4:
        score = 60

    elif skill_count >= 2:
        score = 45

    else:
        score = 25

    return score, detected


# ============================================================
# EDUCATION SCORE
# ============================================================

def calculate_education_score(text):

    matches = get_matches(
        text,
        education_keywords
    )

    score = 40

    if len(matches) >= 1:
        score += 20

    if any(
        x in text
        for x in [
            "data science",
            "computer science",
            "statistics",
            "engineering"
        ]
    ):
        score += 20

    if any(
        x in text
        for x in [
            "bsc",
            "bachelor",
            "b.tech",
            "btech",
            "msc",
            "master"
        ]
    ):
        score += 10

    if "cgpa" in text or "gpa" in text:
        score += 5

    return min(score, 100), matches


# ============================================================
# PROJECT SCORE
# ============================================================

def calculate_project_score(text):

    matches = get_matches(
        text,
        project_keywords
    )

    project_indicators = [
        "project",
        "developed",
        "built",
        "designed",
        "implemented",
        "dashboard",
        "model",
        "simulation"
    ]

    indicators = get_matches(
        text,
        project_indicators
    )

    score = 30

    score += min(
        len(matches) * 3,
        30
    )

    if len(indicators) >= 3:
        score += 15

    if len(indicators) >= 5:
        score += 15

    return min(
        round(score, 2),
        100
    ), matches


# ============================================================
# EXPERIENCE SCORE
# ============================================================

def calculate_experience_score(text):

    matches = get_matches(
        text,
        experience_keywords
    )

    if "internship" in text or "intern" in text:
        score = 70

    elif "experience" in text:
        score = 65

    elif "worked" in text:
        score = 60

    elif "project" in text:

        # Academic projects count as
        # relevant practical exposure,
        # but less than formal employment.
        score = 55

    else:
        score = 40

    if len(matches) >= 5:
        score += 10

    return min(
        score,
        100
    ), matches


# ============================================================
# CERTIFICATION SCORE
# ============================================================

def calculate_certification_score(text):

    matches = get_matches(
        text,
        certification_keywords
    )

    count = len(matches)

    if count >= 6:
        score = 95

    elif count >= 4:
        score = 85

    elif count >= 2:
        score = 75

    elif count >= 1:
        score = 60

    else:
        score = 40

    return score, matches


# ============================================================
# SOFT SKILL SCORE
# ============================================================

def calculate_soft_skill_score(text):

    matches = get_matches(
        text,
        soft_skill_keywords
    )

    count = len(matches)

    if count >= 8:
        score = 90

    elif count >= 6:
        score = 82

    elif count >= 4:
        score = 72

    elif count >= 2:
        score = 60

    elif count >= 1:
        score = 50

    else:
        score = 40

    return score, matches


# ============================================================
# ATS SCORE
# ============================================================

def calculate_ats_score(
    skill,
    education,
    projects,
    experience,
    certification,
    soft_skills
):

    score = (

        skill * 0.35

        +

        education * 0.15

        +

        projects * 0.20

        +

        experience * 0.10

        +

        certification * 0.10

        +

        soft_skills * 0.10
    )

    return round(
        min(score, 100),
        2
    )


# ============================================================
# K-ARMED BANDIT + GRADIENT ASCENT
# ============================================================

def train_bandit(
    features,
    episodes
):

    random.seed(42)
    np.random.seed(42)

    number_of_arms = len(roles)

    # --------------------------------------------------------
    # EXPLORATION PARAMETER
    # --------------------------------------------------------

    epsilon = 0.20

    # --------------------------------------------------------
    # BANDIT ARM VALUES
    # --------------------------------------------------------

    arm_values = np.zeros(
        number_of_arms
    )

    # Number of times each role is selected
    arm_counts = np.zeros(
        number_of_arms
    )

    # --------------------------------------------------------
    # GRADIENT ASCENT WEIGHTS
    # --------------------------------------------------------

    weights = np.random.rand(
        len(features)
    )

    learning_rate = 0.005

    reward_history = []

    normalized_features = (
        features / 100
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(
        episodes
    ):

        # ----------------------------------------------------
        # EPSILON-GREEDY ACTION SELECTION
        # ----------------------------------------------------

        if random.random() < epsilon:

            # Explore a random role
            arm = random.randint(
                0,
                number_of_arms - 1
            )

        else:

            # Select role with highest
            # estimated reward
            arm = int(
                np.argmax(
                    arm_values
                )
            )

        # ----------------------------------------------------
        # ROLE-SPECIFIC SCORE
        # ----------------------------------------------------

        role_weights = roles[
            arm
        ]["weights"]

        role_score = np.dot(
            role_weights,
            normalized_features
        )

        # ----------------------------------------------------
        # EXPLORATION NOISE
        # ----------------------------------------------------
        #
        # Without this, the resume features remain fixed
        # and the reward becomes almost identical at every
        # episode.
        #
        # Small noise simulates variation in candidate/job
        # interaction while keeping the reward realistic.
        # ----------------------------------------------------

        noise = np.random.normal(
            0,
            0.02
        )

        reward = (
            role_score
            +
            noise
        )

        reward = np.clip(
            reward,
            0,
            1
        )

        # ----------------------------------------------------
        # UPDATE ARM VALUE
        # ----------------------------------------------------

        arm_counts[arm] += 1

        arm_values[arm] += (

            reward
            -
            arm_values[arm]

        ) / arm_counts[arm]

        # ----------------------------------------------------
        # GRADIENT ASCENT
        # ----------------------------------------------------

        weights += (

            learning_rate
            *
            reward
            *
            normalized_features

        )

        weights = np.clip(
            weights,
            0,
            1
        )

        reward_history.append(
            reward
        )

    return (
        weights,
        arm_values,
        arm_counts,
        reward_history
    )


# ============================================================
# ROLE MATCHING
# ============================================================

def calculate_role_scores(
    features
):

    normalized_features = (
        features / 100
    )

    results = []

    for role in roles:

        score = np.dot(

            role["weights"],

            normalized_features

        )

        score *= 100

        results.append({

            "Role":
                role["name"],

            "Score":
                round(
                    float(score),
                    2
                ),

            "Link":
                role["link"]

        })

    results.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    return results


# ============================================================
# TF-IDF JOB MATCHING
# ============================================================

def tfidf_matching(
    resume_text
):

    documents = [

        resume_text

    ] + list(
        job_descriptions.values()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    similarities = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]

    results = {}

    for index, role in enumerate(
        job_descriptions
    ):

        results[role] = round(

            float(
                similarities[index]
            ) * 100,

            2
        )

    return results


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def find_skill_gaps(
    resume_text,
    role
):

    missing = []

    for skill in required_skills[
        role
    ]:

        if skill.lower() not in resume_text:

            missing.append(
                skill
            )

    return missing


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Resume Analyzer"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

st.sidebar.markdown("---")

episodes = st.sidebar.slider(
    "Bandit Training Episodes",
    min_value=50,
    max_value=1000,
    value=300,
    step=50
)


# ============================================================
# LANDING PAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a PDF resume from the sidebar "
        "to start the analysis."
    )

    st.markdown(
        """
        ### System Workflow

        Resume PDF
        ↓

        NLP Skill Extraction
        ↓

        Resume Feature Analysis
        ↓

        ATS Score
        ↓

        K-Armed Bandit
        ↓

        Job Role Recommendation
        ↓

        TF-IDF Similarity
        ↓

        Skill Gap Analysis
        """
    )

    st.stop()


# ============================================================
# READ RESUME
# ============================================================

try:

    resume_text = extract_resume_text(
        uploaded_file
    )

except Exception as e:

    st.error(
        "Unable to read the PDF."
    )

    st.exception(e)

    st.stop()


if not resume_text:

    st.error(
        "No readable text was found in this PDF."
    )

    st.stop()


# ============================================================
# 1. NLP SKILL EXTRACTION
# ============================================================

st.header(
    "1. NLP Skill Extraction"
)

skill_score, detected_skills = (
    calculate_skill_score(
        resume_text
    )
)

if detected_skills:

    st.success(
        f"{len(detected_skills)} relevant skills detected"
    )

    columns = st.columns(4)

    for i, skill in enumerate(
        detected_skills
    ):

        columns[
            i % 4
        ].markdown(
            f"`{skill}`"
        )

else:

    st.warning(
        "No recognized technical skills were detected."
    )


# ============================================================
# 2. RESUME COMPONENT SCORES
# ============================================================

st.header(
    "2. Resume Component Scores"
)

education_score, education_matches = (
    calculate_education_score(
        resume_text
    )
)

project_score, project_matches = (
    calculate_project_score(
        resume_text
    )
)

experience_score, experience_matches = (
    calculate_experience_score(
        resume_text
    )
)

certification_score, certification_matches = (
    calculate_certification_score(
        resume_text
    )
)

soft_skill_score, soft_skill_matches = (
    calculate_soft_skill_score(
        resume_text
    )
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Technical Skills",
    f"{skill_score}/100"
)

col2.metric(
    "Education",
    f"{education_score}/100"
)

col3.metric(
    "Projects",
    f"{project_score}/100"
)


col4, col5, col6 = st.columns(3)

col4.metric(
    "Experience",
    f"{experience_score}/100"
)

col5.metric(
    "Certifications",
    f"{certification_score}/100"
)

col6.metric(
    "Soft Skills",
    f"{soft_skill_score}/100"
)


# ============================================================
# 3. ATS SCORE
# ============================================================

ats_score = calculate_ats_score(

    skill_score,

    education_score,

    project_score,

    experience_score,

    certification_score,

    soft_skill_score
)


st.header(
    "3. Overall ATS Score"
)

st.progress(
    int(ats_score)
)

if ats_score >= 85:

    st.success(
        f"Excellent Resume — {ats_score}/100"
    )

elif ats_score >= 75:

    st.success(
        f"Strong Resume — {ats_score}/100"
    )

elif ats_score >= 65:

    st.warning(
        f"Good Resume, but there is room "
        f"for improvement — {ats_score}/100"
    )

elif ats_score >= 50:

    st.warning(
        f"Average Resume — {ats_score}/100"
    )

else:

    st.error(
        f"Resume needs improvement — {ats_score}/100"
    )


# ============================================================
# 4. K-ARMED BANDIT
# ============================================================

st.header(
    "4. K-Armed Bandit Learning"
)

features = np.array([

    skill_score,
    education_score,
    project_score,
    experience_score,
    certification_score,
    soft_skill_score

])


(
    learned_weights,
    arm_values,
    arm_counts,
    reward_history

) = train_bandit(

    features,
    episodes

)


col1, col2 = st.columns(2)


# ============================================================
# LEARNED WEIGHTS
# ============================================================

with col1:

    st.subheader(
        "Learned Feature Weights"
    )

    weight_table = {

        "Feature": [

            "Technical Skills",
            "Education",
            "Projects",
            "Experience",
            "Certifications",
            "Soft Skills"

        ],

        "Weight": np.round(
            learned_weights,
            4
        )

    }

    st.dataframe(
        weight_table,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# IMPROVED TRAINING GRAPH
# ============================================================

with col2:

    st.subheader(
        "Training Reward"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    # --------------------------------------------------------
    # RAW REWARD
    # --------------------------------------------------------

    ax.plot(
        reward_history,
        alpha=0.25,
        label="Raw Reward"
    )

    # --------------------------------------------------------
    # MOVING AVERAGE
    # --------------------------------------------------------

    window = 30

    if len(reward_history) >= window:

        moving_average = np.convolve(

            reward_history,

            np.ones(window) / window,

            mode="valid"

        )

        ax.plot(

            range(
                window - 1,
                len(reward_history)
            ),

            moving_average,

            linewidth=2,

            label="30-Episode Moving Average"
        )

    ax.set_title(
        "K-Armed Bandit Learning Progress"
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Reward"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    st.pyplot(
        fig
    )

    plt.close(fig)


# ============================================================
# 5. JOB ROLE RECOMMENDATION
# ============================================================

st.header(
    "5. Recommended Job Roles"
)

role_results = calculate_role_scores(
    features
)

best_role = role_results[0]


st.success(

    f"Best Match: "
    f"{best_role['Role']} "
    f"— {best_role['Score']}%"

)


for result in role_results:

    st.write(

        f"**{result['Role']}** "
        f"— {result['Score']}%"

    )

    st.progress(
        int(result["Score"])
    )


# ============================================================
# ROLE CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

role_names = [

    result["Role"]

    for result in role_results

]

role_scores = [

    result["Score"]

    for result in role_results

]

ax.bar(
    role_names,
    role_scores
)

ax.set_ylim(
    0,
    100
)

ax.set_xlabel(
    "Job Role"
)

ax.set_ylabel(
    "Match Score (%)"
)

ax.set_title(
    "Job Role Recommendation"
)

ax.tick_params(
    axis="x",
    rotation=30
)

st.pyplot(
    fig
)

plt.close(fig)


# ============================================================
# APPLY LINK
# ============================================================

st.subheader(
    "Application"
)

st.markdown(

    f"[Apply for "
    f"{best_role['Role']}]"
    f"({best_role['Link']})"

)


# ============================================================
# 6. TF-IDF
# ============================================================

st.header(
    "6. TF-IDF Job Similarity"
)

tfidf_scores = tfidf_matching(
    resume_text
)

sorted_tfidf = sorted(

    tfidf_scores.items(),

    key=lambda x: x[1],

    reverse=True

)


for role, score in sorted_tfidf:

    st.write(
        f"**{role}** — "
        f"{score}% similarity"
    )

    st.progress(
        int(min(score, 100))
    )


# ============================================================
# 7. SKILL GAP ANALYSIS
# ============================================================

st.header(
    "7. Skill Gap Analysis"
)

missing_skills = find_skill_gaps(

    resume_text,

    best_role["Role"]

)

st.write(

    f"Skills required for "
    f"**{best_role['Role']}**:"

)


if missing_skills:

    st.warning(
        "Skills that could be improved:"
    )

    for skill in missing_skills:

        st.write(
            f"• {skill}"
        )

else:

    st.success(
        "Excellent! No major skill gaps "
        "were detected for this role."
    )


# ============================================================
# 8. BANDIT STATISTICS
# ============================================================

st.header(
    "8. Bandit Statistics"
)

bandit_table = {

    "Job Role": [

        role["name"]

        for role in roles

    ],

    "Estimated Value": np.round(

        arm_values,

        4

    ),

    "Selections": (

        arm_counts.astype(int)

    )

}


st.dataframe(

    bandit_table,

    hide_index=True,

    use_container_width=True

)


# ============================================================
# 9. RESUME ANALYSIS DETAILS
# ============================================================

st.header(
    "9. Resume Analysis Details"
)


with st.expander(
    "Detected Education"
):

    if education_matches:

        for item in education_matches:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No education keywords detected."
        )


with st.expander(
    "Detected Project Indicators"
):

    if project_matches:

        for item in project_matches:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No major project indicators detected."
        )


with st.expander(
    "Detected Certifications"
):

    if certification_matches:

        for item in certification_matches:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No certification indicators detected."
        )


with st.expander(
    "Detected Soft Skills"
):

    if soft_skill_matches:

        for item in soft_skill_matches:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No major soft-skill keywords detected."
        )


# ============================================================
# 10. EXTRACTED RESUME
# ============================================================

with st.expander(
    "View Extracted Resume Text"
):

    st.write(
        resume_text
    )
