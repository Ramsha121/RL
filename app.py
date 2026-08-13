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
st.write(
    "AI-powered Resume Analysis using NLP, K-Armed Bandit, "
    "Gradient Ascent and TF-IDF"
)


# ============================================================
# SKILL DATABASE
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
    "scikit",
    "git",
    "github",
    "api",
    "dashboard",
    "analytics",
    "regression",
    "classification"
]


# ============================================================
# EDUCATION KEYWORDS
# ============================================================

education_keywords = [
    "bsc",
    "bachelor",
    "msc",
    "master",
    "phd",
    "computer science",
    "data science",
    "statistics",
    "mathematics",
    "engineering",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "analytics",
    "data analytics",
    "big data",
    "cloud computing",
    "algorithms",
    "data structures",
    "probability",
    "linear algebra",
    "calculus",
    "project",
    "research"
]


# ============================================================
# EXPERIENCE KEYWORDS
# ============================================================

experience_keywords = [
    "intern",
    "internship",
    "project",
    "research",
    "experience",
    "worked",
    "company",
    "client",
    "team project",
    "data analysis",
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "research assistant",
    "developed",
    "implemented",
    "designed",
    "built",
    "trained model",
    "analyzed data",
    "created dashboard"
]


# ============================================================
# CULTURE / SOFT SKILLS
# ============================================================

culture_keywords = [
    "team",
    "leadership",
    "communication",
    "presentation",
    "management",
    "problem solving",
    "adaptability",
    "collaboration",
    "teamwork",
    "critical thinking",
    "decision making",
    "time management",
    "planning",
    "creativity",
    "innovation",
    "initiative"
]


# ============================================================
# JOB ROLES
# ============================================================

roles = [
    {
        "name": "Business Analyst",
        "weights": np.array([0.4, 0.3, 0.2, 0.1]),
        "link": "https://jobs.company.com/business-analyst"
    },

    {
        "name": "Machine Learning Intern",
        "weights": np.array([0.5, 0.2, 0.2, 0.1]),
        "link": "https://jobs.company.com/ml-intern"
    },

    {
        "name": "Data Analyst",
        "weights": np.array([0.45, 0.25, 0.2, 0.1]),
        "link": "https://jobs.company.com/data-analyst"
    },

    {
        "name": "AI Intern",
        "weights": np.array([0.55, 0.15, 0.2, 0.1]),
        "link": "https://jobs.company.com/ai-intern"
    },

    {
        "name": "Research Analyst",
        "weights": np.array([0.3, 0.3, 0.25, 0.15]),
        "link": "https://jobs.company.com/research-analyst"
    }
]


# ============================================================
# JOB DESCRIPTIONS FOR TF-IDF
# ============================================================

job_descriptions = {

    "Business Analyst":
        "business analysis sql excel dashboard reporting "
        "data visualization data analytics",

    "Machine Learning Intern":
        "machine learning python deep learning pandas numpy "
        "model training artificial intelligence",

    "Data Analyst":
        "data analysis sql python statistics tableau "
        "excel visualization reporting",

    "AI Intern":
        "artificial intelligence neural networks deep learning "
        "nlp machine learning python",

    "Research Analyst":
        "research statistics data analysis research methodology "
        "modeling probability"
}


# ============================================================
# REQUIRED SKILLS FOR EACH ROLE
# ============================================================

required_skills = {

    "Business Analyst": [
        "sql",
        "excel",
        "data analysis",
        "dashboard",
        "tableau"
    ],

    "Machine Learning Intern": [
        "python",
        "machine learning",
        "pandas",
        "numpy",
        "scikit"
    ],

    "Data Analyst": [
        "sql",
        "python",
        "statistics",
        "tableau",
        "excel"
    ],

    "AI Intern": [
        "python",
        "deep learning",
        "machine learning",
        "nlp"
    ],

    "Research Analyst": [
        "statistics",
        "research",
        "data analysis",
        "modeling"
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
# NLP SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    detected_skills = []

    for skill in skill_dictionary:

        if skill in text:

            detected_skills.append(skill)

    return sorted(set(detected_skills))


# ============================================================
# FEATURE SCORING
# ============================================================

def calculate_feature_score(text, keywords):

    matches = 0

    for keyword in keywords:

        if keyword in text:

            matches += 1

    score = (
        matches /
        len(keywords)
    ) * 10

    return round(score, 2)


# ============================================================
# ATS SCORE
# ============================================================

def calculate_ats_score(features):

    skill = features[0]
    experience = features[1]
    education = features[2]
    culture = features[3]

    score = (
        skill * 3 +
        experience * 3 +
        education * 2 +
        culture * 2
    )

    return round(
        min(score, 100),
        2
    )


# ============================================================
# K-ARMED BANDIT + GRADIENT ASCENT
# ============================================================

def train_bandit(features, episodes):

    random.seed(42)
    np.random.seed(42)

    number_of_arms = len(roles)

    epsilon = 0.2

    arm_values = np.zeros(
        number_of_arms
    )

    arm_counts = np.zeros(
        number_of_arms
    )

    # Initial random weights
    weights = np.random.rand(4)

    learning_rate = 0.01

    reward_history = []

    for episode in range(episodes):

        # ----------------------------------------------------
        # EPSILON GREEDY
        # ----------------------------------------------------

        if random.random() < epsilon:

            arm = random.randint(
                0,
                number_of_arms - 1
            )

        else:

            arm = int(
                np.argmax(
                    arm_values
                )
            )

        # ----------------------------------------------------
        # CALCULATE SCORE
        # ----------------------------------------------------

        score = np.dot(
            weights,
            features
        ) / 10

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward = min(
            0.5 * score
            +
            0.3 * (features[1] / 10)
            +
            0.2 * (features[3] / 10),

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
            features
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

def calculate_role_scores(features):

    results = []

    for role in roles:

        score = np.dot(
            role["weights"],
            features
        ) / 10

        score = min(
            score * 100,
            100
        )

        results.append({

            "Role": role["name"],

            "Score": round(
                float(score),
                2
            ),

            "Link": role["link"]

        })

    results.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    return results


# ============================================================
# TF-IDF MATCHING
# ============================================================

def tfidf_matching(resume_text):

    documents = [
        resume_text
    ] + list(
        job_descriptions.values()
    )

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        matrix[0],
        matrix[1:]
    )

    scores = {}

    for index, role in enumerate(
        job_descriptions
    ):

        scores[role] = round(
            float(
                similarity[0][index]
            ) * 100,
            2
        )

    return scores


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def find_skill_gaps(
    resume_text,
    role
):

    missing_skills = []

    for skill in required_skills[role]:

        if skill not in resume_text:

            missing_skills.append(
                skill
            )

    return missing_skills


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Resume Upload"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

st.sidebar.markdown("---")

episodes = st.sidebar.slider(
    "Training Episodes",
    min_value=50,
    max_value=1000,
    value=300,
    step=50
)


# ============================================================
# APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a resume PDF from the sidebar."
    )

    st.markdown(
        """
        ### System Workflow

        **Resume PDF**

        ↓

        **Text Extraction**

        ↓

        **NLP Skill Extraction**

        ↓

        **Feature Scoring**

        ↓

        **ATS Score**

        ↓

        **K-Armed Bandit**

        ↓

        **Gradient Ascent**

        ↓

        **Job Recommendation**

        ↓

        **TF-IDF Similarity**

        ↓

        **Skill Gap Analysis**
        """
    )

else:

    try:

        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        resume_text = extract_resume_text(
            uploaded_file
        )

        if not resume_text:

            st.error(
                "No readable text found in the PDF."
            )

            st.stop()


        # ====================================================
        # SECTION 1 - NLP
        # ====================================================

        st.header(
            "1. NLP Skill Extraction"
        )

        detected_skills = extract_skills(
            resume_text
        )

        if detected_skills:

            st.success(
                f"{len(detected_skills)} skills detected"
            )

            cols = st.columns(4)

            for index, skill in enumerate(
                detected_skills
            ):

                cols[index % 4].markdown(
                    f"`{skill}`"
                )

        else:

            st.warning(
                "No matching skills detected."
            )


        # ====================================================
        # SECTION 2 - FEATURE SCORING
        # ====================================================

        st.header(
            "2. Resume Feature Scores"
        )

        skill_score = calculate_feature_score(
            resume_text,
            skill_dictionary
        )

        experience_score = calculate_feature_score(
            resume_text,
            experience_keywords
        )

        education_score = calculate_feature_score(
            resume_text,
            education_keywords
        )

        culture_score = calculate_feature_score(
            resume_text,
            culture_keywords
        )

        features = np.array([

            skill_score,
            experience_score,
            education_score,
            culture_score

        ])


        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Skills",
            f"{skill_score}/10"
        )

        col2.metric(
            "Experience",
            f"{experience_score}/10"
        )

        col3.metric(
            "Education",
            f"{education_score}/10"
        )

        col4.metric(
            "Culture Fit",
            f"{culture_score}/10"
        )


        # ====================================================
        # SECTION 3 - ATS
        # ====================================================

        st.header(
            "3. ATS Score"
        )

        ats_score = calculate_ats_score(
            features
        )

        st.progress(
            int(ats_score)
        )

        st.metric(
            "Overall ATS Score",
            f"{ats_score}/100"
        )


        # ====================================================
        # SECTION 4 - BANDIT
        # ====================================================

        st.header(
            "4. K-Armed Bandit Training"
        )

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


        # ----------------------------------------------------
        # LEARNED WEIGHTS
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "Learned Feature Weights"
            )

            weight_data = {

                "Feature": [

                    "Skills",
                    "Experience",
                    "Education",
                    "Culture Fit"

                ],

                "Weight": np.round(
                    learned_weights,
                    4
                )

            }

            st.dataframe(
                weight_data,
                hide_index=True,
                use_container_width=True
            )


        # ----------------------------------------------------
        # REWARD GRAPH
        # ----------------------------------------------------

        with col2:

            fig, ax = plt.subplots()

            ax.plot(
                reward_history
            )

            ax.set_title(
                "Reward During Training"
            )

            ax.set_xlabel(
                "Episode"
            )

            ax.set_ylabel(
                "Reward"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            st.pyplot(fig)

            plt.close(fig)


        # ====================================================
        # SECTION 5 - ROLE RECOMMENDATION
        # ====================================================

        st.header(
            "5. Recommended Job Roles"
        )

        role_results = calculate_role_scores(
            features
        )

        best_role = role_results[0]

        st.success(
            f"Best Match: {best_role['Role']} "
            f"({best_role['Score']}%)"
        )


        for result in role_results:

            st.write(
                f"**{result['Role']}** — "
                f"{result['Score']}%"
            )

            st.progress(
                int(result["Score"])
            )


        # ====================================================
        # ROLE CHART
        # ====================================================

        role_names = [

            result["Role"]

            for result in role_results

        ]

        role_scores = [

            result["Score"]

            for result in role_results

        ]


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            role_names,
            role_scores
        )

        ax.set_title(
            "Job Role Recommendation"
        )

        ax.set_xlabel(
            "Job Role"
        )

        ax.set_ylabel(
            "Match Score (%)"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # APPLY LINK
        # ====================================================

        st.subheader(
            "Apply"
        )

        st.markdown(
            f"[Apply for {best_role['Role']}]"
            f"({best_role['Link']})"
        )


        # ====================================================
        # SECTION 6 - TF-IDF
        # ====================================================

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


        # ====================================================
        # SECTION 7 - SKILL GAP
        # ====================================================

        st.header(
            "7. Skill Gap Analysis"
        )

        missing_skills = find_skill_gaps(

            resume_text,

            best_role["Role"]

        )


        st.write(
            f"Recommended Role: "
            f"**{best_role['Role']}**"
        )


        if missing_skills:

            st.warning(
                "Skills you should improve:"
            )

            for skill in missing_skills:

                st.write(
                    f"• {skill}"
                )

        else:

            st.success(
                "No major skill gaps detected."
            )


        # ====================================================
        # SECTION 8 - BANDIT STATISTICS
        # ====================================================

        st.header(
            "8. Bandit Statistics"
        )

        bandit_data = {

            "Role": [
                role["name"]
                for role in roles
            ],

            "Estimated Value":
                np.round(
                    arm_values,
                    4
                ),

            "Selections":
                arm_counts.astype(int)

        }

        st.dataframe(
            bandit_data,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # RESUME TEXT
        # ====================================================

        with st.expander(
            "View Extracted Resume Text"
        ):

            st.write(
                resume_text
            )


    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the resume."
        )

        st.exception(e)
