import gradio as gr
from dotenv import load_dotenv

from src.retrieval.rag import load_books, build_vector_store
from src.recommendation.recommender import retrieve_semantic_recommendations, format_authors
from src.llm.explanation import generate_explanation

load_dotenv()

books = load_books()
db_books = build_vector_store()

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


def recommend_books(query: str, category: str, tone: str) -> list[tuple[str, str]]:
    recommendations = retrieve_semantic_recommendations(db_books, books, query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        explanation = generate_explanation(query, row["description"])
        description_preview = " ".join(row["description"].split()[:30]) + "..."
        authors_str = format_authors(row["authors"])
        caption = f"{row['title']} by {authors_str}: {explanation}"
        results.append((row["large_thumbnail"], caption))

    return results


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic Book Recommender")

        with gr.Row():
            user_query = gr.Textbox(
                label="Please enter a description of a book:",
                placeholder="e.g., A story about forgiveness",
            )
            category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
            submit_button = gr.Button("Find recommendations")

        gr.Markdown("## Recommendations")
        output = gr.Gallery(label="Recommended books", columns=8, rows=2)

        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output,
        )

    return dashboard
