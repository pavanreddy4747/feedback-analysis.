# ============================================================
#          AI FEEDBACK ANALYSIS SYSTEM
#          Author  : Your Name
#          Project : GitHub First Project
#          Language: Python
# ============================================================

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from collections import Counter
import re
import os

# ---- Download required NLTK packages (runs only once) ----
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords',    quiet=True)
nltk.download('punkt',        quiet=True)

from nltk.corpus import stopwords


# ==============================================================
#                    FEEDBACK ANALYZER CLASS
# ==============================================================

class FeedbackAnalyzer:
    """
    A complete AI-powered Feedback Analysis tool.
    - Loads feedback from CSV or uses built-in sample data
    - Performs sentiment analysis (Positive / Negative / Neutral)
    - Generates charts and word clouds
    - Exports results to CSV
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()   # VADER Sentiment Model
        self.df  = None                            # Will hold the DataFrame

    # ----------------------------------------------------------
    # 1. LOAD DATA
    # ----------------------------------------------------------

    def load_data(self, filepath):
        """Load your own feedback CSV file.
        The CSV must have at least a 'feedback' column.
        Optional columns: 'id', 'category'
        """
        self.df = pd.read_csv(filepath)
        print(f"✅  Loaded {len(self.df)} feedback entries from '{filepath}'")
        return self.df

    def load_sample_data(self):
        """Use built-in sample data if you don't have a CSV yet."""
        sample = {
            'id': range(1, 21),
            'feedback': [
                "The product is absolutely amazing! Best purchase I've ever made.",
                "Very disappointed. The quality is terrible and not worth the price.",
                "It is okay, nothing special. Does what it is supposed to do.",
                "Excellent customer service! The team was very helpful and responsive.",
                "The app crashes frequently. Very frustrating experience overall.",
                "Great value for money. I am really happy with my purchase.",
                "The delivery was extremely slow. Took almost 3 weeks to arrive.",
                "Outstanding quality! It exceeded my expectations completely.",
                "Not satisfied at all. The product does not match the description.",
                "Pretty good overall. Minor issues but nothing too major.",
                "Terrible customer support. Nobody ever responds to emails.",
                "Love the new features! The latest update made everything better.",
                "Average product. Could be improved with better packaging.",
                "Absolutely love it! Will definitely recommend to all my friends.",
                "The website is confusing and very hard to navigate.",
                "Quick delivery and the product works perfectly. Very happy!",
                "Quality has gone down compared to before. Very disappointed.",
                "Neutral experience. Neither good nor bad, just average.",
                "Fantastic service! The best experience I have had in years.",
                "The instructions were unclear and setup was extremely difficult."
            ],
            'category': [
                'Product', 'Product', 'Product', 'Service', 'App',
                'Product', 'Delivery', 'Product', 'Product', 'Product',
                'Service', 'App', 'Product', 'Product', 'Website',
                'Delivery', 'Product', 'General', 'Service', 'Product'
            ]
        }
        self.df = pd.DataFrame(sample)
        print(f"✅  Loaded {len(self.df)} built-in sample feedback entries")
        return self.df

    # ----------------------------------------------------------
    # 2. SENTIMENT ANALYSIS
    # ----------------------------------------------------------

    def analyze_sentiment(self):
        """
        Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
        to score every feedback entry and classify it as:
          Positive  → compound score >= 0.05
          Negative  → compound score <= -0.05
          Neutral   → everything in between
        """
        if self.df is None:
            print("❌  No data found. Call load_data() or load_sample_data() first.")
            return

        print("\n⏳  Running sentiment analysis ...")

        # Calculate VADER scores
        self.df['compound_score']  = self.df['feedback'].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound'])
        self.df['positive_score']  = self.df['feedback'].apply(
            lambda x: self.sia.polarity_scores(str(x))['pos'])
        self.df['negative_score']  = self.df['feedback'].apply(
            lambda x: self.sia.polarity_scores(str(x))['neg'])
        self.df['neutral_score']   = self.df['feedback'].apply(
            lambda x: self.sia.polarity_scores(str(x))['neu'])

        # Assign final sentiment label
        self.df['sentiment'] = self.df['compound_score'].apply(self._label)

        print("\n📋  DETAILED RESULTS")
        print("-" * 70)
        for _, row in self.df.iterrows():
            emoji = "✅" if row['sentiment'] == 'Positive' \
                    else ("❌" if row['sentiment'] == 'Negative' else "➖")
            print(f"{emoji} [{row['sentiment']:<8}]  Score: {row['compound_score']:+.3f}  |  "
                  f"{str(row['feedback'])[:60]}...")
        return self.df

    def _label(self, score):
        """Convert a numeric compound score to a text label."""
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # ----------------------------------------------------------
    # 3. SUMMARY STATISTICS
    # ----------------------------------------------------------

    def generate_summary(self):
        """Print a clean summary of the analysis results."""
        if 'sentiment' not in self.df.columns:
            print("❌  Run analyze_sentiment() first.")
            return

        total  = len(self.df)
        counts = self.df['sentiment'].value_counts()
        pos    = counts.get('Positive', 0)
        neg    = counts.get('Negative', 0)
        neu    = counts.get('Neutral',  0)
        avg    = self.df['compound_score'].mean()

        print("\n" + "=" * 52)
        print("       📊  AI FEEDBACK ANALYSIS — SUMMARY")
        print("=" * 52)
        print(f"  Total Feedbacks Analyzed  :  {total}")
        print(f"  ✅  Positive              :  {pos}  ({pos/total*100:.1f}%)")
        print(f"  ❌  Negative              :  {neg}  ({neg/total*100:.1f}%)")
        print(f"  ➖  Neutral               :  {neu}  ({neu/total*100:.1f}%)")
        print(f"  📈  Average Sentiment Score:  {avg:+.3f}")
        print("=" * 52)

        if 'category' in self.df.columns:
            print("\n📂  Sentiment breakdown by Category:")
            breakdown = self.df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            print(breakdown.to_string())
            print()

    # ----------------------------------------------------------
    # 4. VISUALIZATIONS
    # ----------------------------------------------------------

    # Color palette used throughout all charts
    COLORS = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}

    def visualize(self):
        """Generate a 2×2 dashboard of charts and save it."""
        if 'sentiment' not in self.df.columns:
            print("❌  Run analyze_sentiment() first.")
            return

        os.makedirs('output', exist_ok=True)
        counts = self.df['sentiment'].value_counts()
        c      = [self.COLORS.get(s, '#95a5a6') for s in counts.index]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🤖  AI Feedback Analysis Dashboard', fontsize=17,
                     fontweight='bold', y=1.01)

        # ── Chart 1 : Pie chart ──────────────────────────────
        axes[0, 0].pie(counts.values, labels=counts.index, colors=c,
                       autopct='%1.1f%%', startangle=90, shadow=True)
        axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')

        # ── Chart 2 : Bar chart ──────────────────────────────
        bars = axes[0, 1].bar(counts.index, counts.values, color=c, edgecolor='black')
        axes[0, 1].set_title('Feedback Count by Sentiment', fontweight='bold')
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Count')
        for bar, val in zip(bars, counts.values):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                            val + 0.1, str(val), ha='center', fontweight='bold')

        # ── Chart 3 : Compound score histogram ───────────────
        axes[1, 0].hist(self.df['compound_score'], bins=10,
                        color='#9b59b6', edgecolor='black', alpha=0.75)
        axes[1, 0].axvline(x=0.05,  color='#2ecc71', linestyle='--',
                           linewidth=1.5, label='Positive threshold (+0.05)')
        axes[1, 0].axvline(x=-0.05, color='#e74c3c', linestyle='--',
                           linewidth=1.5, label='Negative threshold (−0.05)')
        axes[1, 0].set_title('Compound Score Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Compound Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # ── Chart 4 : Category breakdown (or scatter) ────────
        if 'category' in self.df.columns:
            cat_df = self.df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            bar_colors = [self.COLORS.get(col, '#95a5a6') for col in cat_df.columns]
            cat_df.plot(kind='bar', ax=axes[1, 1], color=bar_colors, edgecolor='black')
            axes[1, 1].set_title('Sentiment by Category', fontweight='bold')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend(title='Sentiment')
        else:
            scatter_c = [self.COLORS.get(s, '#95a5a6') for s in self.df['sentiment']]
            axes[1, 1].scatter(range(len(self.df)), self.df['compound_score'],
                               c=scatter_c, alpha=0.8, edgecolors='black', s=60)
            axes[1, 1].set_title('Compound Score per Feedback Entry', fontweight='bold')
            axes[1, 1].set_xlabel('Feedback Index')
            axes[1, 1].set_ylabel('Compound Score')
            patches = [mpatches.Patch(color=v, label=k) for k, v in self.COLORS.items()]
            axes[1, 1].legend(handles=patches)

        plt.tight_layout()
        path = 'output/sentiment_dashboard.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅  Dashboard saved  →  {path}")

        # Also generate word clouds
        self._generate_wordcloud()

    def _generate_wordcloud(self):
        """Generate one word cloud per sentiment (Positive / Neutral / Negative)."""
        stop = set(stopwords.words('english'))

        def clean(text):
            text  = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            words = [w for w in text.split() if w not in stop and len(w) > 2]
            return ' '.join(words)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('☁️  Word Clouds by Sentiment', fontsize=14, fontweight='bold')

        for idx, (sentiment, cmap) in enumerate(
                zip(['Positive', 'Neutral', 'Negative'], ['Greens', 'Blues', 'Reds'])):

            subset = self.df[self.df['sentiment'] == sentiment]['feedback']
            axes[idx].set_title(f'{sentiment}  ({len(subset)} entries)', fontweight='bold')
            axes[idx].axis('off')

            if subset.empty:
                axes[idx].text(0.5, 0.5, f'No {sentiment}\nfeedback found',
                               ha='center', va='center', transform=axes[idx].transAxes)
                continue

            text = ' '.join(subset.apply(clean))
            if text.strip():
                wc = WordCloud(width=500, height=300, background_color='white',
                               colormap=cmap, max_words=50).generate(text)
                axes[idx].imshow(wc, interpolation='bilinear')

        plt.tight_layout()
        path = 'output/wordclouds.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅  Word clouds saved  →  {path}")

    # ----------------------------------------------------------
    # 5. EXPORT RESULTS
    # ----------------------------------------------------------

    def export_results(self):
        """Save the fully analysed DataFrame to a CSV file."""
        if 'sentiment' not in self.df.columns:
            print("❌  Run analyze_sentiment() first.")
            return

        os.makedirs('output', exist_ok=True)
        path = 'output/analyzed_feedback.csv'
        self.df.to_csv(path, index=False)
        print(f"✅  Results exported  →  {path}")


# ==============================================================
#                     MAIN — ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    print("=" * 52)
    print("       🤖  AI FEEDBACK ANALYSIS SYSTEM")
    print("=" * 52)

    analyzer = FeedbackAnalyzer()

    # ── STEP 1 : Load data ──────────────────────────────────
    # Option A — Use your own CSV (must have a 'feedback' column):
    #   analyzer.load_data('data/my_feedback.csv')
    #
    # Option B — Use the built-in sample data:
    analyzer.load_sample_data()

    # ── STEP 2 : Run sentiment analysis ─────────────────────
    analyzer.analyze_sentiment()

    # ── STEP 3 : Print summary ──────────────────────────────
    analyzer.generate_summary()

    # ── STEP 4 : Generate charts & word clouds ───────────────
    analyzer.visualize()

    # ── STEP 5 : Export results to CSV ──────────────────────
    analyzer.export_results()

    print("\n🎉  All done!  Check the 'output/' folder for charts and CSV results.")
