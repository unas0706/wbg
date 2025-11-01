import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# SDG keywords used to generate weak labels
sdg_keywords = {
    1: ['poverty','income','welfare'],
    2: ['hunger','agriculture','food','nutrition'],
    3: ['health','disease','medical','hospital'],
    4: ['education','school','learning','training'],
    5: ['women','gender','female','equality'],
    6: ['water','sanitation','clean water'],
    7: ['energy','renewable','solar','electricity'],
    8: ['employment','jobs','economic','growth'],
    9: ['infrastructure','industry','innovation','technology'],
    10: ['inequality','equal opportunity','minorities'],
    11: ['cities','urban','housing','transport'],
    12: ['sustainable','consumption','recycle','waste'],
    13: ['climate','carbon','emission'],
    14: ['ocean','marine','sea','fish'],
    15: ['biodiversity','forest','ecosystem','land'],
    16: ['peace','justice','corruption','governance'],
    17: ['partnership','international','cooperation']
}

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'projects.csv')
models_dir = os.path.join(base_dir, 'app', 'models')
os.makedirs(models_dir, exist_ok=True)

print('Loading data from', data_path)
df = pd.read_csv(data_path)

# ensure Description column exists
if 'Description' not in df.columns:
    raise SystemExit('No Description column found in CSV')

# Function to compute SDG vector labels
def score_sdg(text):
    text = str(text).lower()
    vector = []
    for sdg, words in sdg_keywords.items():
        score = min(1.0, sum(w in text for w in words) / max(1, len(words)))
        vector.append(round(score, 2))
    return vector

print('Generating SDG labels (weak labels from keywords)')
labels = df['Description'].apply(score_sdg)
Y = pd.DataFrame(labels.tolist(), columns=[f'SDG{i}' for i in range(1,18)])

# Vectorize descriptions
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Description'].astype(str))

# Train simple regression model
print('Training SDG MultiOutputRegressor...')
model = MultiOutputRegressor(LinearRegression())
model.fit(X, Y)

# Save artifacts
vfile = os.path.join(models_dir, 'vectorizer.pkl')
mdfile = os.path.join(models_dir, 'sdg_regression.pkl')
joblib.dump(vectorizer, vfile)
joblib.dump(model, mdfile)
print('Saved vectorizer to', vfile)
print('Saved SDG model to', mdfile)
print('Done')
