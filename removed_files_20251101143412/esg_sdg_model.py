import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

print('✅ Libraries loaded')

# Create sample data
print('\nCreating sample data...')
sample_data = {
    'ProjectID': range(1, 11),
    'Country': ['India', 'Brazil', 'Kenya', 'China', 'Mexico', 'Indonesia', 'Nigeria', 'Egypt', 'Vietnam', 'Thailand'],
    'Description': [
        'Solar power plant installation with community training program',
        'Sustainable forestry management and biodiversity conservation',
        'Rural education and healthcare infrastructure development',
        'Urban water treatment and waste management system',
        'Women entrepreneurship and microfinance program',
        'Renewable energy grid expansion and policy reform',
        'Agricultural modernization and farmer training initiative',
        'Public sector transparency and anti-corruption measures',
        'Climate resilient infrastructure and flood prevention',
        'Digital governance and public service modernization'
    ]
}
df = pd.DataFrame(sample_data)
df.dropna(inplace=True)
df.to_csv('projects.csv', index=False)
print('✅ Data downloaded and saved')

# Generate ESG scores
print('\nGenerating ESG scores...')
def score_esg(text):
    text = str(text).lower()
    E_keywords = ['renewable','solar','wind','climate','water','carbon','forest','pollution','sustainability']
    S_keywords = ['education','health','community','women','youth','poverty','training','employment','social','housing']
    G_keywords = ['governance','transparency','policy','regulation','anti-corruption','institution','audit','compliance','reform']
    def calc_score(keywords): return round(min(1, sum(w in text for w in keywords) / 5), 2)
    return calc_score(E_keywords), calc_score(S_keywords), calc_score(G_keywords)

df[['E','S','G']] = df['Description'].apply(lambda x: pd.Series(score_esg(x)))
df.to_csv('projects_scored.csv', index=False)
print('✅ ESG scores generated and saved')

# Train Regression Model
print('\nTraining regression model...')
X = df['Description']
y = df[['E','S','G']]
vectorizer = TfidfVectorizer(max_features=4000)
X_vec = vectorizer.fit_transform(X)
model = MultiOutputRegressor(LinearRegression()).fit(X_vec, y)
joblib.dump(model, 'esg_regression.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print('✅ ESG Regression Model trained and saved')

# Test prediction
print('\nTesting prediction...')
def predict_esg(text):
    vec = joblib.load('vectorizer.pkl')
    model = joblib.load('esg_regression.pkl')
    pred = model.predict(vec.transform([text]))[0]
    return {'Environment': round(pred[0],2),'Social': round(pred[1],2),'Governance': round(pred[2],2)}

test_text = 'This project installs solar microgrids for rural villages and trains women workers.'
result = predict_esg(test_text)
print(f'Test prediction for: "{test_text}"')
print(f'ESG Scores: {result}')