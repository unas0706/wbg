import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import os
import requests
from io import StringIO

def download_and_process_data():
    # World Bank Projects API endpoint
    api_url = "https://search.worldbank.org/api/v2/projects"
    
    params = {
        'format': 'json',
        'rows': 1000,  # Get 1000 projects
        'fl': 'id,project_abstract,boardapprovaldate,countryname,project_name',
        'source': 'IBRD'  # International Bank for Reconstruction and Development projects
    }
    
    print("Downloading World Bank project data...")
    response = requests.get(api_url, params=params)
    data = response.json()
    
    # Extract projects
    projects = data.get('projects', [])
    
    # Create DataFrame
    df = pd.DataFrame(projects)
    
    # Clean and prepare data
    df['description'] = df['project_abstract'].fillna('') + ' ' + df['project_name'].fillna('')
    df = df[['id', 'countryname', 'description']].dropna()
    df.columns = ['ProjectID', 'Country', 'Description']
    
    print(f"✅ Processed {len(df)} projects")
    return df

print('✅ Libraries loaded')
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

# Ensure the data directory exists
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)
df.to_csv(os.path.join(data_dir, 'projects.csv'), index=False)
print('✅ Sample data created and saved')

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
df.to_csv(os.path.join(data_dir, 'projects_scored.csv'), index=False)
print('✅ ESG scores generated and saved')

# Train Regression Model
print('\nTraining regression model...')
X = df['Description']
y = df[['E','S','G']]
vectorizer = TfidfVectorizer(max_features=4000)
X_vec = vectorizer.fit_transform(X)
model = MultiOutputRegressor(LinearRegression()).fit(X_vec, y)

# Save models
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, os.path.join(models_dir, 'esg_regression.pkl'))
joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))
print('✅ ESG Regression Model trained and saved')

# Test prediction
print('\nTesting predictions...')
def predict_esg(text):
    vec = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
    model = joblib.load(os.path.join(models_dir, 'esg_regression.pkl'))
    pred = model.predict(vec.transform([text]))[0]
    return {'Environment': round(pred[0],2),'Social': round(pred[1],2),'Governance': round(pred[2],2)}

test_text = 'This project installs solar microgrids for rural villages and trains women workers.'
print(f'Test prediction for: "{test_text}"')
print(f'ESG Scores: {predict_esg(test_text)}')