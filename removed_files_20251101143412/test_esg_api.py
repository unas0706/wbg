import requests
import json

def print_json(data):
    """Print JSON data in a readable format"""
    print(json.dumps(data, indent=2))

def test_api():
    base_url = 'http://localhost:5000'
    headers = {'Content-Type': 'application/json'}

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_response = requests.get(f'{base_url}/health')
    print_json(health_response.json())

    # Test home endpoint for documentation
    print("\n2. Testing home endpoint for documentation...")
    home_response = requests.get(base_url)
    print_json(home_response.json())

    # Test predict endpoint with different scenarios
    print("\n3. Testing predict endpoint with different scenarios...")

    test_cases = [
        {
            "name": "High Environmental Impact Project",
            "description": "Renewable energy project implementing solar and wind power with carbon reduction initiatives"
        },
        {
            "name": "High Social Impact Project",
            "description": "Community development program focusing on education equality and healthcare access"
        },
        {
            "name": "High Governance Impact Project",
            "description": "Implementing transparency initiative with strong accountability framework and compliance program"
        },
        {
            "name": "Mixed Impact Project",
            "description": "Solar power installation with community training program and transparent governance structure"
        }
    ]

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        response = requests.post(
            f'{base_url}/predict',
            headers=headers,
            json={"description": test_case['description']}
        )
        print_json(response.json())

if __name__ == '__main__':
    test_api()