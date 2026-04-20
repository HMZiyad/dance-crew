import os
from dotenv import load_dotenv
from runwayml import RunwayML

# Load environment variables from the .env file
load_dotenv()

def check_runway_api_key():
    # The SDK automatically uses the RUNWAYML_API_SECRET environment variable.
    # If the user has RUNWAY_API_KEY in .env, we can explicitly pass it.
    api_key = os.environ.get("RUNWAY_API_KEY") or os.environ.get("RUNWAYML_API_SECRET")
    
    if not api_key:
        print("RUNWAY_API_KEY or RUNWAYML_API_SECRET not found in .env file.")
        return

    print("Initializing RunwayML client...")
    
    try:
        # Set the environment variable before initialization so we don't have to guess kwargs
        os.environ["RUNWAYML_API_SECRET"] = api_key
        client = RunwayML()
        
        print("Fetching voices to verify authentication...")
        # Attempt to list voices to test authentication
        voices = client.voices.list(limit=10)
        
        print("The API key is valid and working.")
        # Just safely grab the first few items if iterator
        voice_count = sum(1 for _ in zip(voices, range(10)))
        if voice_count > 0:
            print("Successfully retrieved voice data.")
        
    except Exception as e:
        print("The API key is invalid or an error occurred during authentication.")
        print("Error details:", str(e))

if __name__ == "__main__":
    check_runway_api_key()
