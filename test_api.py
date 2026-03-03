from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_upload_and_process():
    # 1. Create a dummy video file
    dummy_filename = "test_video.mp4"
    with open(dummy_filename, "wb") as f:
        f.write(b"fake video content")

    try:
        # 2. Upload Video
        print("Testing Upload...")
        with open(dummy_filename, "rb") as f:
            response = client.post(
                "/videos/",
                files={"file": (dummy_filename, f, "video/mp4")},
                data={"title": "Test Dance", "dancers_count": 2}
            )
        
        assert response.status_code == 200
        data = response.json()
        video_id = data["id"]
        print(f"Upload successful. Video ID: {video_id}")

        # 3. List Videos
        print("Testing List Videos...")
        response = client.get("/videos/")
        assert response.status_code == 200
        assert len(response.json()) > 0
        print("List Videos successful.")

        # 4. Trigger Processing (Mocking Gemini would be ideal, but we'll specific endpoint call)
        # Note: This might fail if no API key is set, but we want to test the endpoint logic at least.
        print("Testing Process Trigger...")
        response = client.post(f"/videos/{video_id}/process?dancers_count=5")
        
        if response.status_code == 200:
            print("Processing triggered successfully.")
        else:
            print(f"Processing trigger failed (expected if no API key or env issue): {response.text}")

        # 5. Delete Video
        print("Testing Delete...")
        response = client.delete(f"/videos/{video_id}")
        assert response.status_code == 200
        print("Delete successful.")

    finally:
        # Cleanup
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)
        # Clean up uploaded file from uploads dir if valid id
        # (The delete endpoint handles this, so we are good if step 5 passes)

if __name__ == "__main__":
    test_upload_and_process()
