# Dance Crew App — API Documentation

> **Base URL:** `http://localhost:8000`

---

## 1. Upload Video

Upload a solo dance video and auto-trigger group dance generation.

| | |
|---|---|
| **Endpoint** | `POST /videos/` |
| **Content-Type** | `multipart/form-data` |

### Request Body (Form Data)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | `string` | ✅ | Title for the video |
| `dancers_count` | `integer` | ✅ | Number of background dancers to generate (set > 0 to auto-process) |
| `file` | `file` | ✅ | Video file (MP4 recommended) |

### Example Request

```javascript
const formData = new FormData();
formData.append("title", "My Dance");
formData.append("dancers_count", 3);
formData.append("file", videoFile);

const response = await fetch("http://localhost:8000/videos/", {
  method: "POST",
  body: formData,
});
const video = await response.json();
```

### Response — `200 OK`

```json
{
  "id": 1,
  "title": "My Dance",
  "description": null,
  "dancers_count": 3,
  "filename": "a1b2c3d4.mp4",
  "processed_status": "pending",
  "uploaded_at": "2026-02-17T03:00:00",
  "original_file_path": "uploads/a1b2c3d4.mp4",
  "processed_file_path": null,
  "gemini_analysis": null,
  "original_video_url": "http://localhost:8000/static/uploads/a1b2c3d4.mp4",
  "generated_video_url": null
}
```

> **Note:** If `dancers_count > 0`, video generation starts automatically in the background. Poll the video status using **Get Video** endpoint.

---

## 2. List All Videos

Retrieve all uploaded videos with their current processing status.

| | |
|---|---|
| **Endpoint** | `GET /videos/` |

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `skip` | `integer` | `0` | Pagination offset |
| `limit` | `integer` | `100` | Max results to return |

### Example Request

```javascript
const response = await fetch("http://localhost:8000/videos/?skip=0&limit=10");
const videos = await response.json();
```

### Response — `200 OK`

```json
[
  {
    "id": 1,
    "title": "My Dance",
    "dancers_count": 3,
    "filename": "a1b2c3d4.mp4",
    "processed_status": "completed",
    "uploaded_at": "2026-02-17T03:00:00",
    "original_file_path": "uploads/a1b2c3d4.mp4",
    "processed_file_path": "processed/e5f6g7h8.mp4",
    "gemini_analysis": "Successfully generated group dance video...",
    "original_video_url": "http://localhost:8000/static/uploads/a1b2c3d4.mp4",
    "generated_video_url": "http://localhost:8000/static/processed/e5f6g7h8.mp4"
  }
]
```

---

## 3. Get Single Video

Get details of a specific video including its processing status and generated video path.

| | |
|---|---|
| **Endpoint** | `GET /videos/{video_id}` |

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `video_id` | `integer` | ID of the video |

### Example Request

```javascript
const response = await fetch("http://localhost:8000/videos/1");
const video = await response.json();
```

### Response — `200 OK`

```json
{
  "id": 1,
  "title": "My Dance",
  "dancers_count": 3,
  "filename": "a1b2c3d4.mp4",
  "processed_status": "completed",
  "uploaded_at": "2026-02-17T03:00:00",
  "original_file_path": "uploads/a1b2c3d4.mp4",
  "processed_file_path": "processed/e5f6g7h8.mp4",
  "gemini_analysis": "Successfully generated group dance video...",
  "original_video_url": "http://localhost:8000/static/uploads/a1b2c3d4.mp4",
  "generated_video_url": "http://localhost:8000/static/processed/e5f6g7h8.mp4"
}
```

### Error — `404 Not Found`

```json
{ "detail": "Video not found" }
```

---

## 4. Trigger / Re-trigger Processing

Manually start or re-run video generation for an existing video.

| | |
|---|---|
| **Endpoint** | `POST /videos/{video_id}/process` |

### Parameters

| Param | Location | Type | Description |
|-------|----------|------|-------------|
| `video_id` | Path | `integer` | ID of the video |
| `dancers_count` | Query | `integer` | Number of background dancers |

### Example Request

```javascript
const response = await fetch(
  "http://localhost:8000/videos/1/process?dancers_count=4",
  { method: "POST" }
);
const result = await response.json();
```

### Response — `200 OK`

```json
{
  "message": "Processing started",
  "video_id": 1
}
```

---

## 5. Delete Video

Delete a video and all associated files (original + generated).

| | |
|---|---|
| **Endpoint** | `DELETE /videos/{video_id}` |

### Example Request

```javascript
const response = await fetch("http://localhost:8000/videos/1", {
  method: "DELETE",
});
const result = await response.json();
```

### Response — `200 OK`

```json
{ "ok": true }
```

---

## 6. Serve Video Files (Static)

Access original and generated video files directly.

| | |
|---|---|
| **Original Video** | `GET /static/uploads/{filename}` |
| **Generated Video** | `GET /static/processed/{filename}` |

### Example

```javascript
// Use the URLs directly from the response
const originalUrl = video.original_video_url;
const generatedUrl = video.generated_video_url;
```

---

## 7. List All Generated Videos

Retrieve only videos that have completed generation.

| | |
|---|---|
| **Endpoint** | `GET /generated-videos/` |

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `skip` | `integer` | `0` | Pagination offset |
| `limit` | `integer` | `100` | Max results to return |

### Example Request

```javascript
const response = await fetch("http://localhost:8000/generated-videos/");
const generatedVideos = await response.json();
```

### Postman

- Method: **GET**
- URL: `http://localhost:8000/generated-videos/`

### Response — `200 OK`

Returns only videos with `processed_status: "completed"` and a valid `generated_video_url`.

---

## 8. Get Single Generated Video

Get details of a specific generated video.

| | |
|---|---|
| **Endpoint** | `GET /generated-videos/{video_id}` |

### Example Request

```javascript
const response = await fetch("http://localhost:8000/generated-videos/1");
const video = await response.json();
```

### Postman

- Method: **GET**
- URL: `http://localhost:8000/generated-videos/1`

### Response — `200 OK`

```json
{
  "id": 1,
  "title": "My Dance",
  "dancers_count": 3,
  "processed_status": "completed",
  "original_video_url": "http://localhost:8000/static/uploads/a1b2c3d4.mp4",
  "generated_video_url": "http://localhost:8000/static/processed/e5f6g7h8.mp4"
}
```

### Error — `404`

```json
{ "detail": "No generated video available for this video" }
```

---

## 9. Download Generated Video

Directly download the generated video file with a proper filename.

| | |
|---|---|
| **Endpoint** | `GET /generated-videos/{video_id}/download` |

### Example Request

```javascript
// Opens download in browser
window.open("http://localhost:8000/generated-videos/1/download");
```

### Postman

- Method: **GET**
- URL: `http://localhost:8000/generated-videos/1/download`
- Click **Send and Download** (the dropdown next to Send button)

### Response

Returns the MP4 file directly with filename `{title}_generated.mp4`.

---

## 10. Delete Generated Video Only

Delete only the generated video file. The original uploaded video is preserved.

| | |
|---|---|
| **Endpoint** | `DELETE /generated-videos/{video_id}` |

### Example Request

```javascript
const response = await fetch("http://localhost:8000/generated-videos/1", {
  method: "DELETE",
});
const result = await response.json();
```

### Postman

- Method: **DELETE**
- URL: `http://localhost:8000/generated-videos/1`

### Response — `200 OK`

```json
{
  "ok": true,
  "message": "Generated video deleted. Original video preserved."
}
```

---

## Video Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | `integer` | Unique video ID |
| `title` | `string` | Video title |
| `description` | `string \| null` | Optional description |
| `dancers_count` | `integer` | Number of background dancers requested |
| `filename` | `string` | Stored filename |
| `processed_status` | `string` | `pending` · `processing` · `completed` · `failed` |
| `uploaded_at` | `datetime` | Upload timestamp (ISO 8601) |
| `original_file_path` | `string` | Relative path to uploaded video |
| `processed_file_path` | `string \| null` | Relative path to generated video (null until completed) |
| `gemini_analysis` | `string \| null` | Generation result or error details |
| `original_video_url` | `string \| null` | **Full URL** to the original uploaded video |
| `generated_video_url` | `string \| null` | **Full URL** to the generated group dance video |

---

## Processing Status Flow

```
pending → processing → completed
                    ↘ failed
```

| Status | Description |
|--------|-------------|
| `pending` | Video uploaded, processing not started |
| `processing` | Veo is generating the group dance video (takes ~40-60 seconds) |
| `completed` | Generated video is ready at `processed_file_path` |
| `failed` | Generation failed — check `gemini_analysis` for error details |

---

## Polling for Status (Recommended Pattern)

Since video generation takes ~40-60 seconds, poll the status:

```javascript
async function pollUntilDone(videoId) {
  while (true) {
    const res = await fetch(`http://localhost:8000/videos/${videoId}`);
    const video = await res.json();

    if (video.processed_status === "completed") {
      console.log("Video ready:", video.processed_file_path);
      return video;
    }
    if (video.processed_status === "failed") {
      console.error("Failed:", video.gemini_analysis);
      return video;
    }

    // Wait 5 seconds before next poll
    await new Promise(r => setTimeout(r, 5000));
  }
}
```

---

## Running the Server

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`  
Interactive API docs at `http://localhost:8000/docs`
