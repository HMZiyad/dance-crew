import video_processor
import os

video_path = r"uploads\1e0d7b20-5e55-4319-855b-c23572c8d10c.mp4"
output_path = r"processed\frame_test_output.mp4"

print("Starting frame-based pipeline...")
result = video_processor.generate_group_dance_video(video_path, 2, output_path)
print("")
print("Result status:", result["status"])
print("Generated video:", result["generated_video_path"])
print("Analysis:", result["analysis"])
if result["generated_video_path"] and os.path.exists(result["generated_video_path"]):
    size = os.path.getsize(result["generated_video_path"])
    print("Video file size:", size, "bytes =", round(size/1024/1024, 1), "MB")
