import cv2
import easyocr
import heapq
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)
# Define video paths
video_path = r'C:\path\to\your\video.mp4'
output_path = r'C:\Users\YourUsername\Videos\output_video.mp4'
# Capture video and check if opened successfully
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    exit("Error: Unable to open video file.")
# Video properties
frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
# Parameters
resize_factor, frame_skip = 0.5, 5  # Downscale and skip frames for efficiency
frame_heap, max_heap_size = [], 10  # Priority queue for most confident results
frame_count, processed_count = 0, 0
# Function to preprocess frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor), cv2.COLOR_BGR2GRAY)
    return gray
# Function to annotate frame
def annotate_frame(frame, results):
    for bbox, text, score in results:
        if score > 0.5:  # Confidence threshold
            bbox = [[int(p[0] / resize_factor), int(p[1] / resize_factor)] for p in bbox]
            cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
# Start processing
start_time = cv2.getTickCount()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for efficiency
    if frame_count % frame_skip == 0:
        gray_frame = preprocess_frame(frame)
        results = reader.readtext(gray_frame)

        # Use a heap to store top N confident results
        for bbox, text, score in results:
            if len(frame_heap) < max_heap_size:
                heapq.heappush(frame_heap, (score, bbox, text))
            else:
                heapq.heappushpop(frame_heap, (score, bbox, text))

        # Annotate frame with top results
        top_results = [(b, t, s) for s, b, t in sorted(frame_heap, reverse=True)]
        frame = annotate_frame(frame, top_results)
        out.write(frame)
        processed_count += 1

    frame_count += 1
cap.release()
out.release()
# Calculate processing time
elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
print(f"Processed {processed_count} frames in {elapsed_time:.2f}s. FPS: {frame_count / elapsed_time:.2f}. Output: {output_path}")
