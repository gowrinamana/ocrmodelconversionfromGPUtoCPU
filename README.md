import cv2
import easyocr
import time
# Initialize EasyOCR reader for text extraction, using CPU (gpu=False)
reader = easyocr.Reader(['en'], gpu=False)
# Specify the correct path to the video file
video_path = r'C:\path\to\your\video.mp4'
cap = cv2.VideoCapture(video_path)
# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Set up the output video file path and codec for saving
output_path = r'C:\Users\YourUsername\Videos\output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# Parameters for optimization
frame_skip = 5  # Process every 5th frame
resize_factor = 0.5  # Resize frame to 50% of its original size
frame_count = 0
processed_count = 0
# Start processing the video frames
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
   # Skip frames to reduce processing time
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue
    # Downscale frame to improve processing speed
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    # Convert to grayscale for better OCR accuracy and speed
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    # Use EasyOCR to extract text from the current frame
    results = reader.readtext(gray_frame)
    # Loop through each detected text result and annotate the original frame
    for bbox, text, score in results:
        if score > 0.5:  # Adjust confidence threshold
            # Scale bounding box back to original frame size
            bbox = [[int(point[0] / resize_factor), int(point[1] / resize_factor)] for point in bbox]
            cv2.rectangle(frame, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)
    frame_count += 1
    processed_count += 1
    # Display progress every 10 frames
    if processed_count % 10 == 0:
        print(f"Processed {processed_count} frames...")
# Release resources
cap.release()
out.release()
# Calculate the elapsed time and FPS of video processing
end_time = time.time()
elapsed_time = end_time - start_time
fps_video = processed_count / elapsed_time if elapsed_time > 0 else 0
# Print out the processing summary
print(f"Processed {processed_count} frames in {elapsed_time:.2f} seconds.")
print(f"FPS: {fps_video:.2f}")
print(f"Output video saved at: {output_path}")
