# ocrmodelconversionfromGPUtoCPU
# OCR Model Conversion refers to the process of adapting an Optical Character Recognition (OCR) model to run efficiently on different hardware or platforms, such as transitioning from a GPU-based environment to a CPU-based one, or from one deep learning framework to another.
import cv2
import easyocr
import time

# Initialize EasyOCR reader for text extraction, using CPU (gpu=False)
reader = easyocr.Reader(['en'], gpu=False)

# Specify the correct path to the video file
video_path = r'video_path'
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties (frame width, height, and FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# Set up the output video file path and codec for saving
output_path = r'C:\Users\YourUsername\Videos\output_video.avi'  # Update this path as needed
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec (XVID)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Start processing the video frames
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Use EasyOCR to extract text from the current frame
    results = reader.readtext(frame)

    # Loop through each detected text result and annotate the frame
    for bbox, text, score in results:
        if score > 0.25:  # Only annotate if the score is above a confidence threshold
            cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)
    
    frame_count += 1

# Release the video capture and output objects once done
cap.release()
out.release()

# Calculate the elapsed time and the FPS of video processing
end_time = time.time()
elapsed_time = end_time - start_time
fps_video = frame_count / elapsed_time if elapsed_time > 0 else 0

# Print out the processing summary
print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
print(f"FPS: {fps_video:.2f}")
print(f"Output video saved at: {output_path}")
