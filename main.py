import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import os
from ultralytics import YOLO

try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    messagebox.showerror("Model Load Error", f"Failed to load YOLOv8 model. Make sure 'yolov8n.pt' is accessible or run 'pip install ultralytics'.\nError: {e}")
    exit()

def predict_folder(folder_path):
    """
    Detects humans in all images within a specified folder using YOLOv8.
    Displays each image with bounding boxes and the detected count.
    """
    results = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        yolo_results = model(file_path, classes=0, conf=0.4)
        
        annotated_img = yolo_results[0].plot() 
        human_count = len(yolo_results[0].boxes)
        
        results.append((filename, human_count))
        
        cv2.putText(annotated_img, f'Detected Humans: {human_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f'Detection Result - {filename}', annotated_img)
        cv2.waitKey(0) 
    
    cv2.destroyAllWindows()
    print("\n--- Folder Processing Summary ---")
    for filename, count in results:
        print(f'File: {filename} | Detected Humans: {count}')
    print("---------------------------------")
    
def predict_video(video_path):
    """
    Performs real-time human detection on a video file using YOLOv8.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video file: {video_path}")
        return
        
    max_concurrent_humans = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=0, conf=0.4, verbose=False)
        
        annotated_frame = results[0].plot()
        human_count = len(results[0].boxes)
        
        if human_count > max_concurrent_humans:
            max_concurrent_humans = human_count

        cv2.putText(annotated_frame, f'Live Count: {human_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Max Concurrent: {max_concurrent_humans}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
        cv2.imshow('YOLOv8 Human Detection - Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Video Processing Complete", f"Maximum concurrent humans detected: {max_concurrent_humans}")

def predict_camera():
    """
    Performs real-time human detection using the webcam.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open the camera. Please check your webcam.")
        return

    print("Press 'q' to quit the camera feed.")
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Could not read frame from camera.")
            break

        results = model(frame, classes=0, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        human_count = len(results[0].boxes)

        cv2.putText(annotated_frame, f'Detected Humans: {human_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('YOLOv8 Live Human Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Camera feed closed by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera detection session ended.")

def select_folder():
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        print(f"Processing folder: {folder_path}")
        predict_folder(folder_path)
    else:
        messagebox.showwarning("No Folder Selected", "Please select a folder to process.")

def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if video_path:
        print(f"Processing video: {video_path}")
        predict_video(video_path)
    else:
        messagebox.showwarning("No Video Selected", "Please select a video file to process.")

def start_camera_detection():
    print("Starting camera detection...")
    try:
        predict_camera()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start camera detection: {str(e)}")

def center_window(window, width=400, height=300):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def main():
    root = tk.Tk()
    root.title("Human Detection System")
    root.geometry("800x600")
    center_window(root, width=800, height=500)
    root.configure(bg="cyan")

    title_label = tk.Label(root, text="Human Detection System", font=("Helvetica", 25, "bold"), bg='cyan', fg='black')
    title_label.pack(pady=20)
    folder_button = tk.Button(root, text="Process a Folder of Images", font=("Helvetica", 20, "bold"), bg='lime', fg='black',
                               command=select_folder, width=30)
    folder_button.pack(pady=10)
    video_button = tk.Button(root, text="Process a Video File", font=("Helvetica", 20, "bold"), bg='lime', fg='black',
                              command=select_video, width=30)
    video_button.pack(pady=10)
    camera_button = tk.Button(root, text="Use Camera for Live Detection", font=("Helvetica", 20, "bold"), bg='lime', fg='black',
                               command=start_camera_detection, width=30)
    camera_button.pack(pady=10)
    exit_button = tk.Button(root, text="EXIT", font=("Helvetica", 20, "bold"), fg='red', borderwidth=2, command=root.quit, width=30)
    exit_button.pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
