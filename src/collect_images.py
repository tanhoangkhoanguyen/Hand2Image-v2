import cv2
import os
import time
import uuid

from src.utils.logger import get_logger
from src.utils.setup import get_classes, get_camera_id

LOGGER = get_logger()

class CaptureImages():
    def __init__(
            self, 
            path: str, 
            classes: dict, 
            camera_id: int
        ) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path 
        self.classes = classes
        
        # Initialize LOGGER and show banner
        LOGGER.print_banner()
        LOGGER.capture("Image capture system initialized")
        
        # Verify camera connection
        if not self.cap.isOpened():
            LOGGER.error(f"Could not open camera {camera_id}")
            raise RuntimeError(f"Could not open camera {camera_id}")

        LOGGER.success(f"Camera {camera_id} connected successfully")
        
        # Ensure output directory exists
        os.makedirs(self.path, exist_ok=True)
        LOGGER.info(f"Output directory: {self.path}")

    def capture(
            self, 
            class_name: str
        ) -> bool:     
        try: 
            ret, frame = self.cap.read() 
            if not ret:
                raise Exception("Failed to read from camera")

            raw_frame = frame.copy()
            image = cv2.putText(
                frame, 
                f'Capturing {class_name}', 
                (0,100), 
                cv2.FONT_HERSHEY_DUPLEX, 
                1, 
                (0,0,0), 
                1, 
                cv2.LINE_AA
            )
            cv2.imshow('', image)
            
            # Generate unique filename
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                LOGGER.warning("Quit key pressed - stop capturing")
                return False

            return True
        except Exception as e: 
            LOGGER.capture_error(class_name, str(e))
            return False

    def run(
            self, 
            num_images: int = 1, 
            sleep_time: int = 1
        ):
        # Display session information
        LOGGER.capture_session_start(
            self.classes, 
            num_images, 
            sleep_time
        )
        
        total_captured = 0
        for _, img_class in enumerate(self.classes): 
            LOGGER.capture_class_start(img_class, num_images)
            
            # Create progress bar for this class
            with LOGGER.create_capture_progress(num_images, img_class) as progress:
                class_task = progress.add_task(
                    f"Capturing {img_class}", 
                    total=num_images
                )

                class_captured = 0
                for idx in range(num_images): 
                    success = self.capture(img_class)
                    
                    if success:
                        class_captured += 1
                        total_captured += 1
                        LOGGER.capture_success(img_class, idx + 1)
                    else:
                        LOGGER.capture_error(img_class, f"Image {idx + 1}")
                    progress.update(class_task, advance=1)

                    time.sleep(sleep_time)

                # Show completion for this class
                LOGGER.success(f"Completed {img_class}: {class_captured}/{num_images} images captured")
        
        # Show session completion
        LOGGER.capture_session_complete(total_captured, len(self.classes))
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        LOGGER.info("Camera released and windows closed")

if __name__ == "__main__":
    cap = CaptureImages(
        "./data/train",
        get_classes() , 
        get_camera_id()
    )
    cap.run(num_images=100)