import argparse
import yaml
import cv2
import time
from depthfinger import DepthFinger

def main(model: str = "vitl", pred_only: bool = False):
    # Load the camera parameters
    with open("./configs/camera.yaml", "r") as f:
        camera_params = yaml.load(f.read(), Loader=yaml.Loader)

    # Initialize the camera
    depth_finger = DepthFinger(model=model, camera_params=camera_params, pred_only=pred_only)
    
    # Initialize the variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Start the loop
    try:
        while True:
            # Get the result
            result = depth_finger.infer()
            # Display the result
            cv2.imshow('Depth Finger', result)

            # Calculate FPS
            frame_count += 1
            if frame_count % 50 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f'FPS: {fps:.2f}', end='\r')
                start_time = time.time()
                frame_count = 0

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    finally:
        del depth_finger
        

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Depth Finger')
    parser.add_argument('--model', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    args = parser.parse_args()
    
    # Run the main function
    main(model = args.model, pred_only=args.pred_only)