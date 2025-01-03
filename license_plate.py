import cv2
import numpy as np
import easyocr

def detect_license_plate(image_path):
    """
    Detect and recognize Thai license plates using primarily OCR.
    
    Args:
        image_path (str): Path to the input image
    Returns:
        list: Detected license plate numbers
        image: Annotated image
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Initialize EasyOCR with Thai and English
    reader = easyocr.Reader(['th', 'en'], gpu=False)
    
    # Direct OCR on the image
    results = reader.readtext(image, allowlist='กขคจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789')
    
    detected_plates = []
    
    for (bbox, text, prob) in results:
        # Filter by confidence
        if prob > 0.5:
            # Clean the text
            text = ''.join(c for c in text if c.isdigit() or '\u0E00' <= c <= '\u0E7F')
            
            # Check if text matches Thai license plate pattern (numbers and Thai characters)
            if any(c.isdigit() for c in text) and any('\u0E00' <= c <= '\u0E7F' for c in text):
                detected_plates.append(text)
                
                # Draw bounding box
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                
                # Draw rectangle and text
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(image, text, (top_left[0], top_left[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return detected_plates, image

def main():
    """
    Main function to demonstrate license plate detection.
    """
    image_path = 'one.jpg'
    try:
        plates, annotated_image = detect_license_plate(image_path)
        
        if plates:
            print("Detected license plates:")
            for plate in plates:
                print(f"- {plate}")
            
            # Display the result
            cv2.imshow('Detected License Plates', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No license plates detected.")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()