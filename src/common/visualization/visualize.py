# src/common/visualization/visualize.py
import cv2

def show_overlay_with_severity(overlay, severity_percent):
    """
    Shows overlay with severity text
    """
    display_img = overlay.copy()
    
    # Add severity text
    severity_text = f"Severity: {severity_percent:.2f}%"
    cv2.putText(
        display_img, severity_text, (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
    )
    
    cv2.imshow("Leaf Analysis", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
