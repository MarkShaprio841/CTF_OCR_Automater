import cv2
import easyocr
import os
import glob
import pandas as pd
import shutil
import time
import re
import numpy as np
from datetime import datetime

# Global reader to avoid reinitialization
READER = None

def detect_eye_side(image_path):
    """
    Detect if an image is for left or right eye by looking for the blue filled circle
    next to OD (right eye) or OS (left eye) in the top-right corner
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # The OD/OS indicators are typically in the top right corner with blue circles
    # Extract the region where OD/OS markers would be (top right area)
    top_height = height // 6  # Top 1/6th of the image
    marker_region = image[0:top_height, width//2:width]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(marker_region, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color (the filled circle is typically blue)
    # Blue HSV range can vary, so we'll use a range that should capture most blue circles
    lower_blue = np.array([90, 50, 50])  # Lighter blues
    upper_blue = np.array([130, 255, 255])  # Darker blues
    
    # Create a mask for blue regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours of blue regions - these could be our filled circles
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If we found some blue regions, let's analyze them
    if contours:
        # Filter for circular contours of appropriate size
        blue_circles = []
        for contour in contours:
            # Get the area of the contour
            area = cv2.contourArea(contour)
            
            # Filter out very small or very large contours
            if 20 < area < 2000:  # Adjust these thresholds based on your images
                # Check if it's approximately circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:  # Avoid division by zero
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # More circular shapes have values closer to 1
                        # Get the center of the contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            blue_circles.append((cx, cy, area))
        
        # If we found blue circles, try to determine eye side
        if blue_circles:
            # Sort circles by size (largest first)
            blue_circles.sort(key=lambda x: x[2], reverse=True)
            
            # For debugging, draw the circles on a copy of the region
            debug_img = marker_region.copy()
            for cx, cy, area in blue_circles:
                cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
            
            # Now use OCR to find the position of the OD and OS text
            gray = cv2.cvtColor(marker_region, cv2.COLOR_BGR2GRAY)
            
            # Use OCR to find text in the region
            global READER
            results = READER.readtext(gray)
            
            # Look for OD and OS text
            od_pos = None
            os_pos = None
            
            for bbox, text, conf in results:
                # Get the center of the text box
                center_x = sum(p[0] for p in bbox) / 4
                center_y = sum(p[1] for p in bbox) / 4
                
                # Check for OD and OS (case insensitive)
                if "OD" in text.upper():
                    od_pos = (center_x, center_y)
                if "OS" in text.upper():
                    os_pos = (center_x, center_y)
            
            # If we found both OD and OS positions, see which one is closer to the blue circle
            if od_pos and os_pos:
                # Get the largest blue circle
                circle_x, circle_y, _ = blue_circles[0]
                
                # Calculate distances from circle to OD and OS
                dist_to_od = np.sqrt((circle_x - od_pos[0])**2 + (circle_y - od_pos[1])**2)
                dist_to_os = np.sqrt((circle_x - os_pos[0])**2 + (circle_y - os_pos[1])**2)
                
                # The closer one indicates the eye
                if dist_to_od < dist_to_os:
                    print(f"Detected RIGHT eye (OD) - Blue circle is closer to OD")
                    return "right"
                else:
                    print(f"Detected LEFT eye (OS) - Blue circle is closer to OS")
                    return "left"
    
    # If we reach here, try fallback methods
    # Look for just the text OD or OS which may indicate the eye even without the blue circle
    gray = cv2.cvtColor(marker_region, cv2.COLOR_BGR2GRAY)
    text = ' '.join(READER.readtext(gray, detail=0))
    
    if "OD" in text.upper() and "OS" not in text.upper():
        print("Detected RIGHT eye (OD) by text")
        return "right"
    elif "OS" in text.upper() and "OD" not in text.upper():
        print("Detected LEFT eye (OS) by text")
        return "left"
    elif "OD" in text.upper() and "OS" in text.upper():
        # Both appear, check the order - typically the selected eye is marked first
        if text.upper().find("OD") < text.upper().find("OS"):
            print("Detected RIGHT eye (OD) by text order")
            return "right"
        else:
            print("Detected LEFT eye (OS) by text order")
            return "left"
    
    # If all else fails
    print("Could not determine eye side")
    return None

def advanced_eye_detection(image_path):
    """
    More advanced eye detection methods for images that failed initial classification.
    This function uses different approaches than the primary detection method.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # 1. Check for specific patterns in the Zeiss format report header
    # This method focuses specifically on the filled circle in OD/OS indicators at top
    try:
        # Get the exact region where the OD/OS indicators appear in Zeiss reports
        header_region = image[170:200, width-220:width-80]
        
        # Convert to grayscale and threshold to isolate filled circles
        gray = cv2.cvtColor(header_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Split into OD and OS sides (left half is OD, right half is OS)
        w = binary.shape[1]
        od_region = binary[:, :w//2]
        os_region = binary[:, w//2:]
        
        # Count dark pixels in each region (filled circles have more dark pixels)
        od_pixels = np.sum(od_region > 0)
        os_pixels = np.sum(os_region > 0)
        
        # Need significant difference to be confident
        if od_pixels > 100 and od_pixels > os_pixels * 1.3:
            print(f"ADVANCED: Detected RIGHT eye (OD) by filled circle pattern")
            return "right"
        elif os_pixels > 100 and os_pixels > od_pixels * 1.3:
            print(f"ADVANCED: Detected LEFT eye (OS) by filled circle pattern")
            return "left"
    except Exception as e:
        print(f"Error in advanced circle detection: {e}")
    
    # 2. Wider color range method - scan for blue/dark areas in a broader region
    try:
        # Get an expanded region for the header
        top_region = image[0:height//4, width//2:width]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
        
        # Try multiple color ranges to find filled indicators
        color_ranges = [
            # Dark blue
            (np.array([100, 100, 50]), np.array([140, 255, 255])),
            # Broader blue
            (np.array([90, 50, 50]), np.array([150, 255, 255])),
            # Dark areas (for black filled circles)
            (np.array([0, 0, 0]), np.array([180, 255, 100]))
        ]
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            
            # Split into left and right halves
            h, w = mask.shape
            left_half = mask[:, :w//2]
            right_half = mask[:, w//2:]
            
            # Count pixels in each half
            left_pixels = np.sum(left_half > 0)
            right_pixels = np.sum(right_half > 0)
            
            # Compare with threshold
            if left_pixels > 200 and left_pixels > right_pixels * 1.5:
                print(f"ADVANCED: Detected RIGHT eye (OD) by color distribution")
                return "right"
            elif right_pixels > 200 and right_pixels > left_pixels * 1.5:
                print(f"ADVANCED: Detected LEFT eye (OS) by color distribution")
                return "left"
    except Exception as e:
        print(f"Error in advanced color detection: {e}")
    
    # 3. Image feature differences (often right and left eye scans have different patterns)
    try:
        # Analyze general image brightness distribution
        # Right eye and left eye OCT scans often have different brightness patterns
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray_img.shape
        
        # Divide the image into quadrants
        left_side = gray_img[:, :w//2]
        right_side = gray_img[:, w//2:]
        
        # Calculate brightness differences
        left_brightness = np.mean(left_side)
        right_brightness = np.mean(right_side)
        
        brightness_diff = abs(left_brightness - right_brightness)
        if brightness_diff > 10:  # Significant difference
            if left_brightness > right_brightness:
                print(f"ADVANCED: Detected potential RIGHT eye (OD) by brightness pattern")
                return "right"
            else:
                print(f"ADVANCED: Detected potential LEFT eye (OS) by brightness pattern")
                return "left"
    except Exception as e:
        print(f"Error in advanced pattern detection: {e}")
    
    # 4. OCR with a larger region and additional text markers
    try:
        # Use a larger region for OCR to catch more text
        top_half = image[0:height//2, :]
        gray_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
        
        global READER
        all_text = ' '.join(READER.readtext(gray_top, detail=0)).upper()
        
        # Look for specific patterns in the text that might indicate eye side
        od_indicators = ["OD", "RIGHT EYE", "RIGHT", "R EYE"]
        os_indicators = ["OS", "LEFT EYE", "LEFT", "L EYE"]
        
        # Count occurrences of each indicator
        od_count = sum(all_text.count(indicator) for indicator in od_indicators)
        os_count = sum(all_text.count(indicator) for indicator in os_indicators)
        
        if od_count > 0 and od_count > os_count:
            print(f"ADVANCED: Detected RIGHT eye (OD) by text indicators")
            return "right"
        elif os_count > 0 and os_count > od_count:
            print(f"ADVANCED: Detected LEFT eye (OS) by text indicators")
            return "left"
    except Exception as e:
        print(f"Error in advanced OCR detection: {e}")
    
    # 5. Last resort - check image filename for indicators of eye side
    try:
        filename = os.path.basename(image_path).upper()
        
        # Check for common filename patterns indicating eye side
        if "_OD_" in filename or "_R_" in filename or "_RIGHT_" in filename or filename.startswith("OD_") or filename.startswith("R_"):
            print(f"ADVANCED: Detected RIGHT eye (OD) from filename")
            return "right"
        elif "_OS_" in filename or "_L_" in filename or "_LEFT_" in filename or filename.startswith("OS_") or filename.startswith("L_"):
            print(f"ADVANCED: Detected LEFT eye (OS) from filename")
            return "left"
    except Exception as e:
        print(f"Error in filename analysis: {e}")
    
    # If all advanced methods fail
    print("ADVANCED: Could not determine eye side with advanced methods")
    return None

def check_if_macula_thickness(image_path):
    """Check if the image is a macula thickness map based on title text"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    height, width = image.shape[:2]
    
    # Extract the title area - this is the region where "Macula Thickness" would appear
    title_region = image[150:220, 100:800]
    
    # Convert to grayscale
    gray_title = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)
    
    # Use OCR to read the title text
    global READER
    title_text = ' '.join(READER.readtext(gray_title, detail=0)).lower()
    print(f"Title text: {title_text}")
    
    # Look for key phrases
    if "macula thickness" in title_text:
        print("Found 'Macula Thickness' in title - keeping this image")
        return True
    
    # If we didn't find the title, do an additional check for "central subfield thickness"
    # which appears in thickness maps but not in line raster scans
    try:
        # Try to look for the central thickness table that appears only in thickness maps
        bottom_half = image[height//2:height, :]
        bottom_text = ' '.join(READER.readtext(cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY), detail=0)).lower()
        
        if "central" in bottom_text and "thickness" in bottom_text:
            print("Found 'Central Thickness' in bottom half - keeping this image")
            return True
    except:
        pass
    
    print("Could not determine if this is a macula thickness map")
    return False

def extract_central_thickness(image_path):
    """Extract the Central Subfield Thickness value from the OCT report"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # Try to find the central thickness value in a table at the bottom
    bottom_section = image[height*2//3:height, width//2:width]
    gray_bottom = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY)
    
    # First look for text containing "Central" and "Thickness"
    global READER
    text_results = READER.readtext(gray_bottom, detail=1)
    
    # Sort by vertical position
    text_results.sort(key=lambda x: x[0][0][1])
    
    # Look for keywords then check nearby text for numbers
    for i, (_, text, _) in enumerate(text_results):
        if "central" in text.lower() or "thickness" in text.lower() or "subfield" in text.lower():
            # Check the next few entries for a number
            for j in range(max(0, i-2), min(i+5, len(text_results))):
                candidate = text_results[j][1]
                digits = ''.join(c for c in candidate if c.isdigit())
                
                if digits and 2 <= len(digits) <= 3:
                    if 150 <= int(digits) <= 800:  # Typical range for thickness values
                        print(f"Found thickness value: {digits}")
                        return digits
    
    # If we didn't find it with keywords, look for any 3-digit number in the typical range
    all_text = ' '.join([result[1] for result in text_results])
    numbers = re.findall(r'\b\d{2,3}\b', all_text)
    
    for num in numbers:
        if 150 <= int(num) <= 800:
            print(f"Found potential thickness value: {num}")
            return num
            
    print("Could not find central thickness value")
    return None

def process_image(image_path):
    """Process a single image to extract OCT data and determine eye side"""
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Check if this is a macula thickness map
    is_thickness_map = check_if_macula_thickness(image_path)
    
    if not is_thickness_map:
        print(f"Skipping {os.path.basename(image_path)} - Not a macula thickness map")
        return None, None, None, "other"
    
    # Extract central thickness value
    thickness_value = extract_central_thickness(image_path)
    
    if not thickness_value:
        print(f"No thickness value found in {os.path.basename(image_path)}")
        return None, None, None, "invalid"
    
    # Detect eye side
    eye_side = detect_eye_side(image_path)
    
    # Extract filename info for backup date
    filename = os.path.basename(image_path)
    
    # Try to extract date from filename or image
    date_text = ""
    date_match = re.search(r'_(\d{8})', filename)
    if date_match:
        date_str = date_match.group(1)
        try:
            # Convert from YYYYMMDD to MM/DD/YYYY
            parsed_date = datetime.strptime(date_str, '%Y%m%d')
            date_text = parsed_date.strftime('%m/%d/%Y')
            print(f"Extracted date from filename: {date_text}")
        except ValueError:
            # If date can't be parsed, use today's date
            date_text = datetime.now().strftime('%m/%d/%Y')
    else:
        # If no date in filename, try to extract from image
        image = cv2.imread(image_path)
        if image is not None:
            # Look in top section where date would be
            date_area = image[80:120, 400:600]
            
            # Try OCR on this area
            global READER
            date_text_results = ' '.join(READER.readtext(cv2.cvtColor(date_area, cv2.COLOR_BGR2GRAY), detail=0))
            
            # Look for date patterns
            date_patterns = [
                r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
                r'(\d{1,2}/\d{1,2}/\d{2})'   # MM/DD/YY
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_text_results)
                if match:
                    date_text = match.group(1)
                    print(f"Extracted date from image: {date_text}")
                    break
        
        # If still no date, use today's date
        if not date_text:
            date_text = datetime.now().strftime('%m/%d/%Y')
    
    print(f"Success - Thickness: {thickness_value}, Date: {date_text}, Eye: {eye_side if eye_side else 'Unknown'}")
    return thickness_value, date_text, eye_side, "valid"

def setup_folders(patient_folder):
    """Create necessary folders for sorting within patient folder"""
    # Create patient-specific processed folder
    patient_name = os.path.basename(patient_folder)
    parent_dir = os.path.dirname(patient_folder)
    processed_patient_folder = os.path.join(parent_dir, f"{patient_name}_processed")
    
    # Create all required folders within the processed patient folder
    left_folder = os.path.join(processed_patient_folder, "left")
    right_folder = os.path.join(processed_patient_folder, "right")
    unsorted_folder = os.path.join(processed_patient_folder, "unsorted")
    processed_folder = os.path.join(processed_patient_folder, "processed")
    invalid_folder = os.path.join(processed_patient_folder, "invalid")
    retry_folder = os.path.join(processed_patient_folder, "retry")
    other_scans_folder = os.path.join(processed_patient_folder, "other_scans")  # For non-macula thickness scans
    
    for folder in [processed_patient_folder, left_folder, right_folder, unsorted_folder, 
                  processed_folder, invalid_folder, retry_folder, other_scans_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return {
        'patient_folder': patient_folder,
        'processed_folder': processed_patient_folder,
        'left_folder': left_folder, 
        'right_folder': right_folder, 
        'unsorted_folder': unsorted_folder, 
        'processed_images_folder': processed_folder, 
        'invalid_folder': invalid_folder, 
        'retry_folder': retry_folder, 
        'other_scans_folder': other_scans_folder
    }

def copy_patient_images(patient_folder, unsorted_folder):
    """Copy all image files from patient folder to unsorted folder"""
    # Get list of images (supports jpg, png, jpeg)
    image_files = glob.glob(os.path.join(patient_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(patient_folder, "*.jpeg"))
    
    # Get images from subfolders as well
    for root, dirs, files in os.walk(patient_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # Copy each image to the unsorted folder
    copied_count = 0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        dest_path = os.path.join(unsorted_folder, filename)
        
        # Check if file already exists in destination
        if os.path.exists(dest_path):
            # Rename with a counter if file exists
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(unsorted_folder, f"{base}_{counter}{ext}")):
                counter += 1
            dest_path = os.path.join(unsorted_folder, f"{base}_{counter}{ext}")
            
        try:
            # Use shutil.copy2 to preserve metadata
            shutil.copy2(image_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {filename}: {e}")
    
    return copied_count

def pre_process_images(folders):
    """Pre-process images to identify and auto-sort images"""
    unsorted_folder = folders['unsorted_folder']
    left_folder = folders['left_folder']
    right_folder = folders['right_folder'] 
    invalid_folder = folders['invalid_folder']
    other_scans_folder = folders['other_scans_folder']
    
    print("\n" + "="*60)
    print("PRE-PROCESSING IMAGES")
    print("="*60)
    
    # Get list of images in the unsorted folder
    image_files = glob.glob(os.path.join(unsorted_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(unsorted_folder, "*.jpeg"))
    
    if not image_files:
        print("No images found in unsorted folder.")
        return False
    
    print(f"Pre-processing {len(image_files)} images...")
    valid_count = 0
    invalid_count = 0
    other_count = 0
    auto_sorted_left = 0
    auto_sorted_right = 0
    manual_sort_needed = 0
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"Checking: {filename}")
        
        # Process the image
        thickness, date, eye_side, result_type = process_image(image_path)
        
        # Handle different cases
        if thickness is None:
            if result_type == "other":
                # This is not a macula thickness map (e.g., line raster)
                dest_path = os.path.join(other_scans_folder, filename)
                try:
                    shutil.move(image_path, dest_path)  # CHANGED: copy2 to move
                    other_count += 1
                    print(f"Moved to other_scans folder: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            else:
                # This is an invalid macula thickness map
                dest_path = os.path.join(invalid_folder, filename)
                try:
                    shutil.move(image_path, dest_path)  # CHANGED: copy2 to move
                    invalid_count += 1
                    print(f"Moved to invalid folder: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
        else:
            # We have a valid macula thickness map
            valid_count += 1
            
            # Auto-sort by eye if detected
            if eye_side == "left":
                dest_path = os.path.join(left_folder, filename)
                try:
                    shutil.move(image_path, dest_path)  # CHANGED: copy2 to move
                    auto_sorted_left += 1
                    print(f"Auto-sorted to left eye folder: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            elif eye_side == "right":
                dest_path = os.path.join(right_folder, filename)
                try:
                    shutil.move(image_path, dest_path)  # CHANGED: copy2 to move
                    auto_sorted_right += 1
                    print(f"Auto-sorted to right eye folder: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            else:
                # Try advanced eye detection for images that couldn't be classified initially
                advanced_eye_side = advanced_eye_detection(image_path)
                if advanced_eye_side == "left":
                    dest_path = os.path.join(left_folder, filename)
                    try:
                        shutil.move(image_path, dest_path)  # CHANGED: copy2 to move
                        auto_sorted_left += 1
                        print(f"Advanced auto-sorted to left eye folder: {filename}")
                    except Exception as e:
                        print(f"Error moving {filename}: {e}")
                elif advanced_eye_side == "right":
                    dest_path = os.path.join(right_folder, filename)
                    try:
                        shutil.move(image_path, dest_path)  # CHANGED: copy2 to move
                        auto_sorted_right += 1
                        print(f"Advanced auto-sorted to right eye folder: {filename}")
                    except Exception as e:
                        print(f"Error moving {filename}: {e}")
                else:
                    # Could not determine eye side, leave in unsorted for manual sorting
                    manual_sort_needed += 1
                    print(f"Eye side not detected, manual sorting needed: {filename}")
    
    print(f"\nPre-processing complete:")
    print(f"  Valid macula thickness maps: {valid_count}")
    print(f"    - Auto-sorted to left eye folder: {auto_sorted_left}")
    print(f"    - Auto-sorted to right eye folder: {auto_sorted_right}")
    print(f"    - Requiring manual eye sorting: {manual_sort_needed}")
    print(f"  Invalid macula thickness maps: {invalid_count}")
    print(f"  Other scan types (not macula thickness): {other_count}")
    
    return valid_count > 0

def process_retry_images(folders):
    """Check for images in the retry folder and log them for later processing"""
    retry_folder = folders['retry_folder']
    processed_folder = folders['processed_folder']
    patient_name = os.path.basename(folders['patient_folder'])
    
    # Get list of images in the retry folder
    image_files = glob.glob(os.path.join(retry_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(retry_folder, "*.jpeg"))
    
    if not image_files:
        print("No images found in retry folder.")
        return None, None
    
    # Create a text file listing images that need manual retry
    retry_file_path = os.path.join(processed_folder, f"{patient_name}_manual_retry_needed.txt")
    with open(retry_file_path, 'w') as f:
        f.write(f"The following {len(image_files)} images for {patient_name} need manual retry processing:\n\n")
        for img_path in image_files:
            f.write(f"- {os.path.basename(img_path)}\n")
    
    print(f"\nFound {len(image_files)} images in retry folder.")
    print(f"These images have been listed in: {retry_file_path}")
    print("You can manually process these images later.")
    
    # We're not processing retry images automatically anymore
    return None, None

def format_and_sort_df(df):
    """Standardize and sort dates in dataframe"""
    if df is None or df.empty:
        return None
        
    # Create a copy to avoid modifying the original
    sorted_df = df.copy()
    
    # Check if dates are missing, if so add a placeholder date
    if sorted_df['date'].isna().all() or (sorted_df['date'] == '').all():
        print("No dates found, using today's date as placeholder")
        sorted_df['date'] = datetime.now().strftime('%m/%d/%Y')
    
    # Convert dates to datetime objects for sorting
    try:
        # First clean up the date format if needed
        sorted_df['clean_date'] = sorted_df['date'].str.replace('/', '-')
        
        # Try to handle various date formats
        date_formats = ['%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d', '%m-%d-%y']
        
        for date_format in date_formats:
            try:
                # For each format, try to convert
                sorted_df['datetime'] = pd.to_datetime(sorted_df['clean_date'], format=date_format, errors='raise')
                print(f"Date format detected: {date_format}")
                break
            except:
                continue
        
        # If none of the formats worked, use a more flexible parser
        if 'datetime' not in sorted_df.columns:
            sorted_df['datetime'] = pd.to_datetime(sorted_df['clean_date'], errors='coerce')
            
        # For any still-invalid dates, use today's date
        if sorted_df['datetime'].isna().any():
            print("Some dates couldn't be parsed, using today's date")
            today = pd.Timestamp(datetime.now())
            sorted_df.loc[sorted_df['datetime'].isna(), 'datetime'] = today
        
        # Format date to DD-MMM-YY
        sorted_df['formatted_date'] = sorted_df['datetime'].dt.strftime('%d-%b-%y')
        
        # Sort by date
        sorted_df = sorted_df.sort_values('datetime')
        
        # Create final formatted dataframe
        final_df = pd.DataFrame({
            'Date': sorted_df['formatted_date'],
            'Thickness': sorted_df['thickness'],
            'Filename': sorted_df['filename']
        })
        
        return final_df
        
    except Exception as e:
        print(f"Error formatting dates: {e}")
        # If date conversion fails, create a basic dataframe without date formatting
        final_df = pd.DataFrame({
            'Date': sorted_df['date'],
            'Thickness': sorted_df['thickness'],
            'Filename': sorted_df['filename']
        })
        return final_df

def process_patient_images(folders):
    """Process images for a single patient"""
    patient_name = os.path.basename(folders['patient_folder'])
    processed_folder = folders['processed_folder']
    left_folder = folders['left_folder']
    right_folder = folders['right_folder']
    unsorted_folder = folders['unsorted_folder']
    
    # Check if there are any unsorted images that need manual review
    unsorted_files = glob.glob(os.path.join(unsorted_folder, "*.[jp][pn]g")) + \
                     glob.glob(os.path.join(unsorted_folder, "*.jpeg"))
    
    # If there are unsorted images, write them to a text file for later review
    if unsorted_files:
        needs_review_file = os.path.join(processed_folder, f"{patient_name}_needs_review.txt")
        with open(needs_review_file, 'w') as f:
            f.write(f"The following {len(unsorted_files)} images for {patient_name} need manual review to determine eye side:\n\n")
            for img_path in unsorted_files:
                f.write(f"- {os.path.basename(img_path)}\n")
        print(f"\nWARNING: {len(unsorted_files)} images could not be auto-classified as left/right eye.")
        print(f"These images have been listed in: {needs_review_file}")
        print("You can manually review these images later.")
    
    # Process left images
    left_df = None
    right_df = None
    
    # Get all images in left folder
    left_files = glob.glob(os.path.join(left_folder, "*.[jp][pn]g")) + \
                 glob.glob(os.path.join(left_folder, "*.jpeg"))
    
    if left_files:
        print(f"\nProcessing {len(left_files)} left eye images")
        left_results = []
        
        for image_path in left_files:
            # Try to process - but ignore eye side result since we know it's left
            thickness, date, _, result_type = process_image(image_path)
            if thickness is not None:
                left_results.append({
                    "filename": os.path.basename(image_path),
                    "date": date,
                    "thickness": thickness
                })
        
        if left_results:
            left_df = pd.DataFrame(left_results)
    
    # Get all images in right folder
    right_files = glob.glob(os.path.join(right_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(right_folder, "*.jpeg"))
    
    if right_files:
        print(f"\nProcessing {len(right_files)} right eye images")
        right_results = []
        
        for image_path in right_files:
            # Try to process - but ignore eye side result since we know it's right
            thickness, date, _, result_type = process_image(image_path)
            if thickness is not None:
                right_results.append({
                    "filename": os.path.basename(image_path),
                    "date": date,
                    "thickness": thickness
                })
        
        if right_results:
            right_df = pd.DataFrame(right_results)
    
    # Format and save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Process and save left eye data
    if left_df is not None and not left_df.empty:
        print(f"LEFT DF before formatting: {len(left_df)} rows")
        print(left_df.head())
        formatted_left_df = format_and_sort_df(left_df)
        if formatted_left_df is not None and not formatted_left_df.empty:
            print(f"LEFT DF after formatting: {len(formatted_left_df)} rows")
            print(formatted_left_df.head())
            left_excel = os.path.join(processed_folder, f"{patient_name}_left_results_{timestamp}.xlsx")
            formatted_left_df.to_excel(left_excel, index=False)
            print(f"\nSaved {len(formatted_left_df)} left eye results to {left_excel}")
        else:
            print("No left eye data to save after formatting")
    else:
        print("No left eye data to process")
    
    # Process and save right eye data
    if right_df is not None and not right_df.empty:
        print(f"RIGHT DF before formatting: {len(right_df)} rows")
        print(right_df.head())
        formatted_right_df = format_and_sort_df(right_df)
        if formatted_right_df is not None and not formatted_right_df.empty:
            print(f"RIGHT DF after formatting: {len(formatted_right_df)} rows")
            print(formatted_right_df.head())
            right_excel = os.path.join(processed_folder, f"{patient_name}_right_results_{timestamp}.xlsx")
            formatted_right_df.to_excel(right_excel, index=False)
            print(f"Saved {len(formatted_right_df)} right eye results to {right_excel}")
        else:
            print("No right eye data to save after formatting")
    else:
        print("No right eye data to process")
    
    # Automatically copy processed images to processed folder
    moved_count = 0
    processed_images_folder = folders['processed_images_folder']
    
    for folder in [left_folder, right_folder]:
        image_files = glob.glob(os.path.join(folder, "*.[jp][pn]g")) + \
                      glob.glob(os.path.join(folder, "*.jpeg"))
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            
            # Add a prefix to indicate which eye it was
            folder_name = os.path.basename(os.path.dirname(image_path))
            new_filename = f"{folder_name}_{filename}"
            dest_path = os.path.join(processed_images_folder, new_filename)
            
            try:
                shutil.copy2(image_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    print(f"Copied {moved_count} processed images to the 'processed' folder")
    
    return True

def main():
    # Start timing the entire process
    start_time = time.time()
    
    # Initialize EasyOCR reader with English language (just once, globally)
    global READER
    print("Initializing EasyOCR reader... (this may take a moment on first run)")
    READER = easyocr.Reader(['en'], gpu=True)  # Using GPU if available, set to False if no GPU
    
    # Main folder containing individual patient folders
    patients_main_folder = input("Enter the path to the main folder containing patient folders: ").strip()
    if not os.path.exists(patients_main_folder):
        print(f"Error: Folder {patients_main_folder} does not exist.")
        return
    
    # Get all immediate subdirectories (patient folders)
    patient_folders = [f.path for f in os.scandir(patients_main_folder) if f.is_dir() 
                       and not f.name.endswith("_processed")]
    
    if not patient_folders:
        print("No patient folders found in the specified directory.")
        return
    
    print(f"\nFound {len(patient_folders)} patient folders to process.")
    for i, folder in enumerate(patient_folders):
        print(f"{i+1}. {os.path.basename(folder)}")
    
    # Ask if user wants to process all or select specific patients
    process_choice = input("\nDo you want to process (a)ll patients or (s)elect specific ones? (a/s): ").strip().lower()
    
    selected_patients = []
    if process_choice == 's':
        while True:
            try:
                indices = input("Enter the numbers of patients to process (comma-separated, e.g., 1,3,5): ").strip()
                if not indices:
                    break
                    
                selected_indices = [int(idx.strip()) - 1 for idx in indices.split(',')]
                selected_patients = [patient_folders[idx] for idx in selected_indices if 0 <= idx < len(patient_folders)]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid patient numbers.")
    else:
        selected_patients = patient_folders
    
    total_patients = len(selected_patients)
    print(f"\nWill process {total_patients} patient folders.")
    
    # Create a summary file to track all patients needing review
    summary_file_path = os.path.join(patients_main_folder, "patients_needing_review.txt")
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write("PATIENTS NEEDING MANUAL REVIEW\n")
        summary_file.write("============================\n\n")
        summary_file.write("The following patients have images that could not be automatically classified:\n\n")
    
    patients_needing_review = []
    
    # Process each patient
    patient_times = []  # Track time for each patient to estimate remaining time
    for patient_index, patient_folder in enumerate(selected_patients):
        patient_start_time = time.time()
        patient_name = os.path.basename(patient_folder)
        
        print("\n" + "="*80)
        print(f"PROCESSING PATIENT {patient_index+1}/{total_patients}: {patient_name}")
        
        # Show estimated time remaining if we have processed at least one patient
        if patient_times:
            avg_time_per_patient = sum(patient_times) / len(patient_times)
            remaining_patients = total_patients - (patient_index)
            est_time_remaining = avg_time_per_patient * remaining_patients
            
            # Format time remaining
            if est_time_remaining < 60:
                time_str = f"{est_time_remaining:.1f} seconds"
            elif est_time_remaining < 3600:
                time_str = f"{est_time_remaining/60:.1f} minutes"
            else:
                time_str = f"{est_time_remaining/3600:.1f} hours"
                
            print(f"Estimated time remaining: {time_str} ({patient_index}/{total_patients} patients processed)")
        
        print("="*80)
        
        # Setup folders for this patient
        folders = setup_folders(patient_folder)
        
        # Copy images to unsorted folder
        copied_count = copy_patient_images(patient_folder, folders['unsorted_folder'])
        print(f"Copied {copied_count} images to the unsorted folder")
        
        if copied_count == 0:
            print(f"No images found for patient {patient_name}. Skipping.")
            continue
        
        # Pre-process images to identify valid ones and auto-sort by eye when possible
        has_valid_images = pre_process_images(folders)
        
        if not has_valid_images:
            print(f"No valid macula thickness maps found for patient {patient_name}. Skipping.")
            continue
        
        # Process this patient's images
        process_patient_images(folders)
        
        # Check if this patient has unsorted images
        unsorted_files = glob.glob(os.path.join(folders['unsorted_folder'], "*.[jp][pn]g")) + \
                         glob.glob(os.path.join(folders['unsorted_folder'], "*.jpeg"))
        if unsorted_files:
            patients_needing_review.append((patient_name, len(unsorted_files), folders['processed_folder']))
        
        # Record how long this patient took
        patient_end_time = time.time()
        patient_duration = patient_end_time - patient_start_time
        patient_times.append(patient_duration)
        
        print(f"\nFinished processing patient: {patient_name} (took {patient_duration:.1f} seconds)")
    
    # Update the summary file with all patients needing review
    if patients_needing_review:
        with open(summary_file_path, 'a') as summary_file:
            for patient_name, unsorted_count, processed_folder in patients_needing_review:
                review_file = os.path.join(processed_folder, f"{patient_name}_needs_review.txt")
                summary_file.write(f"- {patient_name}: {unsorted_count} images need review\n")
                summary_file.write(f"  Review file: {review_file}\n\n")
        
        print(f"\nATTENTION: {len(patients_needing_review)} patients have images that need manual review.")
        print(f"See the summary file: {summary_file_path}")
    else:
        print("\nAll patients were processed successfully with no images needing manual review.")
    
    # Calculate and display total time taken
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Format total time in a readable way
    if total_duration < 60:
        time_str = f"{total_duration:.1f} seconds"
    elif total_duration < 3600:
        time_str = f"{total_duration/60:.1f} minutes"
    else:
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        time_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"
    
    print("\n" + "="*80)
    print(f"PROCESSING COMPLETE! Total time taken: {time_str}")
    print(f"Processed {len(selected_patients)} patients with {len(patients_needing_review)} needing manual review")
    print("="*80)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()