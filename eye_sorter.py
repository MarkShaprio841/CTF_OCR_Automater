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

def setup_folders(main_folder):
    """Create necessary folders for sorting"""
    # Create all required folders if they don't exist
    left_folder = os.path.join(main_folder, "left")
    right_folder = os.path.join(main_folder, "right")
    unsorted_folder = os.path.join(main_folder, "unsorted")
    processed_folder = os.path.join(main_folder, "processed")
    invalid_folder = os.path.join(main_folder, "invalid")
    retry_folder = os.path.join(main_folder, "retry")
    other_scans_folder = os.path.join(main_folder, "other_scans")  # For non-macula thickness scans
    
    for folder in [left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder, other_scans_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder, other_scans_folder

def move_unsorted_images(main_folder, unsorted_folder):
    """Move all image files from main folder to unsorted folder"""
    # Get list of images (supports jpg, png, jpeg)
    image_files = glob.glob(os.path.join(main_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(main_folder, "*.jpeg"))
    
    # Move each image to the unsorted folder
    moved_count = 0
    for image_path in image_files:
        # Skip if the file is in a subfolder
        if os.path.dirname(image_path) != main_folder:
            continue
            
        filename = os.path.basename(image_path)
        dest_path = os.path.join(unsorted_folder, filename)
        
        # Check if file already exists in destination
        if os.path.exists(dest_path):
            print(f"File already exists in unsorted folder: {filename}")
            continue
            
        try:
            shutil.move(image_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")
    
    return moved_count

def pre_process_images(main_folder, folders):
    """Pre-process images to identify and auto-sort images"""
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder, other_scans_folder = folders
    
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
                    shutil.move(image_path, dest_path)
                    other_count += 1
                    print(f"Moved to other_scans folder: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            else:
                # This is an invalid macula thickness map
                dest_path = os.path.join(invalid_folder, filename)
                try:
                    shutil.move(image_path, dest_path)
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
                    shutil.move(image_path, dest_path)
                    auto_sorted_left += 1
                    print(f"Auto-sorted to left eye folder: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            elif eye_side == "right":
                dest_path = os.path.join(right_folder, filename)
                try:
                    shutil.move(image_path, dest_path)
                    auto_sorted_right += 1
                    print(f"Auto-sorted to right eye folder: {filename}")
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

def process_retry_images(retry_folder, left_folder, right_folder):
    """Process images in the retry folder with manual input"""
    print("\n" + "="*60)
    print("PROCESSING RETRY IMAGES")
    print("="*60)
    
    # Get list of images in the retry folder
    image_files = glob.glob(os.path.join(retry_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(retry_folder, "*.jpeg"))
    
    if not image_files:
        print("No images found in retry folder.")
        return None, None
    
    print(f"Processing {len(image_files)} retry images...")
    
    left_results = []
    right_results = []
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"\nProcessing retry image: {filename}")
        
        # Try to process automatically first
        thickness, date, eye_side, result_type = process_image(image_path)
        
        # If automatic processing failed, or eye side wasn't detected, get manual input
        if thickness is None or result_type == "invalid":
            print("Could not automatically extract thickness value.")
            thickness = input("Enter the thickness value manually: ").strip()
            if not thickness:
                print(f"Skipping {filename} (no thickness value entered).")
                continue
        else:
            print(f"Found thickness value: {thickness}")
            confirm = input(f"Is this thickness value correct? (y/n): ").strip().lower()
            if confirm != 'y':
                thickness = input("Enter the correct thickness value: ").strip()
                if not thickness:
                    print(f"Skipping {filename} (no thickness value entered).")
                    continue
        
        # Use the date if found, otherwise use today's date
        if not date:
            date = datetime.now().strftime('%m/%d/%Y')
            print(f"Using today's date: {date}")
        
        # If eye side wasn't detected, ask for it
        if not eye_side:
            while True:
                eye_choice = input("Is this a left or right eye? (l/r): ").strip().lower()
                if eye_choice in ['l', 'r']:
                    eye_side = "left" if eye_choice == 'l' else "right"
                    break
                print("Invalid input. Please enter 'l' for left or 'r' for right.")
        else:
            # Confirm eye side if auto-detected
            print(f"Detected {eye_side.upper()} eye")
            confirm = input(f"Is this correct? (y/n): ").strip().lower()
            if confirm != 'y':
                while True:
                    eye_choice = input("Is this a left or right eye? (l/r): ").strip().lower()
                    if eye_choice in ['l', 'r']:
                        eye_side = "left" if eye_choice == 'l' else "right"
                        break
                    print("Invalid input. Please enter 'l' for left or 'r' for right.")
        
        # Add to appropriate results list
        if eye_side == "left":
            left_results.append({
                "filename": filename,
                "date": date,
                "thickness": thickness
            })
            # Copy to left folder
            dest_path = os.path.join(left_folder, filename)
            try:
                shutil.copy(image_path, dest_path)
                print(f"Copied to left folder: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        else:  # eye_side == "right"
            right_results.append({
                "filename": filename,
                "date": date,
                "thickness": thickness
            })
            # Copy to right folder
            dest_path = os.path.join(right_folder, filename)
            try:
                shutil.copy(image_path, dest_path)
                print(f"Copied to right folder: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    # Convert results to DataFrames
    left_df = pd.DataFrame(left_results) if left_results else None
    right_df = pd.DataFrame(right_results) if right_results else None
    
    return left_df, right_df

def main():
    # Initialize EasyOCR reader with English language (just once, globally)
    global READER
    print("Initializing EasyOCR reader... (this may take a moment on first run)")
    READER = easyocr.Reader(['en'], gpu=True)  # Using GPU
    
    # Folder path containing your images
    main_folder = r"C:\Users\Markk\Downloads\TBP"
    
    print("\n" + "="*60)
    print("OCT REPORT PROCESSOR - AUTO EYE DETECTION")
    print("="*60)
    print(f"Main folder: {main_folder}")
    
    # Setup the folder structure
    folders = setup_folders(main_folder)
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder, other_scans_folder = folders
    
    print(f"Created folders: left, right, unsorted, processed, invalid, retry, other_scans")
    
    # Move images to unsorted folder
    moved_count = move_unsorted_images(main_folder, unsorted_folder)
    print(f"Moved {moved_count} images to the unsorted folder")
    
    # Pre-process images to identify valid ones and auto-sort by eye when possible
    has_valid_images = pre_process_images(main_folder, folders)
    
    if not has_valid_images:
        return
    
    print("\n" + "="*60)
    print("INSTRUCTIONS")
    print("="*60)
    print("1. Check the left and right folders - images have been auto-sorted when possible")
    print("2. Manually move any remaining images from the 'unsorted' folder to either 'left' or 'right' folder")
    print("3. For images that weren't detected but should be processed, move them to the 'retry' folder")
    print("4. After sorting all images, press Enter to continue processing")
    print("="*60)
    
    input("\nPress Enter when you've finished manually sorting the images...")
    
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
    
    # Process retry images with manual input
    left_retry_df, right_retry_df = process_retry_images(retry_folder, left_folder, right_folder)
    
    # Combine main dataframes with retry dataframes
    if left_df is not None and left_retry_df is not None and not left_retry_df.empty:
        left_df = pd.concat([left_df, left_retry_df]).reset_index(drop=True)
        print(f"Added {len(left_retry_df)} manually processed left eye images")
    elif left_df is None and left_retry_df is not None and not left_retry_df.empty:
        left_df = left_retry_df
        print(f"Created left eye dataframe with {len(left_retry_df)} manually processed images")
    
    if right_df is not None and right_retry_df is not None and not right_retry_df.empty:
        right_df = pd.concat([right_df, right_retry_df]).reset_index(drop=True)
        print(f"Added {len(right_retry_df)} manually processed right eye images")
    elif right_df is None and right_retry_df is not None and not right_retry_df.empty:
        right_df = right_retry_df
        print(f"Created right eye dataframe with {len(right_retry_df)} manually processed images")
    
    # Sort and format date before saving to Excel
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Function to standardize and sort dates
    def format_and_sort_df(df):
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
    
    # Process and save left eye data
    if left_df is not None and not left_df.empty:
        print(f"LEFT DF before formatting: {len(left_df)} rows")
        print(left_df.head())
        formatted_left_df = format_and_sort_df(left_df)
        if formatted_left_df is not None and not formatted_left_df.empty:
            print(f"LEFT DF after formatting: {len(formatted_left_df)} rows")
            print(formatted_left_df.head())
            left_excel = os.path.join(main_folder, f"left_results_{timestamp}.xlsx")
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
            right_excel = os.path.join(main_folder, f"right_results_{timestamp}.xlsx")
            formatted_right_df.to_excel(right_excel, index=False)
            print(f"Saved {len(formatted_right_df)} right eye results to {right_excel}")
        else:
            print("No right eye data to save after formatting")
    else:
        print("No right eye data to process")
    
    # Ask if user wants to move processed images
    move_processed = input("\nDo you want to move all processed images to the 'processed' folder? (y/n): ")
    
    if move_processed.lower() == 'y':
        # Move images from left and right folders to processed folder
        moved_count = 0
        for folder in [left_folder, right_folder]:
            image_files = glob.glob(os.path.join(folder, "*.[jp][pn]g")) + \
                          glob.glob(os.path.join(folder, "*.jpeg"))
            
            for image_path in image_files:
                filename = os.path.basename(image_path)
                
                # Add a prefix to indicate which eye it was
                folder_name = os.path.basename(os.path.dirname(image_path))
                new_filename = f"{folder_name}_{filename}"
                dest_path = os.path.join(processed_folder, new_filename)
                
                try:
                    shutil.move(image_path, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
        
        print(f"Moved {moved_count} processed images to the 'processed' folder")
    
    print("\nProcessing complete!")

if __name__ == '__main__':
    main()
