import cv2
import easyocr
import os
import glob
import pandas as pd
import shutil
import time
import re
from datetime import datetime

# Global reader to avoid reinitialization
READER = None

def quick_check_macula_thickness(image):
    """Fast check if image is likely an OCT report based on colors and layout"""
    # Check for specific colors in OCT reports (pink/green in the ETDRS grid)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Check for pink (common in ETDRS grid)
    pink_lower = (140, 50, 150)
    pink_upper = (170, 255, 255)
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    
    # Check for green (common in ETDRS grid)
    green_lower = (40, 50, 50)
    green_upper = (80, 255, 255)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # If we find enough pink or green areas, it's likely an OCT report
    pink_count = cv2.countNonZero(pink_mask)
    green_count = cv2.countNonZero(green_mask)
    
    if pink_count > 1000 or green_count > 1000:
        return True
    
    # If no color match, check for a grid-like structure
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Check for circular patterns (ETDRS grid is circular)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=30, maxRadius=100)
    
    if circles is not None and len(circles[0]) >= 1:
        return True
    
    return False

def extract_central_thickness(image, reader):
    """Extract the Central Subfield Thickness value from the OCT report"""
    height, width = image.shape[:2]
    
    # Targeted approach - only scan bottom right corner for table
    # This is where "Central Subfield Thickness" typically appears
    x = max(0, width // 2)
    y = max(0, height * 2 // 3)  # Bottom third
    w = width - x
    h = height - y
    
    # Crop the bottom-right section
    bottom_right = image[y:y+h, x:x+w]
    
    # Resize for faster OCR (smaller size)
    scale_factor = 0.8
    small_br = cv2.resize(bottom_right, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Convert to grayscale
    gray_br = cv2.cvtColor(small_br, cv2.COLOR_BGR2GRAY)
    
    # First try to find numbers that appear after "Central" or "Thickness"
    try:
        text_results = reader.readtext(gray_br, detail=1)
        
        # Sort text by y-coordinate (top to bottom)
        text_results.sort(key=lambda x: x[0][0][1])  # Sort by y-coordinate of top-left point
        
        # Look for keywords and then check the next items
        for i, (bbox, text, conf) in enumerate(text_results):
            if any(keyword in text.lower() for keyword in ['central', 'subfield', 'thickness', 'µm']):
                # Look at nearby text elements for numbers
                for j in range(max(0, i-3), min(i+4, len(text_results))):
                    candidate = text_results[j][1]
                    digits = ''.join(c for c in candidate if c.isdigit())
                    
                    # Check if it's a plausible thickness value (200-600 µm is typical)
                    if digits and 2 <= len(digits) <= 3:
                        if 150 <= int(digits) <= 800:
                            print(f"Found thickness value: {digits}")
                            return digits
    except Exception as e:
        print(f"Error in extracting text: {e}")
    
    # If still not found, try a simpler approach - just find all 3-digit numbers
    try:
        all_text = reader.readtext(gray_br, detail=0)
        for text in all_text:
            # Extract all numbers
            numbers = re.findall(r'\b\d{2,3}\b', text)
            for num in numbers:
                if 150 <= int(num) <= 800:
                    print(f"Found potential thickness value: {num}")
                    return num
    except:
        pass
    
    # Last resort - manual analysis
    # Look for text in center box of ETDRS grid (typically positioned in the upper right quadrant)
    try:
        # Focus just on the upper right quadrant where ETDRS grid is usually located
        grid_region = image[0:height//2, width//2:width]
        # Make the image grayscale and apply threshold to highlight numbers
        gray_grid = cv2.cvtColor(grid_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_grid, 180, 255, cv2.THRESH_BINARY_INV)
        
        grid_text = reader.readtext(thresh, detail=0)
        
        for text in grid_text:
            digits = ''.join(c for c in text if c.isdigit())
            if digits and 2 <= len(digits) <= 3:
                if 150 <= int(digits) <= 800:
                    print(f"Found thickness value in ETDRS grid: {digits}")
                    return digits
    except:
        pass
        
    print("Could not find central thickness value")
    return None

def process_image(image_path, reader):
    """Process a single image to extract OCT data"""
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None

    # Quick check if this looks like an OCT report (based on colors/features)
    if not quick_check_macula_thickness(image):
        print(f"Invalid: Not an OCT report: {os.path.basename(image_path)}")
        return None

    # Extract central thickness value
    thickness_value = extract_central_thickness(image, reader)
    
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
        # Use today's date if no date found
        date_text = datetime.now().strftime('%m/%d/%Y')
    
    # If we don't have a thickness value, this image isn't useful
    if not thickness_value:
        print(f"No thickness value found in {filename}")
        return None
        
    return date_text, thickness_value

def setup_folders(main_folder):
    """Create necessary folders for manual sorting"""
    # Create all required folders if they don't exist
    left_folder = os.path.join(main_folder, "left")
    right_folder = os.path.join(main_folder, "right")
    unsorted_folder = os.path.join(main_folder, "unsorted")
    processed_folder = os.path.join(main_folder, "processed")
    invalid_folder = os.path.join(main_folder, "invalid")
    retry_folder = os.path.join(main_folder, "retry")  # Added retry folder
    
    for folder in [left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder

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

def pre_process_images(main_folder, unsorted_folder, invalid_folder, reader):
    """Pre-process images to identify which ones have valid data"""
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
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"Checking: {filename}")
        
        # Process the image to see if it has valid data
        result = process_image(image_path, reader)
        
        if result is None:
            # Move to invalid folder
            dest_path = os.path.join(invalid_folder, filename)
            try:
                shutil.move(image_path, dest_path)
                invalid_count += 1
                print(f"Moved to invalid folder: {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
        else:
            valid_count += 1
    
    print(f"\nPre-processing complete: {valid_count} valid images, {invalid_count} invalid images")
    print("Invalid images have been moved to the 'invalid' folder")
    
    if valid_count == 0:
        print("No valid images found. Exiting process.")
        return False
    
    return True

def process_sorted_images(folder, side, reader):
    """Process images in the left or right folder"""
    # Get list of images in the folder
    image_files = glob.glob(os.path.join(folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(folder, "*.jpeg"))
                  
    if not image_files:
        print(f"No image files found in {side} folder")
        return None
        
    print(f"\nProcessing {len(image_files)} {side} eye images")
    results = []
    
    for image_path in image_files:
        ocr_result = process_image(image_path, reader)
        if ocr_result is None:
            continue
            
        date_text, thickness_value = ocr_result
        results.append({
            "filename": os.path.basename(image_path),
            "date": date_text,
            "thickness": thickness_value
        })
    
    if results:
        # Create a DataFrame
        df = pd.DataFrame(results)
        return df
    else:
        print(f"No valid results from {side} eye images.")
        return None

def process_retry_images(retry_folder, left_folder, right_folder):
    """Process images in the retry folder with manual input (no GUI)"""
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
        
        # Extract date from filename (same as in process_image)
        date_text = ""
        date_match = re.search(r'_(\d{8})', filename)
        if date_match:
            date_str = date_match.group(1)
            try:
                parsed_date = datetime.strptime(date_str, '%Y%m%d')
                date_text = parsed_date.strftime('%m/%d/%Y')
                print(f"Extracted date from filename: {date_text}")
            except ValueError:
                date_text = datetime.now().strftime('%m/%d/%Y')
        else:
            date_text = datetime.now().strftime('%m/%d/%Y')
            
        # Ask for manual input
        print("\nPlease manually enter the data for this image:")
        print(f"Filename: {filename}")
        print(f"Date: {date_text}")
        print("Look for the Central Subfield Thickness value in the bottom table")
        
        # Ask for the thickness value
        while True:
            thickness_value = input("Enter the Central Subfield Thickness (e.g., 359): ").strip()
            if thickness_value == "" or thickness_value.isdigit():
                break
            print("Invalid input. Please enter digits only or press Enter to skip.")
        
        if thickness_value == "":
            print(f"Skipping {filename} (no thickness value entered).")
            continue
        
        # Ask which eye this is (left or right)
        while True:
            eye_side = input("Is this a left or right eye? (l/r): ").strip().lower()
            if eye_side in ['l', 'r']:
                break
            print("Invalid input. Please enter 'l' for left or 'r' for right.")
        
        # Add to the appropriate results list
        result = {
            "filename": filename,
            "date": date_text,
            "thickness": thickness_value
        }
        
        dest_path = ""
        if eye_side == 'l':
            left_results.append(result)
            dest_path = os.path.join(left_folder, filename)
        else:  # eye_side == 'r'
            right_results.append(result)
            dest_path = os.path.join(right_folder, filename)
        
        # Move the image to the appropriate folder
        try:
            shutil.copy(image_path, dest_path)
            print(f"Copied to {'left' if eye_side == 'l' else 'right'} folder: {filename}")
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
    READER = easyocr.Reader(['en'], gpu=True)  # Basic initialization
    
    # Folder path containing your images
    main_folder = r"C:\Users\Markk\Downloads\TBP"
    
    print("\n" + "="*60)
    print("OCT REPORT PROCESSOR - OPTIMIZED VERSION")
    print("="*60)
    print(f"Main folder: {main_folder}")
    
    # Setup the folder structure
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder = setup_folders(main_folder)
    
    print(f"Created folders: left, right, unsorted, processed, invalid, retry")
    
    # Move images to unsorted folder
    moved_count = move_unsorted_images(main_folder, unsorted_folder)
    print(f"Moved {moved_count} images to the unsorted folder")
    
    # Pre-process images to identify valid ones
    has_valid_images = pre_process_images(main_folder, unsorted_folder, invalid_folder, READER)
    
    if not has_valid_images:
        return
    
    print("\n" + "="*60)
    print("INSTRUCTIONS")
    print("="*60)
    print("1. Manually move VALID images from the 'unsorted' folder to either 'left' or 'right' folder")
    print("2. For images that weren't detected but should be processed, move them to the 'retry' folder")
    print("3. After sorting all images, press Enter to continue processing")
    print("="*60)
    
    input("\nPress Enter when you've finished manually sorting the images...")
    
    # Process left images
    left_df = process_sorted_images(left_folder, "left", READER)
    
    # Process right images
    right_df = process_sorted_images(right_folder, "right", READER)
    
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
