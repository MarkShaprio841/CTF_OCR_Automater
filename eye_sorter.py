import cv2
import easyocr
import os
import glob
import pandas as pd
import shutil
import time
import re
from datetime import datetime
import difflib

def check_macula_thickness(image, reader):
    """Check if the image contains 'Macula Thickness' text in the specified region"""
    height, width = image.shape[:2]
    
    # Define the region to search for "Macula Thickness" text
    # Between top (200, 300) and bottom (620, 400)
    x = 200
    y = 300
    w = 620 - 200
    h = 400 - 300
    
    # Ensure the region is within image boundaries
    if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
        return False
    
    # Crop the region
    macula_region = image[y:y+h, x:x+w]
    
    if macula_region.size == 0:
        return False
    
    # Convert to grayscale
    macula_gray = cv2.cvtColor(macula_region, cv2.COLOR_BGR2GRAY)
    
    # Try to detect text in this region
    try:
        text_results = reader.readtext(macula_gray, detail=0)
    except TypeError:
        text_results = [result[1] for result in reader.readtext(macula_gray)]
    
    # Join all text results
    all_text = ' '.join(text_results).lower()
    print(f"Text detected in macula region: {all_text}")
    
    # Check if "Macula Thickness" (or close variant) is in the text
    if "macula thickness" in all_text or "macular thickness" in all_text:
        print("Found 'Macula Thickness' text")
        return True
    
    # If not exact match, check for close matches using fuzzy matching
    if "macula" in all_text and "thick" in all_text:
        print("Found partial 'Macula Thickness' text")
        return True
        
    # Use difflib to check for similar text (handles OCR errors)
    for text in text_results:
        similarity = difflib.SequenceMatcher(None, text.lower(), "macula thickness").ratio()
        if similarity > 0.6:  # 60% similarity threshold
            print(f"Found similar text to 'Macula Thickness': {text} (similarity: {similarity:.2f})")
            return True
    
    print("'Macula Thickness' text not found in the expected region")
    return False

def process_image(image_path, reader):
    """Process a single image to extract date and center number using EasyOCR"""
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None

    # Check for "Macula Thickness" text
    has_macula_text = check_macula_thickness(image, reader)
    if not has_macula_text:
        print(f"Invalid: No 'Macula Thickness' text found in {os.path.basename(image_path)}")
        return None

    height, width = image.shape[:2]

    # IMPORTANT: Always process images even if we can't detect numbers
    # Extract filename info first - this will be our fallback
    filename = os.path.basename(image_path)
    
    # Try to extract date from filename
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
    
    # Try to extract a number from filename as backup
    backup_number = ""
    number_match = re.search(r'_(\d{4,6})_', filename)
    if number_match:
        backup_number = number_match.group(1)
        print(f"Backup number from filename: {backup_number}")

    # --- Try multiple regions for center number ---
    # Original center region
    x_center, y_center = 1060, 560
    w_center = 1100 - 1060  # 40
    h_center = 588 - 560    # 28
    
    # Alternative regions to try (adjust as needed)
    center_regions = [
        (1060, 560, 40, 28),    # Original
        (1050, 550, 60, 48),    # Slightly larger and offset
        (1030, 540, 80, 60),    # Even larger
        (1000, 520, 120, 100),  # Much larger area
    ]
    
    center_text = ""
    
    # Try each region until we find a number
    for region in center_regions:
        x, y, w, h = region
        
        # Ensure crop coordinates are within the image boundaries
        if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
            continue  # Skip this region if out of bounds
        
        # Crop the center region
        center_crop = image[y:y+h, x:x+w]
        
        if center_crop.size == 0:
            continue  # Skip empty crops
        
        # Try multiple preprocessing methods
        results = []
        
        # Method 1: Grayscale with histogram equalization
        center_gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        center_gray = cv2.equalizeHist(center_gray)
        # Use detail=0 if available, otherwise process the full result
        try:
            gray_results = reader.readtext(center_gray, detail=0)
        except TypeError:
            # If detail parameter not supported, use default and extract text
            gray_results = [result[1] for result in reader.readtext(center_gray)]
        results.extend(gray_results)
        
        # Method 2: Binary with Otsu thresholding
        _, center_binary = cv2.threshold(center_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            binary_results = reader.readtext(center_binary, detail=0)
        except TypeError:
            binary_results = [result[1] for result in reader.readtext(center_binary)]
        results.extend(binary_results)
        
        # Method 3: Adaptive thresholding
        center_adaptive = cv2.adaptiveThreshold(
            center_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        try:
            adaptive_results = reader.readtext(center_adaptive, detail=0)
        except TypeError:
            adaptive_results = [result[1] for result in reader.readtext(center_adaptive)]
        results.extend(adaptive_results)
        
        # Process results to find any number
        for result in results:
            # Filter to just digits
            digits = ''.join(char for char in result if char.isdigit())
            if digits:
                center_text = digits
                print(f"Found center number: {center_text}")
                break
        
        if center_text:
            break  # Found a number, stop trying regions
    
    # If still no center text but we have a backup, use it
    if not center_text and backup_number:
        center_text = backup_number
        print(f"Using backup number from filename: {center_text}")
    
    # Always return results even if one field is empty
    # This ensures we don't discard images unnecessarily
    print(f"Final values - Date: '{date_text}', Number: '{center_text}'")
    
    # Only skip if BOTH date and number are missing
    if not date_text and not center_text:
        print(f"Skipping {filename} (no date or number detected).")
        return None
        
    return date_text, center_text

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
            
        date_text, center_text = ocr_result
        results.append({
            "filename": os.path.basename(image_path),
            "date": date_text,
            "number": center_text
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
    print("PROCESSING RETRY IMAGES (TEXT ONLY - NO IMAGE DISPLAY)")
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
        print("The center number is typically located around coordinates (1060, 560) to (1100, 588)")
        
        # Ask for the center number
        while True:
            center_text = input("Enter the center number (or press Enter to skip): ").strip()
            if center_text == "" or center_text.isdigit():
                break
            print("Invalid input. Please enter digits only or press Enter to skip.")
        
        if center_text == "":
            print(f"Skipping {filename} (no number entered).")
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
            "number": center_text
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
    # Initialize EasyOCR reader with English language
    print("Initializing EasyOCR reader... (this may take a moment on first run)")
    reader = easyocr.Reader(['en'], gpu=False)  # Basic initialization
    
    # Folder path containing your images
    main_folder = r"C:\Users\Markk\Downloads\TBP"
    
    print("\n" + "="*60)
    print("OCT REPORT PROCESSOR - MANUAL SORTING (EasyOCR Version)")
    print("="*60)
    print(f"Main folder: {main_folder}")
    
    # Setup the folder structure (now including retry folder)
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder = setup_folders(main_folder)
    
    print(f"Created folders: left, right, unsorted, processed, invalid, retry")
    
    # Move images to unsorted folder
    moved_count = move_unsorted_images(main_folder, unsorted_folder)
    print(f"Moved {moved_count} images to the unsorted folder")
    
    # Pre-process images to identify valid ones
    has_valid_images = pre_process_images(main_folder, unsorted_folder, invalid_folder, reader)
    
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
    left_df = process_sorted_images(left_folder, "left", reader)
    
    # Process right images
    right_df = process_sorted_images(right_folder, "right", reader)
    
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
                'Number': sorted_df['number'],
                'Filename': sorted_df['filename']
            })
            
            return final_df
            
        except Exception as e:
            print(f"Error formatting dates: {e}")
            # If date conversion fails, create a basic dataframe without date formatting
            final_df = pd.DataFrame({
                'Date': sorted_df['date'],
                'Number': sorted_df['number'],
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
