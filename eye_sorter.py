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

def get_input_with_default(prompt, default_value):
    """Get input with a default value"""
    response = input(f"{prompt} [{default_value}]: ").strip()
    if not response:
        return default_value
    return response

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
        if "central" in text.lower() or "thickness" in text.lower():
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
    """Process a single image to extract OCT data"""
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Check if this is a macula thickness map
    is_thickness_map = check_if_macula_thickness(image_path)
    
    if not is_thickness_map:
        print(f"Skipping {os.path.basename(image_path)} - Not a macula thickness map")
        return None, None, "other"
    
    # Extract central thickness value
    thickness_value = extract_central_thickness(image_path)
    
    if not thickness_value:
        print(f"No thickness value found in {os.path.basename(image_path)}")
        return None, None, "invalid"
    
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
    
    print(f"Success - Thickness: {thickness_value}, Date: {date_text}")
    return thickness_value, date_text, "valid"

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

def pre_process_images(main_folder, folders, default_eye=None):
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
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"Checking: {filename}")
        
        # Process the image
        thickness, date, result_type = process_image(image_path)
        
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
            
            # If we have a default eye setting, use it
            if default_eye == "left":
                dest_path = os.path.join(left_folder, filename)
                try:
                    shutil.move(image_path, dest_path)
                    auto_sorted_left += 1
                    print(f"Auto-sorted to left eye folder (based on default): {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            elif default_eye == "right":
                dest_path = os.path.join(right_folder, filename)
                try:
                    shutil.move(image_path, dest_path)
                    auto_sorted_right += 1
                    print(f"Auto-sorted to right eye folder (based on default): {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            else:
                # No default, leave in unsorted for manual sorting
                print(f"Valid thickness map, manual sorting needed: {filename}")
    
    print(f"\nPre-processing complete:")
    print(f"  Valid macula thickness maps: {valid_count}")
    if default_eye:
        print(f"    - Auto-sorted to {default_eye} eye folder: {auto_sorted_left if default_eye == 'left' else auto_sorted_right}")
    else:
        print(f"    - All valid maps require manual sorting")
    print(f"  Invalid macula thickness maps: {invalid_count}")
    print(f"  Other scan types (not macula thickness): {other_count}")
    
    return valid_count > 0

def main():
    # Initialize EasyOCR reader with English language (just once, globally)
    global READER
    print("Initializing EasyOCR reader... (this may take a moment on first run)")
    READER = easyocr.Reader(['en'], gpu=True)  # Using GPU
    
    # Folder path containing your images
    main_folder = r"C:\Users\Markk\Downloads\TBP"
    
    print("\n" + "="*60)
    print("OCT REPORT PROCESSOR - MANUAL EYE SELECTION")
    print("="*60)
    print(f"Main folder: {main_folder}")
    
    # Setup the folder structure
    folders = setup_folders(main_folder)
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, retry_folder, other_scans_folder = folders
    
    print(f"Created folders: left, right, unsorted, processed, invalid, retry, other_scans")
    
    # Ask user if all scans are for the same eye
    print("\nAre all the images from the same eye? This can speed up processing.")
    same_eye = input("Are all images from the same eye? (y/n): ").strip().lower()
    
    default_eye = None
    if same_eye == 'y':
        eye_choice = input("Which eye? (l for left, r for right): ").strip().lower()
        if eye_choice == 'l':
            default_eye = "left"
        elif eye_choice == 'r':
            default_eye = "right"
    
    # Move images to unsorted folder
    moved_count = move_unsorted_images(main_folder, unsorted_folder)
    print(f"Moved {moved_count} images to the unsorted folder")
    
    # Pre-process images with optional default eye setting
    has_valid_images = pre_process_images(main_folder, folders, default_eye)
    
    if not has_valid_images:
        return
    
    print("\n" + "="*60)
    print("INSTRUCTIONS")
    print("="*60)
    if default_eye:
        print(f"1. All valid macula thickness maps have been moved to the {default_eye} eye folder")
        print(f"2. Check the {default_eye} eye folder to make sure all images are correct")
    else:
        print("1. Manually move images from the 'unsorted' folder to either 'left' or 'right' folder")
    print("2. For images that weren't detected but should be processed, move them to the 'retry' folder")
    print("3. After sorting all images, press Enter to continue processing")
    print("="*60)
    
    input("\nPress Enter when you've finished manually sorting the images...")
    
    # Process left and right folders (these will contain the manual sorting results)
    left_results = []
    right_results = []
    
    # Process images in left folder
    left_files = glob.glob(os.path.join(left_folder, "*.[jp][pn]g")) + \
                 glob.glob(os.path.join(left_folder, "*.jpeg"))
    
    print(f"\nProcessing {len(left_files)} left eye images")
    for image_path in left_files:
        thickness, date, result_type = process_image(image_path)
        if thickness is not None:
            left_results.append({
                "filename": os.path.basename(image_path),
                "date": date,
                "thickness": thickness
            })
    
    # Process images in right folder
    right_files = glob.glob(os.path.join(right_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(right_folder, "*.jpeg"))
    
    print(f"\nProcessing {len(right_files)} right eye images")
    for image_path in right_files:
        thickness, date, result_type = process_image(image_path)
        if thickness is not None:
            right_results.append({
                "filename": os.path.basename(image_path),
                "date": date,
                "thickness": thickness
            })
    
    # Process any images in the retry folder
    retry_files = glob.glob(os.path.join(retry_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(retry_folder, "*.jpeg"))
    
    if retry_files:
        print(f"\nProcessing {len(retry_files)} retry images")
        
        # Ask if all retry images are for the same eye
        retry_same_eye = input("Are all retry images from the same eye? (y/n): ").strip().lower()
        
        retry_default_eye = None
        if retry_same_eye == 'y':
            retry_eye_choice = input("Which eye? (l for left, r for right): ").strip().lower()
            if retry_eye_choice == 'l':
                retry_default_eye = "left"
            elif retry_eye_choice == 'r':
                retry_default_eye = "right"
        
        # Process each retry image
        for image_path in retry_files:
            filename = os.path.basename(image_path)
            print(f"\nProcessing retry image: {filename}")
            
            # Try to extract thickness and date automatically
            thickness, date, result_type = process_image(image_path)
            
            # If automatic extraction failed, ask for manual input
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
            
            # Use the date if found, otherwise ask for it
            if not date:
                date = datetime.now().strftime('%m/%d/%Y')
                print(f"Using today's date: {date}")
            
            # Determine which eye if not using default
            eye_side = retry_default_eye
            if not eye_side:
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
    
    # Create dataframes from the results
    left_df = pd.DataFrame(left_results) if left_results else None
    right_df = pd.DataFrame(right_results) if right_results else None
    
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
