import cv2
import pytesseract
import os
import glob
import pandas as pd
import shutil
import time

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Markk\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def process_image(image_path):
    """Process a single image to extract date and center number"""
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None

    height, width = image.shape[:2]

    # --- Date and Center regions from your working code ---
    # Date region: from (850, 170) to (965, 200)
    x_date, y_date = 850, 170
    w_date = 965 - 850   # 115
    h_date = 200 - 170   # 30

    # Center region: from (1060, 560) to (1100, 588)
    x_center, y_center = 1060, 560
    w_center = 1100 - 1060  # 40
    h_center = 588 - 560    # 28

    # Ensure crop coordinates are within the image boundaries
    if x_date < 0 or y_date < 0 or (x_date + w_date) > width or (y_date + h_date) > height:
        print(f"Error: Date crop coordinates out of bounds for {image_path}")
        return None
    if x_center < 0 or y_center < 0 or (x_center + w_center) > width or (y_center + h_center) > height:
        print(f"Error: Center crop coordinates out of bounds for {image_path}")
        return None

    # Crop the date and center regions
    date_crop = image[y_date:y_date+h_date, x_date:x_date+w_date]
    center_crop = image[y_center:y_center+h_center, x_center:x_center+w_center]

    if date_crop.size == 0:
        print(f"Error: Date crop is empty for {image_path}")
        return None
    if center_crop.size == 0:
        print(f"Error: Center crop is empty for {image_path}")
        return None

    # Convert cropped regions to grayscale and apply adaptive thresholding
    date_gray = cv2.cvtColor(date_crop, cv2.COLOR_BGR2GRAY)
    center_gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)

    # Using your thresholding parameters
    date_thresh = cv2.adaptiveThreshold(date_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    center_thresh = cv2.adaptiveThreshold(center_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

    # Using your OCR configurations
    date_config = "--psm 7 -c tessedit_char_whitelist=0123456789/"
    center_config = "--psm 7 -c tessedit_char_whitelist=0123456789"

    # Run OCR on the processed images
    date_text = pytesseract.image_to_string(date_thresh, config=date_config).strip()
    center_text = pytesseract.image_to_string(center_thresh, config=center_config).strip()
    
    print(f"Date: '{date_text}', Number: '{center_text}'")
    
    # Return None if center_text is empty - this means we can't use this image
    if center_text == "":
        print(f"Skipping {os.path.basename(image_path)} (no number detected).")
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
    
    for folder in [left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder

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

def process_sorted_images(folder, side):
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
        ocr_result = process_image(image_path)
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

def pre_process_images(main_folder, unsorted_folder, invalid_folder):
    """Pre-process images to identify which ones have valid data"""
    print("\n" + "="*60)
    print("PRE-PROCESSING IMAGES")
    print("="*60)
    
    # Get list of images in the unsorted folder
    image_files = glob.glob(os.path.join(unsorted_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(unsorted_folder, "*.jpeg"))
    
    if not image_files:
        print("No images found in unsorted folder.")
        return
    
    print(f"Pre-processing {len(image_files)} images...")
    valid_count = 0
    invalid_count = 0
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"Checking: {filename}")
        
        # Process the image to see if it has valid data
        result = process_image(image_path)
        
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

def main():
    # Folder path containing your images
    main_folder = r"C:\Users\Markk\Downloads\TBP"
    
    print("\n" + "="*60)
    print("OCT REPORT PROCESSOR - MANUAL SORTING")
    print("="*60)
    print(f"Main folder: {main_folder}")
    
    # Setup the folder structure
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder = setup_folders(main_folder)
    
    print(f"Created folders: left, right, unsorted, processed, invalid")
    
    # Move images to unsorted folder
    moved_count = move_unsorted_images(main_folder, unsorted_folder)
    print(f"Moved {moved_count} images to the unsorted folder")
    
    # Pre-process images to identify valid ones
    has_valid_images = pre_process_images(main_folder, unsorted_folder, invalid_folder)
    
    if not has_valid_images:
        return
    
    print("\n" + "="*60)
    print("INSTRUCTIONS")
    print("="*60)
    print("1. Manually move VALID images from the 'unsorted' folder to either 'left' or 'right' folder")
    print("2. After sorting all images, press Enter to continue processing")
    print("="*60)
    
    input("\nPress Enter when you've finished manually sorting the images...")
    
    # Process left images
    left_df = process_sorted_images(left_folder, "left")
    
    # Process right images
    right_df = process_sorted_images(right_folder, "right")
    
    # Sort and format date before saving to Excel
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Function to standardize and sort dates
    def format_and_sort_df(df):
        if df is None or df.empty:
            return None
            
        # Create a copy to avoid modifying the original
        sorted_df = df.copy()
        
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
                
            # Drop rows with invalid dates
            sorted_df = sorted_df.dropna(subset=['datetime'])
            
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
            # Return the original dataframe if date conversion fails
            return df
    
    # Process and save left eye data
    if left_df is not None:
        formatted_left_df = format_and_sort_df(left_df)
        if formatted_left_df is not None:
            left_excel = os.path.join(main_folder, f"left_results_{timestamp}.xlsx")
            formatted_left_df.to_excel(left_excel, index=False)
            print(f"\nSaved {len(formatted_left_df)} left eye results to {left_excel}")
    
    # Process and save right eye data
    if right_df is not None:
        formatted_right_df = format_and_sort_df(right_df)
        if formatted_right_df is not None:
            right_excel = os.path.join(main_folder, f"right_results_{timestamp}.xlsx")
            formatted_right_df.to_excel(right_excel, index=False)
            print(f"Saved {len(formatted_right_df)} right eye results to {right_excel}")
    
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