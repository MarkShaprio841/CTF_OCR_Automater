import cv2
import pytesseract
import os
import glob
import pandas as pd
import shutil
import time
import numpy as np
import concurrent.futures
import multiprocessing

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Markk\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def preprocess_image_for_ocr(image, is_center=False):
    """
    Apply optimized preprocessing techniques to improve OCR accuracy
    
    Args:
        image (numpy.ndarray): Input image
        is_center (bool): Flag to indicate if processing the center number
    
    Returns:
        list: List of preprocessed image variants
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresholds = []
    
    # For center numbers with colored backgrounds (red, yellow, green)
    if is_center:
        # Extract different color channels
        b, g, r = cv2.split(image)
        
        # Process HSV color space for colored backgrounds
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Special handling for red backgrounds
        # Create a mask for red color
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Invert red mask to get digits
        red_mask_inv = cv2.bitwise_not(red_mask)
        
        # Scale the image for better recognition
        scale_factor = 3
        scaled = cv2.resize(blurred, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # OPTIMIZED: Use just the most effective preprocessing methods
        thresholds = [
            # Best methods for number detection
            cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Value channel works well
            cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            
            # For red backgrounds
            red_mask_inv,
            cv2.threshold(red_mask_inv, 0, 255, cv2.THRESH_BINARY)[1],
        ]
        
        # Add specific methods for "7" digit detection
        # Sharpen the image which helps with "7" recognition
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(scaled, -1, kernel_sharpen)
        thresholds.append(cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY)[1])
        
    else:
        # For date text, use minimal preprocessing for speed
        thresholds = [
            cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        ]
    
    return thresholds

def try_ocr_with_multiple_methods(image_variants, config, is_center=False):
    """
    Try OCR on multiple preprocessed image variants - optimized for speed
    
    Args:
        image_variants (list): List of preprocessed image variants
        config (str): Tesseract configuration
        is_center (bool): Flag to indicate if processing center number
    
    Returns:
        str: Best OCR result or empty string
    """
    results = []
    
    # OPTIMIZED: Use fewer, more effective PSM modes
    psm_configs = ["--psm 7", "--psm 8"] if is_center else ["--psm 7"]
    
    for img in image_variants:
        # Standard OCR
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            if text:
                results.append(text)
        except Exception:
            pass
        
        # Try with limited PSM modes
        for psm in psm_configs:
            full_config = f"{config} {psm}"
            try:
                variant_text = pytesseract.image_to_string(img, config=full_config).strip()
                if variant_text:
                    results.append(variant_text)
            except Exception:
                pass
        
        # For center numbers only - character-by-character recognition for better accuracy
        if is_center:
            try:
                boxes = pytesseract.image_to_data(img, config=f"{config} --psm 10", output_type=pytesseract.Output.DICT)
                digit_text = ''.join([boxes['text'][i] for i in range(len(boxes['text'])) 
                                     if boxes['text'][i].strip() and boxes['text'][i].isdigit()])
                if digit_text:
                    results.append(digit_text)
            except Exception:
                pass
    
    # Filter and clean results
    cleaned_results = []
    for result in results:
        # Remove any non-numeric characters
        cleaned = ''.join(char for char in result if char.isdigit())
        if cleaned:
            if is_center:
                # Knowing numbers are 3 digits helps with accuracy
                if len(cleaned) >= 3:
                    cleaned_results.append(cleaned[:3])
                elif 1 <= len(cleaned) < 3:
                    cleaned_results.append(cleaned.zfill(3))
            else:
                cleaned_results.append(cleaned)
    
    # Return the most common result, or empty string
    if cleaned_results:
        return max(set(cleaned_results), key=cleaned_results.count)
    
    return ""

def check_macula_thickness(image):
    """
    Check if the image has 'Macula Thickness' text in the specified region
    Returns a confidence score between 0-1
    """
    height, width = image.shape[:2]
    
    # Coordinates for the Macula Thickness text region
    x, y = 211, 316
    w, h = 555 - 211, 367 - 316
    
    # Validate coordinates
    if x < 0 or y < 0 or x + w > width or y + h > height:
        print("Macula Thickness region out of bounds, can't verify image type")
        return 0.0
    
    # Crop the region
    crop = image[y:y+h, x:x+w]
    
    if crop.size == 0:
        return 0.0
    
    # Convert to grayscale and threshold for better text detection
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Detect text
    config = "--psm 7"  # Treat as a single line of text
    text = pytesseract.image_to_string(thresh, config=config).strip().lower()
    
    # Calculate confidence based on word matching
    target_words = ["macula", "thickness"]
    detected_words = text.lower().split()
    
    # Count matches
    matches = sum(word in detected_words for word in target_words)
    
    # Calculate confidence (0.5 per matched word)
    confidence = matches / len(target_words)
    
    # Debug
    print(f"Macula detection: {text} (Confidence: {confidence:.2f})")
    
    return confidence

def process_image(image_path):
    """Process a single image to extract date and center number"""
    filename = os.path.basename(image_path)
    print(f"\nProcessing: {filename}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {filename}")
        return None

    # Check if the image has "Macula Thickness" text
    confidence = check_macula_thickness(image)
    if confidence < 0.5:  # Less than half the words match
        print(f"Skipping {filename} - Not a Macula Thickness image (confidence: {confidence:.2f})")
        return None

    height, width = image.shape[:2]
    
    # Define multiple potential regions to check
    # Primary regions with UPDATED center coordinates from user
    regions = [
        # Primary date region
        {"name": "date_primary", "x": 850, "y": 170, "w": 115, "h": 30},
        # Primary center number region - UPDATED per user coordinates
        {"name": "center_primary", "x": 1040, "y": 534, "w": 63, "h": 69},
    ]
    
    # Add alternative regions to try
    if width > 1200:  # For high-resolution images
        regions.extend([
            # Alternative positions for different image types
            {"name": "date_alt1", "x": int(width*0.7), "y": int(height*0.1), "w": 150, "h": 40},
            {"name": "center_alt1", "x": int(width*0.85), "y": int(height*0.56), "w": 60, "h": 30},
        ])
    
    date_text = ""
    center_text = ""
    
    # Check debug folder
    debug_folder = os.path.join(os.path.dirname(image_path), "debug")
    os.makedirs(debug_folder, exist_ok=True)
    
    # Store crops for debugging
    crops = {}
    
    # Try each region
    for region in regions:
        x, y = region["x"], region["y"]
        w, h = region["w"], region["h"]
        
        # Validate coordinates
        if x < 0 or y < 0 or x + w > width or y + h > height:
            print(f"Region {region['name']} out of bounds, skipping")
            continue
        
        # Crop the region
        crop = image[y:y+h, x:x+w]
        
        if crop.size == 0:
            print(f"Empty crop for {region['name']}, skipping")
            continue
        
        # Save crop for debugging
        crops[region["name"]] = crop
        
        # Process the crop
        is_center = "center" in region["name"]
        
        # Only attempt OCR if we haven't already found text for this type
        if is_center and not center_text:
            # Add a small border
            crop_with_border = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # Preprocess
            variants = preprocess_image_for_ocr(crop_with_border, is_center=True)
            # OCR
            config = "-c tessedit_char_whitelist=0123456789"
            text = try_ocr_with_multiple_methods(variants, config, is_center=True)
            if text:
                center_text = text
                print(f"Found center number '{center_text}' in {region['name']} for file {filename}")
        
        elif not is_center and not date_text:
            # Preprocess
            variants = preprocess_image_for_ocr(crop, is_center=False)
            # OCR
            config = "-c tessedit_char_whitelist=0123456789/"
            text = try_ocr_with_multiple_methods(variants, config, is_center=False)
            if text:
                date_text = text
                print(f"Found date '{date_text}' in {region['name']} for file {filename}")
        
        # If we have both, we can stop
        if date_text and center_text:
            break
    
    # Save all crops for debugging with a more descriptive filename
    base_filename = os.path.splitext(filename)[0]
    for region_name, crop in crops.items():
        debug_filename = f"{base_filename}_{region_name}"
        if region_name == "center_primary":
            debug_filename += f"_detected_{center_text}"
        cv2.imwrite(os.path.join(debug_folder, f"{debug_filename}.png"), crop)
    
    # Also save the Macula Thickness region for verification
    macula_region = image[316:367, 211:555]
    cv2.imwrite(os.path.join(debug_folder, f"{base_filename}_macula_region.png"), macula_region)
    
    # Add more detailed final output
    print(f"FINAL RESULTS for {filename}:")
    print(f"  - Date: {date_text or 'Not detected'}")
    print(f"  - Center Number: {center_text or 'Not detected'}")
    
    # Return what we found - require at least one to be valid
    if date_text or center_text:
        return date_text, center_text
    
    print(f"No data found in {filename}")
    return None
def setup_folders(main_folder):
    """Create necessary folders for manual sorting"""
    # Create all required folders if they don't exist
    left_folder = os.path.join(main_folder, "left")
    right_folder = os.path.join(main_folder, "right")
    unsorted_folder = os.path.join(main_folder, "unsorted")
    processed_folder = os.path.join(main_folder, "processed")
    invalid_folder = os.path.join(main_folder, "invalid")
    uncertain_folder = os.path.join(main_folder, "uncertain")  # New folder for uncertain cases
    
    for folder in [left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, uncertain_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, uncertain_folder

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

def pre_process_images(main_folder, unsorted_folder, invalid_folder, uncertain_folder):
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
    uncertain_count = 0
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"Checking: {filename}")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image: {filename}")
            # Move to invalid folder
            shutil.move(image_path, os.path.join(invalid_folder, filename))
            invalid_count += 1
            continue
            
        # Check if it's a Macula Thickness image
        confidence = check_macula_thickness(image)
        
        if confidence >= 0.5:  # More than half the words match
            # It's a Macula Thickness image - keep it in unsorted
            valid_count += 1
            print(f"Valid Macula Thickness image: {filename}")
        elif confidence > 0:  # Some words match but not enough
            # Move to uncertain folder
            shutil.move(image_path, os.path.join(uncertain_folder, filename))
            uncertain_count += 1
            print(f"Uncertain Macula image: {filename}")
        else:  # No words match
            # Move to invalid folder
            shutil.move(image_path, os.path.join(invalid_folder, filename))
            invalid_count += 1
            print(f"Not a Macula image: {filename}")
    
    print(f"\nPre-processing complete: {valid_count} valid images, {uncertain_count} uncertain images, {invalid_count} invalid images")
    print("- Invalid images have been moved to the 'invalid' folder")
    print("- Uncertain images have been moved to the 'uncertain' folder for manual review")
    
    if valid_count == 0:
        print("No valid images found. Exiting process.")
        return False
    
    return True

def pre_process_images_parallel(main_folder, unsorted_folder, invalid_folder, uncertain_folder):
    """Pre-process images in parallel to identify which ones have valid data"""
    print("\n" + "="*60)
    print("PRE-PROCESSING IMAGES (PARALLEL)")
    print("="*60)
    
    # Get list of images in the unsorted folder
    image_files = glob.glob(os.path.join(unsorted_folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(unsorted_folder, "*.jpeg"))
    
    if not image_files:
        print("No images found in unsorted folder.")
        return False
    
    print(f"Pre-processing {len(image_files)} images in parallel...")
    
    # Process all images to check if they have valid data
    valid_count = 0
    invalid_count = 0
    uncertain_count = 0
    
    # Use more cores but process fewer images at once to avoid overloading
    cpu_count = multiprocessing.cpu_count()
    max_workers = max(1, int(cpu_count * 0.85))  # 85% of cores for i9
    batch_size = 20
    
    total_count = len(image_files)
    processed_count = 0
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, total_count, batch_size):
        end_idx = min(start_idx + batch_size, total_count)
        current_batch = image_files[start_idx:end_idx]
        batch_size_actual = len(current_batch)
        
        print(f"\nProcessing batch {start_idx//batch_size + 1}/{(total_count + batch_size - 1)//batch_size} ({batch_size_actual} images)")
        
        # Use a more efficient number of workers for the batch
        num_workers = min(max_workers, batch_size_actual)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all image loading tasks
            future_to_path = {}
            for path in current_batch:
                future = executor.submit(lambda p: (p, cv2.imread(p)), path)
                future_to_path[future] = path
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                filename = os.path.basename(path)
                processed_count += 1
                
                try:
                    _, image = future.result()
                    if image is None:
                        print(f"[{processed_count}/{total_count}] Could not load image: {filename}")
                        # Move to invalid folder
                        shutil.move(path, os.path.join(invalid_folder, filename))
                        invalid_count += 1
                        continue
                    
                    # Check if it's a Macula Thickness image
                    confidence = check_macula_thickness(image)
                    
                    if confidence >= 0.5:  # More than half the words match
                        # It's a Macula Thickness image - process it
                        valid_count += 1
                        print(f"[{processed_count}/{total_count}] Valid Macula Thickness image: {filename}")
                    elif confidence > 0:  # Some words match but not enough
                        # Move to uncertain folder
                        shutil.move(path, os.path.join(uncertain_folder, filename))
                        uncertain_count += 1
                        print(f"[{processed_count}/{total_count}] Uncertain Macula image: {filename}")
                    else:  # No words match
                        # Move to invalid folder
                        shutil.move(path, os.path.join(invalid_folder, filename))
                        invalid_count += 1
                        print(f"[{processed_count}/{total_count}] Not a Macula image: {filename}")
                    
                except Exception as e:
                    print(f"[{processed_count}/{total_count}] Error processing {filename}: {e}")
                    # Move to invalid folder on error
                    try:
                        shutil.move(path, os.path.join(invalid_folder, filename))
                        invalid_count += 1
                    except Exception as move_err:
                        print(f"Error moving file: {move_err}")
    
    print(f"\nPre-processing complete: {valid_count} valid images, {uncertain_count} uncertain images, {invalid_count} invalid images")
    print("- Invalid images have been moved to the 'invalid' folder")
    print("- Uncertain images have been moved to the 'uncertain' folder for manual review")
    
    if valid_count == 0:
        print("No valid images found. Exiting process.")
        return False
    
    return True

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

def process_images_parallel(image_paths, side=None):
    """Process multiple images in parallel using multiprocessing"""
    # For GPU acceleration - optimized for RTX 4070
    try:
        # Enable OpenCV GPU acceleration if available
        cv2.setUseOptimized(True)
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("GPU acceleration enabled for OpenCV (RTX 4070 detected)")
    except Exception as e:
        print(f"GPU acceleration not available for OpenCV: {e}")
    
    # OPTIMIZED: Use more cores but process fewer images at once to avoid overloading
    cpu_count = multiprocessing.cpu_count()
    # For i9, we can use more cores, but leave some for system tasks
    max_workers = max(1, int(cpu_count * 0.85))  # 85% of cores for i9
    
    # Limit batch size for better memory management
    batch_size = 20
    
    results = []
    processed_count = 0
    total_count = len(image_paths)
    
    print(f"Processing {total_count} images in batches of {batch_size} using {max_workers} CPU cores")
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, total_count, batch_size):
        end_idx = min(start_idx + batch_size, total_count)
        current_batch = image_paths[start_idx:end_idx]
        batch_size_actual = len(current_batch)
        
        print(f"\nProcessing batch {start_idx//batch_size + 1}/{(total_count + batch_size - 1)//batch_size} ({batch_size_actual} images)")
        
        # Use a more efficient number of workers for the batch
        num_workers = min(max_workers, batch_size_actual)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks for this batch
            future_to_path = {executor.submit(process_image, path): path for path in current_batch}
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                filename = os.path.basename(path)
                processed_count += 1
                
                try:
                    ocr_result = future.result()
                    if ocr_result is not None:
                        date_text, center_text = ocr_result
                        results.append({
                            "filename": filename,
                            "date": date_text or "",  # Handle None values
                            "number": center_text or ""
                        })
                        print(f"[{processed_count}/{total_count}] Success: {filename} - Date: {date_text or 'N/A'}, Number: {center_text or 'N/A'}")
                    else:
                        print(f"[{processed_count}/{total_count}] No valid data: {filename}")
                except Exception as e:
                    print(f"[{processed_count}/{total_count}] Error processing {filename}: {e}")
    
    if results:
        return pd.DataFrame(results)
    else:
        if side:
            print(f"No valid results from {side} eye images.")
        else:
            print("No valid results from images.")
        return None

def process_sorted_images_parallel(folder, side):
    """Process images in the left or right folder using parallel processing"""
    # Get list of images in the folder
    image_files = glob.glob(os.path.join(folder, "*.[jp][pn]g")) + \
                  glob.glob(os.path.join(folder, "*.jpeg"))
                  
    if not image_files:
        print(f"No image files found in {side} folder")
        return None
        
    print(f"\nProcessing {len(image_files)} {side} eye images in parallel")
    return process_images_parallel(image_files, side)

def format_and_sort_df(df):
    """Standardize and sort dates with improved handling of unusual formats"""
    if df is None or df.empty:
        return None
        
    # Create a copy to avoid modifying the original
    sorted_df = df.copy()
    
    # Keep the original date string for fallback
    sorted_df['original_date'] = sorted_df['date']
    
    # Format dates - this needs to handle unusual date formats like "1031202"
    try:
        # First, try to normalize various date formats by adding separators
        cleaned_dates = []
        for date_str in sorted_df['date']:
            if pd.isna(date_str) or date_str == "":
                cleaned_dates.append(None)
                continue
                
            # Remove any existing separators
            digits_only = ''.join(c for c in date_str if c.isdigit())
            
            # Try to intelligently add separators based on length
            if len(digits_only) >= 8:  # Likely MMDDYYYY or DDMMYYYY or YYYYMMDD
                if digits_only.startswith('20'):  # Likely YYYYMMDD
                    formatted = f"{digits_only[:4]}-{digits_only[4:6]}-{digits_only[6:8]}"
                else:  # Likely MMDDYYYY or DDMMYYYY
                    formatted = f"{digits_only[:2]}-{digits_only[2:4]}-{digits_only[4:8]}"
            elif len(digits_only) == 7:  # Likely missing a leading 0
                formatted = f"0{digits_only[:1]}-{digits_only[1:3]}-{digits_only[3:7]}"
            elif len(digits_only) == 6:  # Likely MMDDYY
                formatted = f"{digits_only[:2]}-{digits_only[2:4]}-{digits_only[4:6]}"
            else:
                # For other lengths, do our best guess
                formatted = digits_only
                
            cleaned_dates.append(formatted)
            
        sorted_df['clean_date'] = cleaned_dates
        
        # Try multiple date formats with robust error handling
        date_formats = ['%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d', '%m-%d-%y']
        
        # Try each format, keeping the first one that works for most dates
        best_format = None
        best_success_count = 0
        
        for date_format in date_formats:
            try:
                # For each format, try to convert all dates
                success_count = 0
                temp_dates = []
                
                for date_str in sorted_df['clean_date']:
                    try:
                        if pd.isna(date_str):
                            temp_dates.append(None)
                        else:
                            temp_date = pd.to_datetime(date_str, format=date_format)
                            temp_dates.append(temp_date)
                            success_count += 1
                    except:
                        temp_dates.append(None)
                
                if success_count > best_success_count:
                    best_format = date_format
                    best_success_count = success_count
                    
                    if best_success_count == len(sorted_df):
                        break  # If all dates parsed successfully, no need to try more formats
                
            except:
                continue
        
        if best_format:
            print(f"Best date format detected: {best_format} (parsed {best_success_count}/{len(sorted_df)} dates)")
            
            # Now parse all dates with the best format, with fallback to flexible parsing
            parsed_dates = []
            for date_str in sorted_df['clean_date']:
                try:
                    if pd.isna(date_str):
                        parsed_dates.append(None)
                    else:
                        parsed_date = pd.to_datetime(date_str, format=best_format)
                        parsed_dates.append(parsed_date)
                except:
                    try:
                        # Fallback to flexible parsing
                        parsed_date = pd.to_datetime(date_str, errors='coerce')
                        parsed_dates.append(parsed_date)
                    except:
                        parsed_dates.append(None)
                        
            sorted_df['datetime'] = parsed_dates
        else:
            # If no format worked well, use flexible parsing
            print("No single date format worked well, using flexible parsing")
            sorted_df['datetime'] = pd.to_datetime(sorted_df['clean_date'], errors='coerce')
            
        # IMPORTANT: Don't drop rows with invalid dates - instead use the original string
        sorted_df['formatted_date'] = sorted_df.apply(
            lambda row: row['datetime'].strftime('%d-%b-%y') if pd.notnull(row['datetime']) else row['original_date'],
            axis=1
        )
        
        # Sort by date when possible - put None dates at the end
        sorted_df['sort_key'] = sorted_df['datetime'].apply(lambda x: pd.Timestamp.max if pd.isna(x) else x)
        sorted_df = sorted_df.sort_values('sort_key')
        
        # Create final formatted dataframe, including ALL rows
        final_df = pd.DataFrame({
            'Date': sorted_df['formatted_date'],
            'Number': sorted_df['number'],
            'Filename': sorted_df['filename']
        })
        
        return final_df
        
    except Exception as e:
        print(f"Error formatting dates: {e}")
        # Return a basic dataframe with the original values if date conversion fails completely
        return pd.DataFrame({
            'Date': sorted_df['date'],
            'Number': sorted_df['number'],
            'Filename': sorted_df['filename']
        })

def main():
    # Folder path containing your images
    main_folder = r"C:\Users\Markk\Downloads\TBP"
    
    print("\n" + "="*60)
    print("OCT REPORT PROCESSOR - MANUAL SORTING (OPTIMIZED)")
    print("="*60)
    print(f"Main folder: {main_folder}")
    
    # Setup the folder structure - including the new uncertain folder
    left_folder, right_folder, unsorted_folder, processed_folder, invalid_folder, uncertain_folder = setup_folders(main_folder)
    
    print(f"Created folders: left, right, unsorted, processed, invalid, uncertain")
    
    # Move images to unsorted folder
    moved_count = move_unsorted_images(main_folder, unsorted_folder)
    print(f"Moved {moved_count} images to the unsorted folder")
    
    # Pre-process images to identify valid ones, uncertain ones, and invalid ones
    parallel_processing = True
    if parallel_processing:
        has_valid_images = pre_process_images_parallel(main_folder, unsorted_folder, invalid_folder, uncertain_folder)
    else:
        # Fallback to non-parallel processing if needed
        print("Using single-threaded processing")
        has_valid_images = pre_process_images(main_folder, unsorted_folder, invalid_folder, uncertain_folder)
    
    if not has_valid_images:
        return
    
    print("\n" + "="*60)
    print("INSTRUCTIONS")
    print("="*60)
    print("1. Check the 'uncertain' folder and move Macula Thickness images to the unsorted folder")
    print("2. Manually move VALID images from the 'unsorted' folder to either 'left' or 'right' folder")
    print("3. After sorting all images, press Enter to continue processing")
    print("="*60)
    
    input("\nPress Enter when you've finished manually sorting the images...")
    
    # Process left and right images
    if parallel_processing:
        left_df = process_sorted_images_parallel(left_folder, "left")
        right_df = process_sorted_images_parallel(right_folder, "right")
    else:
        left_df = process_sorted_images(left_folder, "left")
        right_df = process_sorted_images(right_folder, "right")
    
    # Sort and format date before saving to Excel
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
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