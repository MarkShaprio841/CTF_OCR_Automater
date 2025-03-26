OCT Report Processor

The OCT Report Processor is a Python-based tool designed to automate the extraction and processing of Optical Coherence Tomography (OCT) reports. It leverages OCR (via EasyOCR) to detect and extract key data—such as macula thickness and report dates—from OCT images, and then sorts and processes these images into organized folders. The final output includes Excel reports summarizing the results for both left and right eye scans.

 Table of Contents
- [Overview](overview)
- [Features](features)
- [Installation](installation)
- [Usage](usage)
- [Folder Structure](folder-structure)
- [Code Breakdown](code-breakdown)
- [Dependencies](dependencies)
- [Contributing](contributing)
- [License](license)
- [Contact](contact)

 Overview

This project automates the processing of OCT images by:
- Detecting macula thickness maps: Uses OCR to search for specific keywords (e.g., "Macula Thickness" or "Central Thickness") in predefined regions of an image.
- Extracting key metrics: Identifies the central subfield thickness value and extracts a date from either the image content or filename.
- Sorting images: Automatically moves images to designated folders (left, right, unsorted, invalid, retry, or other scans) based on the extraction results.
- Generating reports: Produces Excel spreadsheets that summarize the processed data (date, thickness, and filename) for each eye.

 Features

- Automated Image Detection: Recognizes OCT scans containing macula thickness maps.
- OCR-Based Data Extraction: Uses EasyOCR to extract thickness values and dates from images.
- Flexible Sorting Options: Supports both automatic sorting (if all images are from the same eye) and manual intervention.
- Comprehensive Folder Management: Organizes images into multiple folders (e.g., `left`, `right`, `unsorted`, `invalid`, `retry`, `other_scans`, `processed`).
- Excel Reporting: Outputs sorted and formatted data into Excel files with standardized date formats.

 Installation

1. Clone the Repository:

   ```bash
   git clone https://github.com/your_username/oct-report-processor.git
   cd oct-report-processor
   ```

2. (Optional) Create and Activate a Virtual Environment:

   ```bash
   python -m venv venv
    On macOS/Linux:
   source venv/bin/activate
    On Windows:
   venv\Scripts\activate
   ```

3. Install Required Packages:

   ```bash
   pip install opencv-python easyocr pandas
   ```

   > *Note:* Ensure you have a compatible version of Python (3.6 or above) installed.

 Usage

1. Configure the Main Folder:
   - In the `main()` function of the script, update the `main_folder` variable to point to the directory containing your OCT images. For example:
     
     ```python
     main_folder = r"C:\Users\YourName\Path\To\Images"
     ```

2. Run the Script:

   ```bash
   python main.py
   ```

3. Follow On-Screen Instructions:
   - Initialization: The script initializes the EasyOCR reader (using GPU if available) and sets up the folder structure.
   - Image Sorting: It moves images from the main folder into an `unsorted` folder and pre-processes them to identify valid macula thickness maps.
   - Eye Selection: You will be prompted to indicate if all images are from the same eye (left or right) to enable auto-sorting.
   - Manual Intervention: For images that couldn’t be automatically processed, you will have the opportunity to manually sort or input missing data.
   - Report Generation: Once processing is complete, Excel reports for left and right eye images are generated.
   - Finalizing: Optionally, you can choose to move processed images to a designated folder.

 Folder Structure

Once the script runs, the following folders will be created (inside your specified main folder):

- left: Contains images sorted as left eye scans.
- right: Contains images sorted as right eye scans.
- unsorted: Images awaiting manual sorting.
- processed: Images that have been fully processed.
- invalid: OCT scans identified as invalid or with missing data.
- retry: Images that require reprocessing or manual intervention.
- other_scans: Scans that are not recognized as macula thickness maps (e.g., line raster scans).

 Code Breakdown

- `get_input_with_default(prompt, default_value)`  
  Prompts the user with a default value if no input is provided.

- `check_if_macula_thickness(image_path)`  
  Uses OCR to determine if an image contains a macula thickness map by searching for key phrases in a predefined title region and an additional check in the bottom half of the image.

- `extract_central_thickness(image_path)`  
  Extracts the central thickness value from the OCT report by scanning for keywords and nearby numerical values within the image.

- `process_image(image_path)`  
  Combines the above functions to process an individual image, extract relevant metrics (thickness and date), and determine its validity.

- `setup_folders(main_folder)`  
  Creates the necessary sub-folders for sorting and processing images.

- `move_unsorted_images(main_folder, unsorted_folder)`  
  Moves all image files from the main folder to the `unsorted` folder.

- `pre_process_images(main_folder, folders, default_eye=None)`  
  Processes the images in the `unsorted` folder, auto-sorts them based on the default eye (if provided), and handles invalid/other scan types.

- `main()`  
  The entry point for the script that initializes OCR, sets up folders, processes images, prompts for manual interventions (if necessary), and finally generates Excel reports.

 Dependencies

- Python 3.6+
- OpenCV: For image processing.
- EasyOCR: For optical character recognition.
- Pandas: For data manipulation and Excel report generation.
- Standard Libraries: `os`, `glob`, `shutil`, `time`, `re`, and `datetime` for file handling and processing.

