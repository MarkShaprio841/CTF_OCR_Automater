OCT Report Processor

This project is a Python script that processes Optical Coherence Tomography (OCT) report images to extract the Central Subfield Thickness value. It uses computer vision and optical character recognition (OCR) to analyze images, sort them into folders (e.g., left eye, right eye, unsorted, invalid), and export the results to Excel files.

 Features

- OCT Report Detection: Quickly checks if an image is likely an OCT report by analyzing colors (pink/green in the ETDRS grid) and layout.
- OCR-Based Extraction: Uses EasyOCR to extract the central thickness value (in µm) from a targeted region of the image.
- Automatic & Manual Sorting: Pre-processes images by moving invalid ones to a separate folder. Allows you to manually sort valid images into left/right folders and to retry processing images that were not automatically handled.
- Excel Export: Compiles the extracted data (date, thickness value, and filename) into Excel files, formatted and sorted by date.
- Folder Management: Automatically creates and manages subfolders for left, right, unsorted, invalid, processed, and retry images.

 Dependencies

Before running the script, make sure you have the following Python packages installed:

- [OpenCV](https://opencv.org/) (`opencv-python`)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Pandas](https://pandas.pydata.org/)

Other modules such as `os`, `glob`, `shutil`, `time`, `re`, and `datetime` are part of the standard Python library.

 Installation

1. Clone the Repository or Download the Script

   Clone the repository or simply download the Python script file.

2. Install Python Dependencies

   It is recommended to use a virtual environment. Then, install the required packages via pip:

   ```bash
   pip install opencv-python easyocr pandas
   ```

   If you do not have pip, please install it or refer to the [Python Packaging User Guide](https://packaging.python.org/tutorials/installing-packages/).

3. Verify Installation

   Make sure that Python is installed by running:

   ```bash
   python --version
   ```

 Usage

1. Prepare Your Images

   - Place all OCT report images (supported formats: JPG, JPEG, PNG) in your main folder (e.g., `C:\Users\Markk\Downloads\TBP`).
   - The script will move these images to an `unsorted` folder inside the main folder.

2. Folder Structure

   Upon execution, the script creates the following subfolders inside the main folder:

   - `left` — For images identified as left eye reports.
   - `right` — For images identified as right eye reports.
   - `unsorted` — Where the script initially moves all images.
   - `invalid` — For images that are not valid OCT reports.
   - `retry` — For images that may need manual reprocessing.
   - `processed` — Where images will be moved after processing.

3. Run the Script

   Execute the script with:

   ```bash
   python your_script_name.py
   ```

   During execution, the following happens:
   
   - Initialization: The EasyOCR reader is initialized.
   - Pre-Processing: Images are checked for valid OCT reports and moved to appropriate folders.
   - Manual Sorting: You will be prompted to move valid images from the `unsorted` folder to either the `left` or `right` folder. For images that weren’t automatically processed, move them to the `retry` folder.
   - Data Extraction: Once you press Enter after sorting, the script processes the left and right eye images, extracts the central thickness values, and compiles the results.
   - Excel Export: The processed data is saved as Excel files (`left_results_TIMESTAMP.xlsx` and `right_results_TIMESTAMP.xlsx`) in the main folder.
   - Processed Images: You are given an option to move all processed images into the `processed` folder.

4. Manual Retry

   If some images could not be processed automatically, they are handled in the `retry` step. The script prompts you to manually enter the thickness value and to designate whether the image is from the left or right eye.

 How It Works

1. Image Verification:  
   The function `quick_check_macula_thickness()` uses HSV color thresholds to detect characteristic pink/green hues and the presence of grid patterns (via edge detection and Hough Circles) to verify that an image is likely an OCT report.

2. Data Extraction:  
   The `extract_central_thickness()` function focuses on a specific section (typically the bottom right corner) where the “Central Subfield Thickness” is expected. It applies OCR to this region, checks for keywords, and extracts plausible numerical values within a realistic range.

3. Folder and File Management:  
   Several helper functions manage image movement:
   - `setup_folders()` creates necessary subdirectories.
   - `move_unsorted_images()` relocates all images into the `unsorted` folder.
   - `pre_process_images()` scans images, validates them, and moves invalid images to an `invalid` folder.
   - `process_sorted_images()` and `process_retry_images()` handle sorted and manually retried images, respectively.

4. Final Processing and Export:  
   The main function coordinates initialization, processing, manual intervention, and finally exports the results to Excel. It also handles date formatting and sorting.

 Troubleshooting

- EasyOCR Initialization:  
  The first run may take some time to initialize the OCR engine. Subsequent runs should be faster.

- Image Loading Errors:  
  Ensure that your images are in supported formats (JPG, JPEG, PNG) and that they are not corrupted.

- Folder Permissions:  
  Verify that the script has permission to create and move files within the designated main folder.

