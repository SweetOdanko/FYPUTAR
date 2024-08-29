Intelligent Image Search Engine with Similarity Detection

Step 1: Set Up Environment

1. Create a virtual environment
python -m venv environment

2. Activate the virtual environment (Windows)
environment/Scripts/activate

3. Install the required dependencies
pip install -r requirements.txt

4. Install CUDA-specific packages
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Step 2: Populate the Database

1. Start the Flask app
python app.py

2. Open Postman (or any API tool)

3. Create a new POST request to
http://127.0.0.1:5000/api/upload_zip

4. Set Up the Request
Method: POST
Body:	-Select 'form-data'
	-Add keys and values
	 file	
	 Example value: train.zip
	 db_url
	 Example value: mongodb://localhost:27017/

5. Send the request
Images saved to the database will use to retrieve as similar images result.


Step 3: Run the Application

1. Make sure MongoDB is running

2. Start the Flask app
python app.py




