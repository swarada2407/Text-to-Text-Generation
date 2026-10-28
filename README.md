# Text-to-Text Generation

A Flask web interface for asking questions with the `google/flan-t5-base`
sequence-to-sequence model. The model is loaded the first time a question is
submitted and then reused for later requests.

## Run locally

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python app.py
   ```

4. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

The first request downloads the model from Hugging Face and may take a few
minutes. An internet connection is required unless the model is already cached.