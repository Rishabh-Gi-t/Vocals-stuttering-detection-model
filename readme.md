
# VOCALS

VOCALS is an innovative application that allows users to record their voice or upload audio files in WAV format, which are then sent to the backend for analysis. The primary goal is to classify the audio as either "stuttering" or "non-stuttering" using a custom-trained model.

## Video Demonstration
https://github.com/user-attachments/assets/1d63b6ad-db5c-4066-8101-682bd8454130

## Project Overview

- **Frontend:**  
  Built with Next.js, the frontend offers an intuitive interface for users to either record their voice directly or upload pre-recorded audio files.

- **Backend:**  
  Developed using FastAPI in Python, the backend hosts a modified version of the openai-whisper-tiny ASR model. This model has been adapted into a classification model and retrained using the sep-28k dataset from Hugging Face with PyTorch. During testing, the model achieved a 92% accuracy in distinguishing between stuttering and non-stuttering audio.

- **User Feedback:**  
  The application integrates the Gemini API to provide enhanced feedback to users. Adjustments such as tweaking the temperature settings have been implemented to optimize the quality of the feedback provided.

## Technologies Used

- **Frontend:** Next.js
- **Backend:** FastAPI (Python)
- **Model Training:** PyTorch

## Model Training Code
For further details about the implementation, please refer to the code


[License](LICENSE)

