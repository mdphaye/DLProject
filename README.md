# Attendance System using Real-Time Face Recognition 

## Overview
This project automates attendance management by leveraging real-time face recognition technology. Traditional attendance systems, whether manual or digital, often suffer from inefficiencies and inaccuracies. By integrating advanced facial recognition methods, this system offers a streamlined, contactless, and secure attendance process, ideal for educational institutions, workplaces, and other organizations.

## Features
- **Real-Time Face Detection**: Utilizes Multi-task Cascaded Convolutional Networks (MTCNN) for accurate face detection and alignment.
- **Face Recognition**: Employs FaceNet embeddings generated in PyTorch to ensure high-precision face matching.
- **Database Integration**: Stores facial embeddings and user details in a MySQL database for secure and efficient management.
- **Attendance Logging**: Automatically marks attendance upon successful face recognition.
- **User-Friendly Interface**: Displays real-time video feed with attendance status updates.

## Workflow
1. **Dataset Preparation**:
   - Collect images of individuals and preprocess them.
   - Generate facial embeddings using a pre-trained FaceNet model.
   - Store embeddings in a MySQL database with associated user details.
2. **Real-Time Recognition**:
   - Capture live video feed and detect faces.
   - Align faces and generate embeddings.
   - Compare embeddings with the database using cosine similarity.
   - Mark attendance for matched individuals.
3. **Interactive Display**:
   - Show live video feed with matched names or "No Match Found" status.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - OpenCV: For video capture and image processing.
  - Facenet-PyTorch: For MTCNN (face detection) and FaceNet (embedding generation).
  - NumPy: Numerical operations.
  - MySQL-Connector: For database integration.
  - SciPy: For calculating cosine similarity.

### Database:
- **MySQL**: Stores facial embeddings and user details.

## Results
- The system successfully detects and recognizes faces in real time.
- Attendance is accurately logged in the database.
- The interface provides clear, interactive feedback on recognition status.

## Challenges and Future Work
### Challenges:
- Variations in lighting and image quality can affect accuracy.
- Faces with extreme angles or occlusions are harder to detect.

### Future Improvements:
- Add data augmentation techniques for better performance.
- Optimize the system with a lightweight model for faster processing.
- Implement a web-based or mobile-friendly version for broader accessibility.
- A model that enables real-time user registration, automatically preprocessing images and generating embeddings for face recognition without requiring manual intervention.

## Contributors
- **Mangalya D. Phaye**
- **Ridima Garg**

